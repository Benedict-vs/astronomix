"""Fused per-cell stiff solve of the chemistry on the GPU (NVIDIA Warp), for the stiff-core hybrid (2026-10-12).

The implicit solve of the cores is a batch of independent 0-D stiff ODEs. Vmapping diffrax over the batch runs every
cell for the step count of the stiffest one; a kernel with one thread per cell lets every cell take its own adaptive
steps. gradsolve (ECLIPSE-AI4Science, MIT) provides the pieces: a jaxpr -> Warp translator that turns the JAX
right-hand side into a ``@wp.func`` field with its analytic Jacobian, and an adaptive Rosenbrock23 integrator written
as a Warp kernel. This module (i) extends the translator to the primitives of the carbox network (closed-over
constants, dot products, gathers/scatters with constant indices, reductions, ``where``), (ii) builds a kernel with a
per-cell step length, and (iii) wraps it with ``warp.jax_kernel`` so that it is an ordinary JAX primitive inside the
jitted (and sharded) time step. On 65,536 core cells of the 256^3 reference at rtol 1e-6 / atol 1e-15 the fused solve
agrees with diffrax Kvaerno5 to 1e-5 dex and is 9-13x faster on an A100 (CONTEXT.md 2026-10-12).

Usage (at set-up time, with CONCRETE chemistry parameters; the translation closes over them):
    fn = build_fused_core_solver(vector_field, dim, rtol=1e-6, atol=1e-15, max_steps=2048)
    y1 = fn(y0, dt)      # y0 [n, dim] float64 (abundances [cm^-3], T [K] last when thermochemistry), dt [n] seconds
"""
import functools
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
# Translator extension (gradsolve.warp.jax_field covers the primitives of its built-in problems only)
# ---------------------------------------------------------------------------------------------------------------------
def _extend_translator(JF):
    """Add the handlers the carbox right-hand side needs to gradsolve's jaxpr -> Warp emitter (idempotent)."""
    if getattr(JF, "_astronomix_extended", False):
        return
    import jax, jax.numpy as jnp
    E = JF._Emitter

    def _num_arr(val):
        shape, comps = val
        if not all(JF._is_num(c) for c in comps):
            raise JF.UnsupportedRHS("index", "gather/scatter/dot indices must be constants")
        return np.array([c for c in comps]).reshape(shape)

    def _fold(e, op, items):
        acc = None
        for c in items:
            acc = c if acc is None else e._scalar_binop(op, acc, c)
        return 0.0 if acc is None else acc

    def _broadcast_pair(self, a, b):
        (sa, ca), (sb, cb) = a, b
        na, nb = len(ca), len(cb)
        if tuple(sa) == tuple(sb):
            return sa, ca, cb
        if na == 1 or nb == 1:
            if na == nb:
                return (sa if len(sa) >= len(sb) else sb), ca, cb
            return (sb, ca * nb, cb) if na == 1 else (sa, ca, cb * na)
        # different shapes, also when the element counts agree: (n, 1) against (n,) is an outer product, not an element-wise
        # pair (2026-11-16: gradsolve 0.2.1 pairs equal counts element-wise, which gave a wrong Jacobian whenever a scalar
        # function of the state multiplied a vector as long as the state)
        A = np.array(ca, dtype=object).reshape(sa); B = np.array(cb, dtype=object).reshape(sb)
        try:
            A2, B2 = np.broadcast_arrays(A, B)
        except ValueError:
            raise JF.UnsupportedRHS("broadcast", f"cannot broadcast shapes {sa} and {sb}")
        return tuple(A2.shape), list(A2.reshape(-1)), list(B2.reshape(-1))
    E._broadcast_pair = _broadcast_pair

    def _h_select_n(e, iv, p):
        pred = iv[0]; cases = iv[1:]
        if len(cases) != 2:
            raise JF.UnsupportedRHS("select_n", "only two-way select supported")
        arrs = [np.array(v[1], dtype=object).reshape(v[0]) for v in [pred] + cases]
        P_, A_, B_ = np.broadcast_arrays(*arrs)
        out = []
        for c, a, b in zip(P_.reshape(-1), A_.reshape(-1), B_.reshape(-1)):   # case 0 if pred is False
            if JF._is_num(c): out.append(b if c else a)
            elif JF._is_num(a) and JF._is_num(b) and a == b: out.append(a)
            else: out.append(e.emit("wp.where(%s, %s, %s)" % (c, e._src(b), e._src(a))))
        return (tuple(P_.shape), out)

    def _h_logic(op):
        tmpl = {"and": "({a} and {b})", "or": "({a} or {b})"}[op]
        fold = {"and": lambda a, b: bool(a) and bool(b), "or": lambda a, b: bool(a) or bool(b)}[op]
        def h(e, iv, p):
            shape, ca, cb = e._broadcast_pair(iv[0], iv[1]); out = []
            for a, b in zip(ca, cb):
                if JF._is_num(a) and JF._is_num(b): out.append(fold(a, b))
                elif JF._is_num(a): out.append(b if (a if op == "and" else not a) else (False if op == "and" else True))
                elif JF._is_num(b): out.append(a if (b if op == "and" else not b) else (False if op == "and" else True))
                else: out.append(e.emit(tmpl.format(a=a, b=b)))
            return (shape, out)
        return h

    def _h_not(e, iv, p):
        shape, comps = iv[0]
        return (shape, [(not c) if JF._is_num(c) else e.emit("(not %s)" % c) for c in comps])

    def _h_reduce(op):
        def h(e, iv, p):
            arr = e._as_obj_array(iv[0]); axes = tuple(p["axes"]); keep = [i for i in range(arr.ndim) if i not in axes]
            moved = np.transpose(arr, keep + list(axes)).reshape([arr.shape[i] for i in keep] + [-1]).reshape(-1, max(1, int(np.prod([arr.shape[a] for a in axes]))))
            return (tuple(arr.shape[i] for i in keep), [_fold(e, op, list(row)) for row in moved])
        return h

    def _h_dot_general(e, iv, p):
        a = e._as_obj_array(iv[0]); b = e._as_obj_array(iv[1]); (lc, rc), (lb, rb) = p["dimension_numbers"]
        lf = [i for i in range(a.ndim) if i not in lc and i not in lb]; rf = [i for i in range(b.ndim) if i not in rc and i not in rb]
        out_shape = tuple([a.shape[i] for i in lb] + [a.shape[i] for i in lf] + [b.shape[i] for i in rf]); out = []
        for bi in np.ndindex(*[a.shape[i] for i in lb]):
            for li in np.ndindex(*[a.shape[i] for i in lf]):
                for ri in np.ndindex(*[b.shape[i] for i in rf]):
                    terms = []
                    for ci in np.ndindex(*[a.shape[i] for i in lc]):
                        ia = [None] * a.ndim; ib = [None] * b.ndim
                        for k, i in enumerate(lb): ia[i] = bi[k]; ib[rb[k]] = bi[k]
                        for k, i in enumerate(lf): ia[i] = li[k]
                        for k, i in enumerate(rf): ib[i] = ri[k]
                        for k, i in enumerate(lc): ia[i] = ci[k]; ib[rc[k]] = ci[k]
                        terms.append(e._scalar_binop("mul", a[tuple(ia)], b[tuple(ib)]))
                    out.append(_fold(e, "add", terms))
        return (out_shape, out)

    def _h_gather(e, iv, p):
        operand = e._as_obj_array(iv[0]); idx = _num_arr(iv[1]).astype(np.int64)
        q = dict(p); fill = p.get("fill_value", None)
        if "fill_value" in q: q["fill_value"] = -1
        grid = jnp.arange(operand.size, dtype=jnp.int32).reshape(operand.shape)
        src = np.asarray(jax.lax.gather_p.bind(grid, jnp.asarray(idx, jnp.int32), **q))
        flat = operand.reshape(-1); fillv = float(fill) if fill is not None else float("nan")
        return (tuple(src.shape), [flat[int(i)] if int(i) >= 0 else fillv for i in src.reshape(-1)])

    def _h_scatter(combine=None):
        def h(e, iv, p):
            operand = e._as_obj_array(iv[0]); idx = jnp.asarray(_num_arr(iv[1]).astype(np.int64), jnp.int32); upd = e._as_obj_array(iv[2])
            of = operand.reshape(-1); uf = upd.reshape(-1)
            kw = dict(indices_are_sorted=p.get("indices_are_sorted", False), mode=p.get("mode", None))
            if combine is None:   # overwrite: one update per slot (the winner among repeated indices is whichever JAX picks)
                base = jnp.full(operand.shape, -1, jnp.int32); tags = jnp.arange(upd.size, dtype=jnp.int32).reshape(upd.shape)
                res = np.asarray(jax.lax.scatter(base, idx, tags, p["dimension_numbers"], unique_indices=p.get("unique_indices", False), **kw))
                return (tuple(operand.shape), [of[k] if tag < 0 else uf[int(tag)] for k, tag in enumerate(res.reshape(-1))])
            # accumulate: with repeated indices a slot receives several updates and all of them are combined, in the order of the
            # updates (2026-11-16: a single tag scatter kept one update per slot and silently dropped the others)
            marks = jnp.eye(upd.size, dtype=jnp.int32).reshape((upd.size,) + tuple(upd.shape))
            hit = np.asarray(jax.vmap(lambda mark: jax.lax.scatter_add(jnp.zeros(operand.shape, jnp.int32), idx, mark, p["dimension_numbers"], unique_indices=False, **kw))(marks))
            hit = hit.reshape(upd.size, -1); out = []
            for k in range(of.size):
                acc = of[k]
                for j in np.flatnonzero(hit[:, k]):
                    acc = e._scalar_binop(combine, acc, uf[int(j)])
                out.append(acc)
            return (tuple(operand.shape), out)
        return h

    def _h_reshape(e, iv, p):
        arr = e._as_obj_array(iv[0]); dims = p.get("dimensions")
        if dims is not None: arr = np.transpose(arr, tuple(int(d) for d in dims))
        return e._from_obj_array(arr.reshape(tuple(int(s) for s in p["new_sizes"])))

    un = lambda f, tmpl: (lambda e, iv, p: (iv[0][0], [f(c) if JF._is_num(c) else e.emit(tmpl % {"a": e._src(c), "s": e.wp_scalar_name}) for c in iv[0][1]]))
    JF._HANDLERS.update({
        "select_n": _h_select_n, "and": _h_logic("and"), "or": _h_logic("or"), "not": _h_not,
        "reduce_sum": _h_reduce("add"), "reduce_prod": _h_reduce("mul"), "reduce_max": _h_reduce("max"), "reduce_min": _h_reduce("min"),
        "dot_general": _h_dot_general, "gather": _h_gather, "scatter": _h_scatter(), "scatter-add": _h_scatter("add"), "scatter_add": _h_scatter("add"),
        "reshape": _h_reshape, "copy": lambda e, iv, p: iv[0], "copy_p": lambda e, iv, p: iv[0], "stop_gradient": lambda e, iv, p: iv[0],
        "exp2": un(lambda c: 2.0 ** c, "wp.pow(%(s)s(2.0), %(a)s)"), "square": un(lambda c: c * c, "(%(a)s * %(a)s)"),
        "rsqrt": un(lambda c: c ** -0.5, "(%(s)s(1.0) / wp.sqrt(%(a)s))"), "log1p": un(np.log1p, "wp.log(%(s)s(1.0) + %(a)s)"), "expm1": un(np.expm1, "(wp.exp(%(a)s) - %(s)s(1.0))"),
    })

    def _walk_with_consts(emitter, jaxpr, invals, consts=()):
        env = {}
        for var, c in zip(getattr(jaxpr, "constvars", ()), consts):
            arr = np.asarray(c); env[id(var)] = (tuple(arr.shape), [JF._as_number(v) for v in arr.reshape(-1)])
        for var, val in zip(jaxpr.invars, invals):
            env[id(var)] = val
        for eqn in jaxpr.eqns:
            resolved = [emitter._val_of(a, env) for a in eqn.invars]; name = eqn.primitive.name
            if name in ("pjit", "jit", "closed_call"):
                inner = eqn.params.get("jaxpr"); sub = getattr(inner, "jaxpr", inner)
                outs = _walk_with_consts(emitter, sub, resolved, getattr(inner, "consts", ()))
                for var, val in zip(eqn.outvars, outs): env[id(var)] = val
                continue
            out = emitter.handle(name, resolved, eqn.params)
            if len(eqn.outvars) == 1: env[id(eqn.outvars[0])] = out
            else: raise JF.UnsupportedRHS(name, "multiple-output primitive not supported")
        return [emitter._val_of(a, env) for a in jaxpr.outvars]

    walk_orig = JF._walk

    def _patched_builder(orig):
        def builder(f_jax, dim, n_params, *, wp_scalar):
            jac = orig.__name__ == "jaxpr_to_warp_jacobian"
            fn = jax.jacfwd(f_jax, argnums=1) if jac else f_jax
            closed = jax.make_jaxpr(fn)(0.0, jnp.zeros(int(dim)), jnp.zeros(int(n_params)))
            class _Closed:
                jaxpr = closed.jaxpr; consts = ()
            mj = jax.make_jaxpr; jax.make_jaxpr = lambda *a, **k: (lambda *aa, **kk: _Closed())
            JF._walk = lambda emitter, jaxpr, invals: _walk_with_consts(emitter, jaxpr, invals, closed.consts if jaxpr is closed.jaxpr else ())
            try:
                return orig(f_jax, dim, n_params, wp_scalar=wp_scalar)
            finally:
                jax.make_jaxpr = mj; JF._walk = walk_orig
        return builder
    JF.jaxpr_to_warp_field = _patched_builder(JF.jaxpr_to_warp_field)
    JF.jaxpr_to_warp_jacobian = _patched_builder(JF.jaxpr_to_warp_jacobian)
    JF._astronomix_extended = True


# ---------------------------------------------------------------------------------------------------------------------
# Kernel: adaptive Rosenbrock23 per cell with a per-cell step length (gradsolve's scheme, constants and controller)
# ---------------------------------------------------------------------------------------------------------------------
_CACHE = {}


def build_fused_core_solver(vector_field, dim, *, rtol=1e-6, atol=1e-15, max_steps=2048, cache_key=None):
    """Return ``fn(y0[n, dim], dt[n]) -> y1[n, dim]`` (float64), a JAX primitive running one Warp thread per cell.

    ``vector_field(t, state, args)`` is the diffrax-style right-hand side with CONCRETE closed-over parameters (it is
    traced once here, at build time). A cell whose ``max_steps`` are exhausted returns its last accepted state, like
    diffrax with ``throw=False``.
    """
    key = cache_key if cache_key is not None else id(vector_field)
    key = (key, int(dim), float(rtol), float(atol), int(max_steps))
    if key in _CACHE:
        return _CACHE[key]
    import warp as wp
    from gradsolve.warp import jax_field as JF
    import gradsolve.warp._warp_rosenbrock as GR
    _extend_translator(JF)
    wp.init()
    D = int(dim)

    def f_jax(t, y, p):
        return vector_field(t, y, None)

    field, vecD, vecP = JF.jaxpr_to_warp_field(f_jax, D, 1, wp_scalar=wp.float64)
    jac, _, matDD, _ = JF.jaxpr_to_warp_jacobian(f_jax, D, 1, wp_scalar=wp.float64)
    dcoef = float(GR._D_COEF); e32 = float(GR._E32)
    SAFETY, ERR_FLOOR, FAC_MIN, FAC_MAX, ORDER_EXP = float(GR.SAFETY), float(GR.ERR_FLOOR), float(GR.FAC_MIN), float(GR.FAC_MAX), float(GR.ORDER_EXP)
    RTOL, ATOL, MAXS = float(rtol), float(atol), int(max_steps)

    @wp.func
    def _gauss_solve(A: matDD, b: vecD):
        M = matDD(); x = vecD()
        for i in range(D):
            x[i] = b[i]
            for j in range(D):
                M[i, j] = A[i, j]
        for col in range(D):
            piv = col; best = wp.abs(M[col, col])
            for r in range(col + 1, D):
                v = wp.abs(M[r, col])
                if v > best:
                    best = v; piv = r
            if piv != col:
                for c in range(D):
                    tmp = M[col, c]; M[col, c] = M[piv, c]; M[piv, c] = tmp
                tb = x[col]; x[col] = x[piv]; x[piv] = tb
            for r in range(col + 1, D):
                factor = M[r, col] / M[col, col]
                for c in range(col, D):
                    M[r, c] = M[r, c] - factor * M[col, c]
                x[r] = x[r] - factor * x[col]
        out = vecD()
        for ii in range(D):
            i = D - 1 - ii; s = x[i]
            for c in range(i + 1, D):
                s = s - M[i, c] * out[c]
            out[i] = s / M[i, i]
        return out

    @wp.kernel
    def rosenbrock23_cells(y0: wp.array(dtype=vecD), dt_cell: wp.array(dtype=wp.float64), y_out: wp.array(dtype=vecD)):
        jt = wp.tid()
        t = wp.float64(0.0); t1 = dt_cell[jt]; dt = t1 / wp.float64(100.0)
        y = y0[jt]; p = vecP(); pj = vecD()   # the field takes a length-1 parameter vector, the Jacobian a length-dim one
        for _ in range(MAXS):
            if t >= t1:
                break
            h = wp.min(dt, t1 - t)
            J = jac(t, y, pj)
            W = matDD()
            for i in range(D):
                for jc in range(D):
                    W[i, jc] = -wp.float64(dcoef) * h * J[i, jc]
                W[i, i] = W[i, i] + wp.float64(1.0)
            F0 = field(t, y, p)
            k1 = _gauss_solve(W, F0)
            F1 = field(t, y + wp.float64(0.5) * h * k1, p)
            k2 = _gauss_solve(W, F1 - k1) + k1
            ynew = y + h * k2
            F2 = field(t, ynew, p)
            rhs3 = F2 - wp.float64(e32) * (k2 - F1) - wp.float64(2.0) * (k1 - F0)
            k3 = _gauss_solve(W, rhs3)
            err_vec = (h / wp.float64(6.0)) * (k1 - wp.float64(2.0) * k2 + k3)
            ssum = wp.float64(0.0)
            for i in range(D):
                sc = wp.float64(ATOL) + wp.float64(RTOL) * wp.max(wp.abs(y[i]), wp.abs(ynew[i]))
                comp = err_vec[i] / sc
                ssum = ssum + comp * comp
            err = wp.sqrt(ssum / wp.float64(D))
            if err <= wp.float64(1.0):
                t = t + h; y = ynew
            fac = wp.float64(SAFETY) * wp.pow(wp.max(err, wp.float64(ERR_FLOOR)), wp.float64(ORDER_EXP))
            dt = h * wp.clamp(fac, wp.float64(FAC_MIN), wp.float64(FAC_MAX))
        y_out[jt] = y

    kernel_fn = wp.jax_kernel(rosenbrock23_cells, num_outputs=1)

    def fn(*args):   # jax_kernel returns a list of outputs; hand back the single array
        out = kernel_fn(*args)
        return out[0] if isinstance(out, (list, tuple)) else out
    _CACHE[key] = fn
    return fn
