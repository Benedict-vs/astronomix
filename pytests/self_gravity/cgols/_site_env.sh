#!/bin/bash
# Machine-specific environment for the cgols runners.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/_site_env.sh"
#
# Everything that differs between clusters lives behind CGOLS_SITE
# (horeka2 | horeka | jupiter | local), so run_cgols_multinode.sh /
# run_cgols_bench.sh stay portable and a new machine is a ~15-line branch here
# plus a different SBATCH header. Defaults to `horeka2` (HoreKa 2, x86 side:
# Ruby gpu-h200 / Teal gpu-h100); `horeka` is the legacy system, retired 12/2026.
#
# No `set -e` anywhere in these scripts: `module` and `micromamba` are shell
# functions that misbehave under it.
#
# Provides, after sourcing:
#   REPO, CGOLS_DIR       - repo root and the cgols script directory
#   cgols_preflight       - version banner (jax / orbax / ptxas / git) into the .out
#   cgols_check_workspace - the ./data symlink exists AND is globally visible
#   start_gpu_logger / stop_gpu_logger - per-node nvidia-smi sampling

CGOLS_SITE="${CGOLS_SITE:-horeka2}"

# Resolve the repo from THIS file's location, never from $0: under Slurm $0 is
# a copy in the spool directory, so $0-relative paths point nowhere.
CGOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$CGOLS_DIR/../../.." && pwd)"   # pytests/self_gravity/cgols -> repo root
export REPO CGOLS_DIR

_cgols_activate_env() {
    source ~/.bashrc
    # A non-interactive batch shell can return from ~/.bashrc before reaching
    # the micromamba init block; fall back to the hook from the binary itself.
    if ! declare -F micromamba >/dev/null; then
        eval "$("${MAMBA_EXE:-$HOME/.local/bin/micromamba}" shell hook --shell bash)"
    fi
    # micromamba, NOT conda: `conda activate` inside a batch job silently falls
    # back to the system python when conda's shell hook was never initialised.
    micromamba activate "${CGOLS_ENV:-astro}"
}

_cgols_pip_nvidia_libs() {
    # Every pip-installed NVIDIA lib, independent of the python version. The
    # glob matters: it picks up nvidia/nccl/lib, which a fixed list of
    # cublas/cufft/cudnn/... misses. NCCL on LD_LIBRARY_PATH is ESSENTIAL once
    # traffic leaves the node - without it the inter-node rendezvous hangs.
    local pysite d
    pysite=$(python -c 'import site; print(site.getsitepackages()[0])')
    for d in "$pysite"/nvidia/*/lib; do
      [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
    done
}

# `module`, ~/.bashrc and the micromamba hook are shell code that reads unset
# variables (HoreKa's /etc/bashrc references BASHRCSOURCED), so under the
# runners' `set -u` they abort the job. Run the site block with nounset off and
# restore the caller's setting afterwards.
_cgols_had_nounset=0
case $- in *u*) _cgols_had_nounset=1; set +u ;; esac

case "$CGOLS_SITE" in
  horeka2)
    # HoreKa 2, x86 login hk2-x86.scc.kit.edu (Ruby gpu-h200, Teal gpu-h100).
    # Jade (gpu-b200) is aarch64 and needs its own env - not this branch.
    # No CUDA module: the software stack moved to EasyBuild with new names,
    # and nothing here needs one - the pip nvidia-* wheels provide every
    # runtime lib incl. NCCL, and ptxas comes from nvidia/cuda_nvcc (the
    # preflight banner prints which ptxas actually won).
    module purge 2>/dev/null
    _cgols_activate_env
    _cgols_pip_nvidia_libs
    ;;

  horeka)
    # Legacy HoreKa (decommissioned 12/2026).
    module purge
    module load devel/cuda/12.9
    _cgols_activate_env
    _cgols_pip_nvidia_libs
    ;;

  jupiter)
    # JSC JUPITER (GH200). Placeholder mirroring the HoreKa branch; adjust the
    # module names once an allocation exists.
    module purge
    module load CUDA
    _cgols_activate_env
    _cgols_pip_nvidia_libs
    ;;

  local)
    # Bare workstation: assume the environment is already active.
    ;;

  *)
    echo "ERROR: unknown CGOLS_SITE='$CGOLS_SITE' (horeka2 | horeka | jupiter | local)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

[ "$_cgols_had_nounset" = 1 ] && set -u
unset _cgols_had_nounset

# --- JAX runtime knobs ------------------------------------------------------
# cuda_async instead of the default BFC allocator: the compiled step needs a
# ~112 GB CONTIGUOUS temp arena at 1024 that BFC could not place next to the
# resident arguments (job 4675223 OOM'd with the memory nominally free). 0.98
# because the program totals ~134 GB of the H200's ~151 GB.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-cuda_async}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.98}"

# astronomix names its mesh axes with integers (VARAXIS=0, XAXIS=1, ...), which
# jax >= 0.10's default Shardy partitioner rejects (sdy.MeshAxisAttr.get wants a
# str). cgols.py also sets this via jax.config, but exporting it here makes it
# hold regardless of import ordering.
export JAX_USE_SHARDY_PARTITIONER=false

# Surface NCCL problems instead of hanging silently. WARN is quiet on a healthy
# run; NCCL_DEBUG=INFO is the escalation when a rendezvous does hang.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"


cgols_preflight() {
    # Version banner. Every job's .out then documents exactly which jax, orbax,
    # CUDA toolchain and commit produced it - the single most useful thing when
    # a run from three months ago has to be reproduced or blamed.
    echo "=== preflight $(date -Is) ==="
    echo "site:      $CGOLS_SITE"
    echo "repo:      $REPO"
    echo "commit:    $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?') \
$(git -C "$REPO" describe --always --dirty --broken 2>/dev/null || true)"
    echo "branch:    $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    # JAX_PLATFORMS=cpu so the banner never grabs (or preallocates on) a GPU.
    JAX_PLATFORMS=cpu python - <<'PY'
import sys
print(f"python:    {sys.version.split()[0]}  ({sys.executable})")
for label, mod, attr in (
    ("jax", "jax", "__version__"),
    ("jaxlib", "jaxlib", "__version__"),
    ("orbax", "orbax.checkpoint", "__version__"),
):
    try:
        import importlib
        print(f"{label + ':':10} {getattr(importlib.import_module(mod), attr)}")
    except Exception as exc:
        print(f"{label + ':':10} UNAVAILABLE ({type(exc).__name__}: {exc})")
try:
    import astronomix
    print(f"astronomix: {astronomix.__file__}")
except Exception as exc:
    print(f"astronomix: IMPORT FAILED ({type(exc).__name__}: {exc})")
PY
    # ptxas: the CUDA toolchain that actually compiles the Pallas/Triton
    # kernels. A mismatch against the driver shows up here, not at runtime.
    # Two places it can come from - the loaded CUDA module, or the pip
    # nvidia-cuda-nvcc wheel that jax[cuda12] pulls in - and which one wins is
    # exactly the kind of thing this banner exists to record.
    local ptxas_bin
    ptxas_bin="$(command -v ptxas 2>/dev/null)"
    if [ -z "$ptxas_bin" ]; then
        ptxas_bin="$(JAX_PLATFORMS=cpu python -c \
            'import site,glob;print(next(iter(glob.glob(site.getsitepackages()[0]+"/nvidia/*/bin/ptxas")),""))' \
            2>/dev/null)"
    fi
    if [ -n "$ptxas_bin" ] && [ -x "$ptxas_bin" ]; then
        echo "ptxas:     $ptxas_bin"
        echo "           $("$ptxas_bin" --version 2>&1 | tr '\n' ' ')"
    else
        echo "ptxas:     NOT FOUND (neither on PATH nor in site-packages/nvidia)"
    fi
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
    nvidia-smi -L 2>/dev/null || echo "nvidia-smi: unavailable"
    echo "=== end preflight ==="
}


cgols_check_workspace() {
    # Bulk data (~2.5 TB at 2048) must live on a workspace, and - unlike the
    # single-node 1024 run - that workspace must be GLOBALLY visible: with 8
    # nodes a node-local ./data means eight private, silently diverging output
    # trees. Checking the link exists is not enough; compare the device:inode
    # every rank sees.
    cd "$CGOLS_DIR" || return 1
    if [ ! -L data ] || [ ! -d data ]; then
        echo "ERROR: link ./data to a workspace first, e.g." >&2
        echo "         ws_allocate cgols 60 && ln -s \"\$(ws_find cgols)\" data" >&2
        return 1
    fi
    echo "workspace: $(readlink -f data)"

    if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
        local ids
        ids=$(srun --gpu-bind=none --ntasks-per-node=1 \
                  bash -c 'stat -Lc "%d:%i" '"$CGOLS_DIR"'/data' 2>/dev/null | sort -u)
        local n
        n=$(echo "$ids" | grep -c .)
        if [ "$n" -ne 1 ]; then
            echo "ERROR: ./data is NOT the same filesystem object on every node" >&2
            echo "       (device:inode seen: $(echo "$ids" | tr '\n' ' '))." >&2
            echo "       A node-local workspace gives each node its own private" >&2
            echo "       checkpoint/frame tree - silent at 1 node, fatal at 8." >&2
            return 1
        fi
        echo "workspace visible identically on all $SLURM_NNODES nodes ($ids)"
    fi
}


start_gpu_logger() {
    # One CSV per NODE. A plain `nvidia-smi -l` in the batch script only ever
    # samples the batch host, which on a 8-node job is 1/8 of the picture.
    GPU_LOG_DIR="${CGOLS_DIR}/cgols_logs"
    mkdir -p "$GPU_LOG_DIR"
    local jobid="${SLURM_JOB_ID:-local}"
    if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
        # --overlap so this step can coexist with the solver's srun rather than
        # queueing behind it (without it the logger waits for the whole run).
        srun --gpu-bind=none --ntasks-per-node=1 --overlap bash -c \
            "nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
             --format=csv -l 30 > '$GPU_LOG_DIR/gpu_${jobid}_'\$(hostname).csv" &
    else
        nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
            --format=csv -l 30 > "$GPU_LOG_DIR/gpu_${jobid}.csv" 2>/dev/null &
    fi
    GPU_LOGGER_PID=$!
}

stop_gpu_logger() {
    [ -n "${GPU_LOGGER_PID:-}" ] && kill "$GPU_LOGGER_PID" 2>/dev/null
    return 0
}
