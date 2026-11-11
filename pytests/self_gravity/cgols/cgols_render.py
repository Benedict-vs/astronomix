"""Volume renderer for CGOLS winds: a JAX emission-absorption ray marcher.

Ray-marches the compact volumes of cgols_render_extract.py (log n, log T, v_r)
front to back with trilinear sampling. A 2D transfer function maps
(log n, second field) to a colour and an extinction coefficient kappa [1/kpc];
each segment adds T * colour * (1 - exp(-kappa ds)) (exact for a constant
segment, so opaque gas cannot overshoot) plus an optional pure-emission glow.
Lighting comes from the analytic gradient of the trilinear interpolant of
log n (no extra samples). The HDR image is tone-mapped (ACES), bloomed and
written as PNG (+ the raw HDR as .npy for re-grading without re-rendering).

Looks (--look):
  density   CGOLS IV Fig. 2 style: blue/black = lowest density, white =
            diffuse, peach = densest (colour and opacity from log n only)
  phase     opacity from log n, colour from log T (hot blue, mixing orange,
            warm 1e4 K peach-white)
  velocity  opacity from log n, colour from v_r (slow red, fast cyan)

    JAX_PLATFORMS=cpu python cgols_render.py data/renders/C512_f1/28 --size 360x640
    python cgols_render.py data/renders/C1024_f1/28 --preset cgols4 --size 1080x1920   # CGOLS IV look
    python cgols_render.py data/renders/C512_f1/28 --look phase --orbit 240 --elev-swing 8   # orbit movie
    python cgols_render.py data/renders/C512_f1/{8,16,28,40,60} --fps 4                       # time-lapse
    python cgols_render.py data/renders/A2048_f2/28 --look phase --elev 12 --azim 35 --size 2160x3840
"""

import argparse
import functools
import json
import os
import shutil
import subprocess
import time

if os.environ.get("JAX_PLATFORMS") != "cpu" and "SLURM_JOB_ID" not in os.environ:
    from autocvd import autocvd

    autocvd(num_gpus=1)

# ruff: noqa: E402
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# Transfer functions: LUT over (log n, b) -> colour rgb, kappa, glow rgb
# ------------------------------------------------------------------

LOGN_RANGE = (-5.0, 3.0)
B_RANGES = {"logT": (3.0, 8.5), "vr": (-500.0, 2000.0)}


def _ramp(x, stops):
    """Piecewise-linear RGB ramp; stops = [(x, (r, g, b)), ...]."""
    xs = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], dtype=np.float64)
    return np.stack([np.interp(x, xs, cs[:, c]) for c in range(3)], axis=-1)


DENSITY_STOPS = [  # CGOLS IV-like: black/blue diffuse wind, white filaments, peach clouds; the opaque disk
    (-3.0, (0.00, 0.01, 0.03)),  # darkens to a dust lane (an opaque region shows its full colour, so a bright
    (-2.5, (0.03, 0.08, 0.22)),  # disk colour turns the edge-on disk envelope into a glowing white slab)
    (-2.0, (0.15, 0.32, 0.65)),
    (-1.7, (0.60, 0.74, 0.93)),
    (-1.45, (0.97, 0.96, 0.97)),
    (-1.15, (0.95, 0.72, 0.55)),
    (-0.9, (0.35, 0.26, 0.22)),
    (-0.3, (0.10, 0.08, 0.07)),
    (1.0, (0.04, 0.03, 0.03)),
]
TEMPERATURE_STOPS = [
    (3.8, (1.00, 0.92, 0.85)),
    (4.3, (1.00, 0.75, 0.55)),
    (5.0, (1.00, 0.45, 0.20)),
    (5.6, (0.95, 0.25, 0.30)),
    (6.2, (0.55, 0.30, 0.85)),
    (6.8, (0.25, 0.45, 1.00)),
    (7.5, (0.55, 0.85, 1.00)),
]
VELOCITY_STOPS = [
    (-200.0, (0.60, 0.10, 0.10)),
    (100.0, (1.00, 0.35, 0.15)),
    (400.0, (1.00, 0.80, 0.45)),
    (800.0, (0.70, 0.90, 1.00)),
    (1200.0, (0.25, 0.60, 1.00)),
    (1700.0, (0.35, 0.25, 0.95)),
]


def kappa_of_logn(logn, kappa_ref, logn_ref, slope, kappa_max=4000.0):
    """Extinction [1/kpc]: kappa_ref at logn_ref, rising by 10^slope per dex."""
    return np.minimum(kappa_ref * 10.0 ** (slope * (logn - logn_ref)), kappa_max)


def build_lut(look, kappa_ref=0.1, logn_ref=-1.5, slope=2.5, glow=0.0, num=256, ramp_shift=0.0,
              hot_fade=1.0, hot_logT=(5.3, 6.3), kappa_knee=None, slope_hi=1.0, dense_dim=None,
              dense_range=(-1.2, -0.2), dense_tint=(1.0, 0.85, 0.75), wind_shift=0.0, wind_pivot=(-2.0, -1.2),
              auto_q=None):
    """(num, num, 7) table over (log n, b): rgb colour, kappa, rgb glow.

    hot_fade    opacity factor for hot gas (log T above hot_logT[1]; smooth from hot_logT[0]), so that the
                dense warm-hot envelope of an adiabatic disk stays translucent while cool gas stays solid
                (density/phase looks only: the velocity look has no temperature axis)
    kappa_knee  above this log n the opacity rises only by 10^slope_hi per dex (softer dense gas)
    wind_shift  shift colour + opacity by this many dex for the diffuse wind only (log n < wind_pivot[0]), fading to
                no shift at wind_pivot[1], so a thinner wind (A vs C) gets the same look while the disk and its
                envelope (similar density in all runs) keep theirs; |wind_shift| < pivot width keeps it monotonic
    dense_dim   darken the colour of dense gas (log n ramping over dense_range) by up to this fraction and
                multiply it by dense_tint, so the opaque disk reads as a solid body instead of a glowing slab
                (default: 0 for density, whose ramp already darkens; 0.8 for phase and velocity)
    """
    logn = np.linspace(*LOGN_RANGE, num)[:, None] * np.ones((1, num))
    if wind_shift:
        w = np.clip((wind_pivot[1] - logn) / (wind_pivot[1] - wind_pivot[0]), 0, 1)
        logn = logn - wind_shift * w  # effective density: the wind looks as if it were |shift| dex denser/thinner
    b_name = {"density": "logT", "phase": "logT", "velocity": "vr"}[look]
    b = np.linspace(*B_RANGES[b_name], num)[None, :] * np.ones((num, 1))
    if kappa_knee is None:
        kappa = kappa_of_logn(logn, kappa_ref, logn_ref, slope)
    else:
        knee = kappa_of_logn(kappa_knee, kappa_ref, logn_ref, slope)
        kappa = np.where(logn < kappa_knee, kappa_of_logn(logn, kappa_ref, logn_ref, slope),
                         kappa_of_logn(logn, knee, kappa_knee, slope_hi))
    if b_name == "logT" and hot_fade != 1.0:
        hot = np.clip((b - hot_logT[0]) / (hot_logT[1] - hot_logT[0]), 0, 1)
        kappa = kappa * 10.0 ** (np.log10(hot_fade) * hot)
    if look == "density":
        colour = _ramp(logn - ramp_shift, DENSITY_STOPS)
    elif look == "phase":
        colour = _ramp(b, TEMPERATURE_STOPS)
    else:
        colour = _ramp(b, VELOCITY_STOPS)
    if dense_dim is None:
        dense_dim = 0.0 if look == "density" else 0.8
    if dense_dim:
        w = np.clip((logn - dense_range[0]) / (dense_range[1] - dense_range[0]), 0, 1)[..., None]
        colour = colour * (1 - dense_dim * w) * (1 - w + w * np.asarray(dense_tint))
    glow_rgb = glow * colour * np.sqrt(kappa)[..., None]  # pure emission, independent of opacity
    lut = np.concatenate([colour, kappa[..., None], glow_rgb], axis=-1).astype(np.float32)
    return lut, b_name


# ------------------------------------------------------------------
# Ray marching
# ------------------------------------------------------------------


def _corners(vol, i, j, k):
    """The 8 trilinear corner values (float32) at integer base indices."""
    return [vol[i + a, j + b, k + c].astype(jnp.float32) for a in (0, 1) for b in (0, 1) for c in (0, 1)]


def _trilinear(corners, fx, fy, fz, with_gradient=False):
    c000, c001, c010, c011, c100, c101, c110, c111 = corners
    c00 = c000 + (c001 - c000) * fz
    c01 = c010 + (c011 - c010) * fz
    c10 = c100 + (c101 - c100) * fz
    c11 = c110 + (c111 - c110) * fz
    c0 = c00 + (c01 - c00) * fy
    c1 = c10 + (c11 - c10) * fy
    value = c0 + (c1 - c0) * fx
    if not with_gradient:
        return value
    gx = c1 - c0
    gy = (c01 - c00) + ((c11 - c10) - (c01 - c00)) * fx
    d0 = (c001 - c000) + ((c011 - c001) - (c010 - c000)) * fy
    d1 = (c101 - c100) + ((c111 - c101) - (c110 - c100)) * fy
    gz = d0 + (d1 - d0) * fx
    return value, jnp.stack([gx, gy, gz], axis=-1)


def _lut_sample(lut, a, b, a_range, b_range):
    num_a, num_b = lut.shape[:2]
    u = jnp.clip((a - a_range[0]) / (a_range[1] - a_range[0]), 0.0, 1.0) * (num_a - 1)
    v = jnp.clip((b - b_range[0]) / (b_range[1] - b_range[0]), 0.0, 1.0) * (num_b - 1)
    i0 = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, num_a - 2)
    j0 = jnp.clip(jnp.floor(v).astype(jnp.int32), 0, num_b - 2)
    fu, fv = (u - i0)[:, None], (v - j0)[:, None]
    top = lut[i0, j0] * (1 - fv) + lut[i0, j0 + 1] * fv
    bottom = lut[i0 + 1, j0] * (1 - fv) + lut[i0 + 1, j0 + 1] * fv
    return top * (1 - fu) + bottom * fu


@functools.lru_cache(maxsize=8)
def make_marcher(shape, box_lo, dx, a_range, b_range, light_dir, shading, ds, max_steps, cyl_radius=None,
                 point_light=None, cyl_fade=0.0):
    """point_light = (pos, r_s, strength, rgb, tau_scale, coarse_shape, coarse_dx) or None: a light at pos whose
    illumination strength * exp(-tau_scale tau) / (r^2 + r_s^2) is scattered by every sample (single scattering,
    colour = the sample's colour times rgb); tau = optical depth from the light, from a coarse grid (tau_grid)."""
    nx, ny, nz = shape
    box_lo = jnp.asarray(box_lo, jnp.float32)
    light = jnp.asarray(np.asarray(light_dir) / np.linalg.norm(light_dir), jnp.float32)

    @jax.jit
    def march(vol_a, vol_b, lut, tau_grid, origins, dirs, t0, t1, jitter):
        num = origins.shape[0]

        def cond(carry):
            s, _, tr = carry
            t = t0 + (s + jitter) * ds
            return (s < max_steps) & jnp.any((tr > 2e-3) & (t < t1))

        def body(carry):
            s, rgb, tr = carry
            t = t0 + (s + jitter) * ds
            active = (t < t1) & (t >= t0)
            p = origins + t[:, None] * dirs
            fade = 1.0
            if cyl_radius is not None:  # round the square box off to a cylinder, feathered over cyl_fade [kpc]
                radius = jnp.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
                active = active & (radius < cyl_radius)
                if cyl_fade > 0:
                    fade = jnp.clip((cyl_radius - radius) / cyl_fade, 0.0, 1.0)
                    fade = fade * fade * (3 - 2 * fade)  # smoothstep
            g = (p - box_lo) / dx - 0.5  # cell-centred index space
            g = jnp.clip(g, 0.0, jnp.array([nx - 1.0, ny - 1.0, nz - 1.0]) - 1e-4)
            base = jnp.minimum(jnp.floor(g).astype(jnp.int32), jnp.array([nx - 2, ny - 2, nz - 2]))
            f = g - base
            i, j, k = base[:, 0], base[:, 1], base[:, 2]
            a, grad = _trilinear(_corners(vol_a, i, j, k), f[:, 0], f[:, 1], f[:, 2], with_gradient=True)
            b = _trilinear(_corners(vol_b, i, j, k), f[:, 0], f[:, 1], f[:, 2])
            entry = _lut_sample(lut, a, b, a_range, b_range)
            colour, kappa, glow = entry[:, :3], entry[:, 3], entry[:, 4:7]
            # gradient lighting: surface normal = -grad log n, blended in by |grad| (sharp edges only)
            gnorm = jnp.linalg.norm(grad, axis=-1)
            normal = -grad / jnp.maximum(gnorm, 1e-6)[:, None]
            lambert = jnp.clip(normal @ light, 0.0, 1.0)
            edge = jnp.clip(gnorm / 0.3, 0.0, 1.0)  # dex per cell
            shade = 1.0 - shading * edge + shading * edge * (0.35 + 0.9 * lambert)
            alpha = jnp.where(active, 1.0 - jnp.exp(-kappa * fade * ds), 0.0)
            glow = glow * jnp.asarray(fade).reshape(-1, 1) if cyl_fade > 0 and cyl_radius is not None else glow
            lit = shade[:, None]
            if point_light is not None:
                pos, r_s, strength, light_rgb, tau_scale, cshape, cdx = point_light
                gc = jnp.clip((p - box_lo) / cdx - 0.5, 0.0, jnp.array(cshape, jnp.float32) - 1.0 - 1e-4)
                bc = jnp.minimum(jnp.floor(gc).astype(jnp.int32), jnp.array(cshape) - 2)
                fc = gc - bc
                tau = _trilinear(_corners(tau_grid, bc[:, 0], bc[:, 1], bc[:, 2]), fc[:, 0], fc[:, 1], fc[:, 2])
                r2 = jnp.sum((p - jnp.asarray(pos, jnp.float32)) ** 2, axis=-1)
                illum = strength * jnp.exp(-tau_scale * tau) / (r2 + r_s**2)
                lit = lit + illum[:, None] * jnp.asarray(light_rgb, jnp.float32)
            rgb = rgb + tr[:, None] * (colour * lit * alpha[:, None]
                                       + jnp.where(active, ds, 0.0)[:, None] * glow)
            tr = tr * (1.0 - alpha)
            return s + 1, rgb, tr

        _, rgb, tr = jax.lax.while_loop(
            cond, body, (jnp.int32(0), jnp.zeros((num, 3), jnp.float32), jnp.ones((num,), jnp.float32))
        )
        return rgb, tr

    return march


def camera_rays(width, height, eye, target, up, fov_deg):
    forward = np.asarray(target, float) - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    tan_half = np.tan(np.radians(fov_deg) / 2)  # vertical field of view
    aspect = width / height
    px = (np.arange(width) + 0.5) / width * 2 - 1
    py = 1 - (np.arange(height) + 0.5) / height * 2
    px, py = np.meshgrid(px * tan_half * aspect, py * tan_half)
    dirs = forward[None, None] + px[..., None] * right + py[..., None] * true_up
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return np.broadcast_to(np.asarray(eye, float), dirs.shape).reshape(-1, 3), dirs.reshape(-1, 3)


def box_intersect(origins, dirs, lo, hi):
    inv = 1.0 / np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    ta, tb = (lo - origins) * inv, (hi - origins) * inv
    t0 = np.maximum(np.minimum(ta, tb).max(axis=1), 0.0)
    t1 = np.maximum(ta, tb).min(axis=1)
    return t0, np.where(t1 > t0, t1, t0)


# ------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------


def aces(x):
    return np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def grade(hdr, exposure=1.0, bloom=0.25, bloom_threshold=0.6, background=(0.0, 0.0, 0.0)):
    img = hdr * exposure
    if bloom > 0:
        bright = np.maximum(img - bloom_threshold, 0)
        size = max(img.shape[:2])
        halo = sum(w * np.stack([gaussian_filter(bright[..., c], s * size) for c in range(3)], -1)
                   for s, w in ((0.004, 0.5), (0.015, 0.3), (0.05, 0.2)))
        img = img + bloom * halo
    img = aces(img)
    img = np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(img, 1 / 2.4) - 0.055)  # sRGB
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def annotate(rgb8, text):
    image = Image.fromarray(rgb8)
    draw = ImageDraw.Draw(image)
    size = max(12, image.width // 40)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((size, size), text, fill=(235, 235, 235), font=font)
    return image


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------


def load_volume(volume_dir, fields):
    with open(os.path.join(volume_dir, "meta.json")) as fh:
        meta = json.load(fh)
    arrays = {k: np.load(os.path.join(volume_dir, f"{k}.npy"), mmap_mode="r") for k in fields}
    return meta, arrays


def camera_eye(target, elev, azim, dist):
    el, az = np.radians(elev), np.radians(azim)
    return np.asarray(target, float) + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def coarse_kappa(logn, second, lut, a_range, b_range, coarse_shape, kappa_max=np.inf):
    """Block-mean extinction kappa [1/kpc] on coarse_shape (kappa averaged at full resolution, so thin dense
    structures keep their optical depth), each cell capped at kappa_max first (the light's opacity: the dense
    disk around a central source would otherwise absorb all of it)."""
    nx, ny, nz = logn.shape
    fx, fy, fz = nx // coarse_shape[0], ny // coarse_shape[1], nz // coarse_shape[2]
    lut_j = jnp.asarray(lut)

    @jax.jit
    def slab_kappa(a, b):
        k = jnp.minimum(_lut_sample(lut_j, a.reshape(-1), b.reshape(-1), a_range, b_range)[:, 3], kappa_max)
        return k.reshape(fx, ny // fy, fy, nz // fz, fz).mean(axis=(0, 2, 4))

    out = np.empty(coarse_shape, np.float32)
    for i in range(coarse_shape[0]):
        out[i] = np.asarray(slab_kappa(jnp.asarray(logn[i * fx:(i + 1) * fx], jnp.float32),
                                       jnp.asarray(second[i * fx:(i + 1) * fx], jnp.float32)))
    return out


def optical_depth_grid(kappa_c, box_lo, cdx, light_pos, samples=192, chunk=1 << 15):
    """tau from light_pos to every coarse cell centre, by marching the coarse kappa grid."""
    cshape = kappa_c.shape
    centres = np.stack(np.meshgrid(*[(np.arange(n) + 0.5) * cdx + lo for n, lo in zip(cshape, box_lo)],
                                   indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    kc = jnp.asarray(kappa_c)
    upper = jnp.array(cshape, jnp.float32) - 1.0 - 1e-4
    frac = (jnp.arange(samples, dtype=jnp.float32) + 0.5) / samples
    pos = jnp.asarray(light_pos, jnp.float32)
    lo = jnp.asarray(box_lo, jnp.float32)

    @jax.jit
    def tau_of(points):
        seg = points - pos
        length = jnp.linalg.norm(seg, axis=-1)

        def step(acc, f):
            q = pos + f * seg
            g = jnp.clip((q - lo) / cdx - 0.5, 0.0, upper)
            b = jnp.minimum(jnp.floor(g).astype(jnp.int32), jnp.array(cshape) - 2)
            fr = g - b
            k = _trilinear(_corners(kc, b[:, 0], b[:, 1], b[:, 2]), fr[:, 0], fr[:, 1], fr[:, 2])
            return acc + k, None

        acc, _ = jax.lax.scan(step, jnp.zeros(points.shape[0], jnp.float32), frac)
        return acc * length / samples

    out = np.empty(len(centres), np.float32)
    for c in range(0, len(centres), chunk):
        part = centres[c:c + chunk]
        pad = chunk - len(part)
        res = tau_of(jnp.asarray(np.pad(part, [(0, pad), (0, 0)]) if pad else part))
        out[c:c + len(part)] = np.asarray(res)[: len(part)]
    return out.reshape(cshape)


WIND_LOGN_P90_REF = -1.70  # C512/C1024 at 35 Myr, |z| > 1 kpc: the run the transfer functions were tuned on
# Reference wind log n per statistic (C512/C1024 at 35 Myr). The median tracks the volume-filling wind, which
# thins ~5x after the starburst; the 90th percentile is dominated by dense gas near the disk and barely moves.
WIND_LOGN_REF = {90: WIND_LOGN_P90_REF, 50: -2.25}


def wind_logn_p90(logn, height_kpc=1.0, samples=128, q=90):
    """q-th percentile (default 90) of log n above |z| = height_kpc, from a strided subsample."""
    nx, _, nz = logn.shape
    s = max(1, nx // samples)
    sub = np.asarray(logn[::s, ::s, ::s], np.float32)
    z = (np.arange(sub.shape[2]) * s + 0.5) * 20.0 / nz - 10.0
    return float(np.percentile(sub[:, :, np.abs(z) > height_kpc], q))


class Scene:
    """One volume on the device with a transfer function; render any number of views of it."""

    def __init__(self, volume_dir, look="density", step_cells=0.75, kappa_ref=0.1, logn_ref=-1.5, slope=2.5,
                 glow=0.0, shading=1.0, ramp_shift=0.0, light=(0.4, -0.3, 0.85), crop_kpc=None, cyl_radius=None,
                 chunk=1 << 16, auto=False, light_strength=0.0, light_radius=0.3, light_rgb=(1.0, 0.72, 0.45),
                 light_tau=1.0, light_pos=(0.0, 0.0, 0.0), light_grid=(128, 128, 256), light_kappa_max=2.0,
                 cyl_fade=0.0, **tf):
        b_name = {"density": "logT", "phase": "logT", "velocity": "vr"}[look]
        self.meta, arrays = load_volume(volume_dir, ("logn", b_name))
        if auto is not False:  # follow the wind: shift ramp + opacity by its density relative to the tuning run
            # True: measure on this volume; a number: a shift fixed elsewhere (time-lapse: from the first frame)
            q = tf.pop("auto_q", 90)
            shift = wind_logn_p90(arrays["logn"], q=q) - WIND_LOGN_REF[q] if auto is True else float(auto)
            if q == 50:  # the median shift can reach ~1 dex: a wider transition keeps the mapping monotonic
                tf.setdefault("wind_pivot", (-2.6, -1.2))
            self.auto_shift = shift
            tf["wind_shift"] = shift  # wind only: the disk keeps its colours and opacity
            print(f"  auto: wind log n p90 shift {shift:+.2f} dex", flush=True)
        lut, _ = build_lut(look, kappa_ref, logn_ref, slope, glow, ramp_shift=ramp_shift, **tf)
        box = np.asarray(self.meta["box_kpc"], float)
        dx = float(self.meta["dx_kpc"])
        shape = tuple(self.meta["shape"])
        self.lo, self.hi = -box / 2, box / 2
        if crop_kpc is not None:  # render only |x|,|y| < crop[0], |z| < crop[1]
            self.lo = np.maximum(self.lo, [-crop_kpc[0], -crop_kpc[0], -crop_kpc[1]])
            self.hi = np.minimum(self.hi, [crop_kpc[0], crop_kpc[0], crop_kpc[1]])
        self.ds = step_cells * dx
        self.step_cells = step_cells
        # fixed for every view (the while loop stops early), so one compilation serves an orbit
        max_steps = int(np.ceil(np.linalg.norm(self.hi - self.lo) / self.ds)) + 2
        start = time.time()
        self.vol_a = jnp.asarray(arrays["logn"])
        self.vol_b = jnp.asarray(arrays[b_name])
        print(f"  loaded {volume_dir}: {shape} x2 in {time.time() - start:.0f} s on {jax.devices()[0].platform}",
              flush=True)
        point_light = None
        self.tau_grid = jnp.zeros((2, 2, 2), jnp.float32)
        if light_strength > 0:
            start = time.time()
            grid = tuple(min(g, n) for g, n in zip(light_grid, shape))
            cdx = box[0] / grid[0]
            kappa_c = coarse_kappa(arrays["logn"], arrays[b_name], lut, LOGN_RANGE, B_RANGES[b_name], grid,
                                   light_kappa_max)
            self.tau_grid = jnp.asarray(optical_depth_grid(kappa_c, -box / 2, cdx, light_pos))
            point_light = (tuple(light_pos), float(light_radius), float(light_strength), tuple(light_rgb),
                           float(light_tau), grid, float(cdx))
            print(f"  light: tau grid {grid} in {time.time() - start:.0f} s, median tau "
                  f"{float(jnp.median(self.tau_grid)):.2f}", flush=True)
        self.march = make_marcher(shape, tuple(-box / 2), dx, LOGN_RANGE, B_RANGES[b_name], tuple(light),
                                  shading, self.ds, max_steps, cyl_radius, point_light, float(cyl_fade))
        self.lut = jnp.asarray(lut)
        self.chunk = chunk

    def view(self, size=(360, 640), elev=10.0, azim=30.0, dist=17.0, fov=50.0, target=(0.0, 0.0, 0.0), seed=0):
        width, height = size
        eye = camera_eye(target, elev, azim, dist)
        origins, dirs = camera_rays(width, height, eye, target, (0, 0, 1), fov)
        t0, t1 = box_intersect(origins, dirs, self.lo, self.hi)
        jitter = np.random.default_rng(seed).random(len(t0)).astype(np.float32)
        rgb = np.zeros((len(t0), 3), np.float32)
        chunk, start = self.chunk, time.time()
        for c in range(0, len(t0), chunk):
            sl = slice(c, c + chunk)
            pad = chunk - len(t0[sl])  # constant chunk shape: one compilation

            def p(x):
                return np.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1)) if pad else x

            out, _ = self.march(self.vol_a, self.vol_b, self.lut, self.tau_grid,
                                *(jnp.asarray(p(x), jnp.float32) for x in
                                  (origins[sl], dirs[sl], t0[sl], t1[sl], jitter[sl])))
            rgb[sl] = np.asarray(out)[: chunk - pad]
        print(f"  rendered {width}x{height} (elev {elev:g}, azim {azim:g}), step {self.step_cells} cells, "
              f"{time.time() - start:.1f} s", flush=True)
        return rgb.reshape(height, width, 3)


def render(volume_dir, look="density", size=(360, 640), elev=10.0, azim=30.0, dist=17.0, fov=50.0,
           target=(0.0, 0.0, 0.0), step_cells=0.75, kappa_ref=0.1, logn_ref=-1.5, slope=2.5, glow=0.0,
           shading=1.0, ramp_shift=0.0, light=(0.4, -0.3, 0.85), chunk=1 << 16, seed=0, crop_kpc=None,
           cyl_radius=None, **tf):
    """One still (convenience wrapper around Scene); tf = extra build_lut options."""
    scene = Scene(volume_dir, look, step_cells, kappa_ref, logn_ref, slope, glow, shading, ramp_shift, light,
                  crop_kpc, cyl_radius, chunk, **tf)
    return scene.view(size, elev, azim, dist, fov, target, seed), scene.meta


def ffmpeg_binary():
    candidates = [os.environ.get("CGOLS_FFMPEG"), shutil.which("ffmpeg"), "/export/data/bschuber/envs/ffmpeg/bin/ffmpeg"]
    return next((c for c in candidates if c and os.path.exists(c)), None)


def encode_movie(frame_dir, out, fps):
    """frame_dir/frame_%05d.png -> H.264 mp4 (yuv420p, plays everywhere)."""
    binary = ffmpeg_binary()
    if binary is None:
        print("no ffmpeg found (set CGOLS_FFMPEG); frames are in", frame_dir)
        return
    subprocess.run([binary, "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(frame_dir, "frame_%05d.png"), "-c:v", "libx264", "-preset", "slow",
                    "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    print("wrote", out)


# Named looks: CLI defaults that explicit flags still override.
PRESETS = {
    # CGOLS IV Fig. 2 / aesthetic-computation #31: sparse blue-hazed wind, white streaks, grey dust-lane disk with a
    # bright rim, the centre lit warm by a (shadowed) central light; colours follow the run's wind density (--auto)
    "cgols4": dict(look="density", auto=True, kappa_ref=0.03, slope=3.0, glow=0.3, light=20.0, light_kappa_max=1.5,
                   light_rgb="1,0.55,0.28", elev=16.0),  # no cylinder cut (user 11-11): the wind runs off the frame edges
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None, help="named look (flags override)")
    parser.add_argument("volume_dir", nargs="+", help="one volume (still/orbit) or several (time-lapse)")
    parser.add_argument("--look", default="density", choices=("density", "phase", "velocity"))
    parser.add_argument("--size", default="360x640", help="WxH")
    parser.add_argument("--elev", type=float, default=10.0)
    parser.add_argument("--azim", type=float, default=30.0)
    parser.add_argument("--dist", type=float, default=17.0)
    parser.add_argument("--fov", type=float, default=50.0)
    parser.add_argument("--target", default="0,0,0")
    parser.add_argument("--step", type=float, default=0.75, help="ray step in cells")
    parser.add_argument("--kappa-ref", type=float, default=0.1, help="opacity [1/kpc] at --logn-ref")
    parser.add_argument("--logn-ref", type=float, default=-1.5)
    parser.add_argument("--slope", type=float, default=2.5, help="opacity rise per dex of n")
    parser.add_argument("--ramp-shift", type=float, default=0.0, help="shift the density colour ramp [dex]")
    parser.add_argument("--glow", type=float, default=0.0)
    parser.add_argument("--hot-fade", type=float, default=1.0, help="opacity factor for T > 2e6 K gas")
    parser.add_argument("--kappa-knee", type=float, default=None, help="log n where the opacity slope softens")
    parser.add_argument("--slope-hi", type=float, default=1.0, help="opacity slope above the knee")
    parser.add_argument("--dense-dim", type=float, default=None, help="darken dense gas (default 0 / 0.8)")
    parser.add_argument("--dense-range", default="-1.2,-0.2", help="log n range over which --dense-dim ramps in")
    parser.add_argument("--dense-tint", default="1,0.85,0.75", help="rgb multiplier for dense gas")
    parser.add_argument("--shading", type=float, default=1.0)
    parser.add_argument("--exposure", type=float, default=1.1)
    parser.add_argument("--bloom", type=float, default=0.25)
    parser.add_argument("--crop", default=None, help="R,Z half-extents in kpc")
    parser.add_argument("--cyl", type=float, default=None, help="render only cylindrical R < CYL kpc")
    parser.add_argument("--cyl-fade", type=float, default=0.0, help="feather the cylinder edge over this [kpc]")
    parser.add_argument("--chunk", type=int, default=1 << 16)
    parser.add_argument("--auto", action="store_true", help="shift ramp + opacity to this run's wind density")
    parser.add_argument("--auto-q", type=int, default=90, choices=(50, 90),
                        help="wind statistic for --auto: 90 = dense structures (default), 50 = volume-filling wind "
                             "(keeps the thin late-time wind visible, i.e. normalises each frame)")
    parser.add_argument("--wind-pivot", default=None,
                        help="lo,hi log n: the wind shift applies fully below lo, fading out at hi")
    parser.add_argument("--wind-shift", type=float, default=None,
                        help="fixed wind shift [dex] instead of measuring it (keeps a time series comparable)")
    parser.add_argument("--light", type=float, default=0.0, help="central light strength (0 = off)")
    parser.add_argument("--light-radius", type=float, default=0.3, help="light softening radius [kpc]")
    parser.add_argument("--light-tau", type=float, default=1.0, help="opacity scale for the light's shadows")
    parser.add_argument("--light-rgb", default="1,0.72,0.45")
    parser.add_argument("--light-kappa-max", type=float, default=2.0, help="cap of the light's opacity [1/kpc]")
    parser.add_argument("--out", default=None, help="still: .png; movie: .mp4 (frames go next to it)")
    parser.add_argument("--label", action="store_true", help="stamp run and time")
    parser.add_argument("--orbit", type=int, default=0, help="movie: N frames over 360 deg of azimuth")
    parser.add_argument("--elev-swing", type=float, default=0.0, help="orbit: elevation += SWING sin(phase)")
    parser.add_argument("--azim-rate", type=float, default=0.0, help="time-lapse: azimuth drift per frame [deg]")
    parser.add_argument("--fps", type=int, default=30)
    preset = parser.parse_known_args()[0].preset
    if preset:
        parser.set_defaults(**PRESETS[preset])
    args = parser.parse_args()

    size = tuple(int(v) for v in args.size.split("x"))
    crop = tuple(float(v) for v in args.crop.split(",")) if args.crop else None
    target = tuple(float(v) for v in args.target.split(","))
    scene_kw = dict(look=args.look, step_cells=args.step, kappa_ref=args.kappa_ref, logn_ref=args.logn_ref,
                    slope=args.slope, glow=args.glow, shading=args.shading, ramp_shift=args.ramp_shift,
                    crop_kpc=crop, cyl_radius=args.cyl, chunk=args.chunk,
                    auto=args.wind_shift if args.wind_shift is not None else args.auto, hot_fade=args.hot_fade,
                    light_strength=args.light, light_radius=args.light_radius, light_tau=args.light_tau,
                    light_kappa_max=args.light_kappa_max, cyl_fade=args.cyl_fade, auto_q=args.auto_q,
                    **({"wind_pivot": tuple(float(v) for v in args.wind_pivot.split(","))} if args.wind_pivot else {}),
                    light_rgb=tuple(float(v) for v in args.light_rgb.split(",")),
                    kappa_knee=args.kappa_knee, slope_hi=args.slope_hi, dense_dim=args.dense_dim,
                    dense_range=tuple(float(v) for v in args.dense_range.split(",")),
                    dense_tint=tuple(float(v) for v in args.dense_tint.split(",")))
    view_kw = dict(size=size, dist=args.dist, fov=args.fov, target=target)

    def finish(hdr, meta, path, save_hdr=True):
        if save_hdr:
            np.save(path.replace(".png", "_hdr.npy"), hdr.astype(np.float16))
        image = grade(hdr, args.exposure, args.bloom)
        image = annotate(image, f"{meta['tag'].lstrip('_')}  t = {meta['time_myr']:.1f} Myr") if args.label \
            else Image.fromarray(image)
        image.save(path)

    first = load_volume(args.volume_dir[0], ())[0]
    name = f"{first['tag'].lstrip('_')}_f{first['factor']}_{first['step']:02d}_{args.look}"
    movie = args.orbit > 0 or len(args.volume_dir) > 1
    if not movie:
        out = args.out or os.path.join(SCRIPT_DIR, "figures", "cgols", "renders",
                                       f"{name}_e{args.elev:g}a{args.azim:g}.png")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        scene = Scene(args.volume_dir[0], **scene_kw)
        finish(scene.view(elev=args.elev, azim=args.azim, **view_kw), scene.meta, out)
        print("wrote", out)
        return

    kind = f"orbit{args.orbit}" if args.orbit else f"time{len(args.volume_dir)}"
    out = args.out or os.path.join(SCRIPT_DIR, "figures", "cgols", "renders", f"{name}_{kind}.mp4")
    frame_dir = out[: -len(".mp4")] + "_frames"
    os.makedirs(frame_dir, exist_ok=True)
    frame = 0
    if args.orbit:
        scene = Scene(args.volume_dir[0], **scene_kw)
        for i in range(args.orbit):
            phase = 2 * np.pi * i / args.orbit
            path = os.path.join(frame_dir, f"frame_{i:05d}.png")
            if not os.path.exists(path):  # resumable
                hdr = scene.view(elev=args.elev + args.elev_swing * np.sin(phase),
                                 azim=args.azim + np.degrees(phase), seed=i, **view_kw)
                finish(hdr, scene.meta, path, save_hdr=False)
    else:
        if args.auto and args.wind_shift is None:  # one shift for the whole time-lapse (real thinning stays visible)
            scene_kw["auto"] = wind_logn_p90(load_volume(args.volume_dir[0], ("logn",))[1]["logn"]) - WIND_LOGN_P90_REF
        for frame, volume_dir in enumerate(args.volume_dir):
            path = os.path.join(frame_dir, f"frame_{frame:05d}.png")
            if not os.path.exists(path):
                scene = Scene(volume_dir, **scene_kw)
                hdr = scene.view(elev=args.elev, azim=args.azim + args.azim_rate * frame, seed=frame, **view_kw)
                finish(hdr, scene.meta, path, save_hdr=False)
                del scene
    encode_movie(frame_dir, out, args.fps)


if __name__ == "__main__":
    main()
