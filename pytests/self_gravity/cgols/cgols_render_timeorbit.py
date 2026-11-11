"""Time-evolving orbit movie: the camera orbits (1 deg per frame by default) while the simulation time runs.

Every checkpoint volume (cgols_render_extract.py, log n + log T) gets FRAMES_PER_STEP frames; in between, each frame
blends the renders of the two neighbouring checkpoints seen from that frame's camera (cross-fade of the HDR
images), so the time steps do not jump. The two current volumes are kept on the GPU as a sliding window.

Brightness follows the volume-filling wind: the per-checkpoint wind shift (median-based, see cgols_render --auto-q
50) is smoothed over time (running mean) and clipped, so the starburst and the later quieting both stay visible
without flicker. Grading (ACES + bloom) runs on the device.

    python cgols_render_timeorbit.py data/renders/C1024_f1 --out .../C1024_timeorbit.mp4
    python cgols_render_timeorbit.py data/renders/A2048_f2 --pivot -3.0,-1.6 --hot-fade 0.3 --out ...
"""

import argparse
import json
import os
import time

if os.environ.get("JAX_PLATFORMS") != "cpu" and "SLURM_JOB_ID" not in os.environ:
    from autocvd import autocvd

    autocvd(num_gpus=1)

# ruff: noqa: E402
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import cgols_render as R


def gaussian_blur_fft(img, sigma_px):
    """Separable-free Gaussian blur of an (H, W, 3) image via FFT (periodic padding by 3 sigma)."""
    pad = int(3 * sigma_px)
    x = jnp.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    h, w = x.shape[:2]
    ky = jnp.fft.fftfreq(h)[:, None]
    kx = jnp.fft.fftfreq(w)[None, :]
    kernel = jnp.exp(-2 * (jnp.pi * sigma_px) ** 2 * (kx**2 + ky**2))
    out = jnp.fft.ifft2(jnp.fft.fft2(x, axes=(0, 1)) * kernel[..., None], axes=(0, 1)).real
    return out[pad:pad + img.shape[0], pad:pad + img.shape[1]]


@jax.jit
def grade_device(hdr, exposure=1.1, bloom=0.25, threshold=0.6):
    """Same look as cgols_render.grade (ACES + 3-scale bloom + sRGB), on the device."""
    img = hdr * exposure
    bright = jnp.maximum(img - threshold, 0.0)
    size = max(img.shape[:2])
    halo = sum(w * gaussian_blur_fft(bright, s * size) for s, w in ((0.004, 0.5), (0.015, 0.3), (0.05, 0.2)))
    img = img + bloom * halo
    img = jnp.clip(img * (2.51 * img + 0.03) / (img * (2.43 * img + 0.59) + 0.14), 0, 1)
    img = jnp.where(img <= 0.0031308, 12.92 * img, 1.055 * jnp.power(img, 1 / 2.4) - 0.055)
    return (jnp.clip(img, 0, 1) * 255 + 0.5).astype(jnp.uint8)


def wind_shifts(root, steps, smooth=5, clip=(-1.4, 0.3), hold_before_myr=40.0):
    """Median-based wind shift per checkpoint, running-mean smoothed; held at its hold_before_myr value before then
    (early on the median is set by the thin initial halo and would over-brighten the superbubble shell; through
    the starburst the fixed look of the stills is right, only the post-starburst thinning needs normalising)."""
    raw, times = [], []
    for step in steps:
        meta = json.load(open(os.path.join(root, str(step), "meta.json")))
        logn = np.load(os.path.join(root, str(step), "logn.npy"), mmap_mode="r")
        raw.append(R.wind_logn_p90(logn, q=50) - R.WIND_LOGN_REF[50])
        times.append(meta["time_myr"])
    raw, times = np.array(raw), np.array(times)
    first = np.argmax(times >= hold_before_myr)
    raw[:first] = raw[first]
    kernel = np.ones(smooth) / smooth
    padded = np.pad(raw, smooth // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")[: len(raw)]
    return np.clip(smoothed, *clip), raw, times


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="data/renders/<run>_f<factor> with one directory per checkpoint")
    parser.add_argument("--out", required=True, help=".mp4; frames go to <out>_frames/")
    parser.add_argument("--steps", default=None, help="a-b (default: all extracted)")
    parser.add_argument("--frames-per-step", type=int, default=6)
    parser.add_argument("--deg-per-frame", type=float, default=1.0)
    parser.add_argument("--azim", type=float, default=30.0)
    parser.add_argument("--elev", type=float, default=16.0)
    parser.add_argument("--elev-swing", type=float, default=6.0)
    parser.add_argument("--size", default="1080x1920")
    parser.add_argument("--step", type=float, default=0.6, help="ray step [cells]")
    parser.add_argument("--pivot", default="-2.6,-1.2", help="wind-shift pivot (log n lo,hi)")
    parser.add_argument("--hot-fade", type=float, default=1.0)
    parser.add_argument("--hold-before", type=float, default=40.0, help="fixed brightness before this time [Myr]")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--chunk", type=int, default=1 << 20)
    parser.add_argument("--label", default=None, help="run label (default: from meta)")
    args = parser.parse_args()

    steps = sorted(int(d) for d in os.listdir(args.root)
                   if d.isdigit() and os.path.exists(os.path.join(args.root, d, "meta.json")))
    if args.steps:
        lo, hi = (int(v) for v in args.steps.split("-"))
        steps = [s for s in steps if lo <= s <= hi]
    shifts, raw, times = wind_shifts(args.root, steps, hold_before_myr=args.hold_before)
    print("wind shift per checkpoint (raw -> smoothed):",
          " ".join(f"{t:.1f}:{r:+.2f}->{s:+.2f}" for t, r, s in zip(times, raw, shifts)), flush=True)

    size = tuple(int(v) for v in args.size.split("x"))
    pivot = tuple(float(v) for v in args.pivot.split(","))
    preset = R.PRESETS["cgols4"]
    scene_kw = dict(look="density", step_cells=args.step, kappa_ref=preset["kappa_ref"], slope=preset["slope"],
                    glow=preset["glow"], light_strength=preset["light"], light_kappa_max=preset["light_kappa_max"],
                    light_rgb=tuple(float(v) for v in preset["light_rgb"].split(",")), chunk=args.chunk,
                    hot_fade=args.hot_fade, wind_pivot=pivot)
    frame_dir = args.out[: -len(".mp4")] + "_frames"
    os.makedirs(frame_dir, exist_ok=True)
    n_frames = (len(steps) - 1) * args.frames_per_step + 1
    window = {}  # step index -> Scene (sliding window of two)

    def scene(i):
        if i not in window:
            for k in [k for k in window if k < i - 1]:
                del window[k]
            window[i] = R.Scene(os.path.join(args.root, str(steps[i])), auto=float(shifts[i]), auto_q=50, **scene_kw)
        return window[i]

    start = time.time()
    for f in range(n_frames):
        path = os.path.join(frame_dir, f"frame_{f:05d}.png")
        if os.path.exists(path):
            continue
        pos = f / args.frames_per_step
        i, w = int(pos), pos - int(pos)
        phase = np.radians(args.deg_per_frame * f)
        view = dict(size=size, elev=args.elev + args.elev_swing * np.sin(phase), azim=args.azim + np.degrees(phase),
                    dist=17.0, fov=50.0, seed=f)
        hdr = jnp.asarray(scene(i).view(**view))
        if w > 1e-6:
            hdr = (1 - w) * hdr + w * jnp.asarray(scene(i + 1).view(**view))
        rgb = np.asarray(grade_device(hdr))
        t_myr = (1 - w) * times[i] + w * times[min(i + 1, len(times) - 1)]
        label = args.label or scene(i).meta["tag"].lstrip("_")
        R.annotate(rgb, f"{label}  t = {t_myr:.1f} Myr").save(path)
        if f % 10 == 0:
            print(f"frame {f}/{n_frames}  t = {t_myr:.1f} Myr  ({time.time() - start:.0f} s)", flush=True)
    R.encode_movie(frame_dir, args.out, args.fps)


if __name__ == "__main__":
    main()
