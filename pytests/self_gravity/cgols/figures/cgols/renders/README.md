# CGOLS volume renders (A-2048, C-1024)

Volume renders of the density field in the style of CGOLS IV Fig. 2 (Schneider et al. 2020,
arXiv:2002.10468, made with NVIDIA IndeX), produced with our own JAX ray marcher.

| file | content |
|---|---|
| `C1024_t{20,35,50,75}Myr_cgols4_1080x1920.png` | model C (16 co-rotating cluster complexes, adiabatic), 1024x1024x2048 |
| `A2048_t{20,35,50,75}Myr_cgols4_1080x1920.png` | model A (central 300 pc injection, adiabatic), full 2048x2048x4096 |
| `A2048_t35Myr_cgols4_2160x3840.png` | 4K hero of A-2048 at the end of the starburst |
| `timeseries_C1024_A2048_cgols4.png` | overview of the eight stills |
| `*_orbit360.mp4` | 360-frame camera orbits at 35 Myr (30 fps, 1080x1920) |

## How they were made

1. `cgols_render_extract.py` reduces an Orbax checkpoint to float16 volumes of log n, log T (mass-weighted)
   and v_r (`data/renders/<tag>_f<factor>/<step>/`; full resolution = factor 1, 103 GB per A-2048 snapshot).
2. `cgols_render.py ... --preset cgols4` ray-marches them front to back (emission + absorption, exact segment
   opacity, trilinear sampling, gradient lighting from the trilinear interpolant), with a shadowed central light
   (single scattering; optical depth from a coarse grid with capped opacity), a faint blue emission haze, ACES
   tone mapping and bloom. Movies: `--orbit N`; ffmpeg lives in `/export/data/bschuber/envs/ffmpeg`.

## Reading the images

- **Colours are density, not an observable.** Blue/black = thin wind, white = denser streaks, peach = dense
  clouds, the disk is drawn as a grey "dust lane" (an opaque body shows its colour at full brightness, so a
  bright disk colour would turn the edge-on disk envelope into a glowing slab).
- **The central light is artistic** (the CGOLS IV figure appears to use one too); it is not a radiative-transfer
  result.
- **Per-run colour shift:** A-2048's wind is ~0.4 dex thinner than C-1024's, so its colours are shifted to match
  (`--auto` / `--wind-shift`, thin gas only; the disk keeps its look). A-2048 also uses `--hot-fade` (hot gas
  more transparent) so its dense, smooth, hot inner free wind does not hide the nozzle.
- **All four times share one brightness scale per run.** After the starburst (SFR 20 -> 5 Msun/yr at 40-45 Myr)
  the volume-filling wind thins ~4-5x (median log n, |z| > 1 kpc), so at 50 and 75 Myr it renders nearly black:
  that fading is real, but exaggerated by the steep (~n^3) opacity. Brightness-normalised late frames
  (`--auto --auto-q 50`; A-2048 `--wind-shift=-1.0 --wind-pivot=-3.0,-1.6`) are planned.
- **No physics outside the box:** the outflow leaves the 10x10x20 kpc domain through the side walls; frames show
  the whole box (no cylindrical cut), so the wind simply runs off the frame edges.
- Both runs are adiabatic: no cool clouds condense out of the mixed gas, unlike in CGOLS IV/V (radiative
  cooling, 10^4 K floor), and our model C injects the CGOLS I loading (12 Msun/yr) rather than CGOLS IV's
  Starburst99 rates (~2 Msun/yr), so its wind is denser.
