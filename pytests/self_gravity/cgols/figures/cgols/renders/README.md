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
| `C1024_timeorbit_1-75Myr_cgols4.mp4` | first version: one camera orbit (1 deg/frame) while time runs 1.25 -> 75 Myr: 6 frames per checkpoint, neighbouring checkpoints cross-faded (`cgols_render_timeorbit.py`); the cross-fade ghosts moving gas (~1 kpc per checkpoint) |
| `C1024_timeorbit_1-75Myr_60frames_3deg_elev28_{12,20}fps.mp4` | same time range, one frame per checkpoint (no blending, no repeats), camera at a fixed 28 deg elevation looking down (no up/down sway, so the centre stays visible) turning 3 deg per frame (half an orbit); 12 fps (5 s) and 20 fps (3 s) |
| `A2048_timeorbit_1-75Myr_60frames_3deg_elev28_{12,20}fps.mp4` | the same for A-2048 (2x downsampled volumes, 1024x1024x2048, extracted on HoreKa); hot-gas fade 0.1 -> off, haze 0.3 -> 0.6 and the wind-shift range ramped between 40 and 55 Myr (`--hot-fade-late --glow-late --pivot-late`), so the central fog is cleared during the starburst and the thin late wind stays visible |

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
- **20 and 35 Myr share one brightness scale per run; 50 and 75 Myr are brightness-normalised per frame.**
  After the starburst (SFR 20 -> 5 Msun/yr at 40-45 Myr) the volume-filling wind thins ~4-5x (median log n,
  |z| > 1 kpc) and would render nearly black on the fixed scale (the steep ~n^3 opacity exaggerates the fading).
  The late frames therefore shift the thin gas to the median wind density of that frame (C-1024:
  `--auto --auto-q 50`; A-2048 50 Myr: `--wind-shift=-1.0 --wind-pivot=-3.0,-1.6 --hot-fade 0.3`; A-2048 75 Myr:
  `--wind-shift=-1.3 --wind-pivot=-3.4,-1.6 --glow 0.6`, the most strongly brightened frame). Do not read their
  brightness quantitatively.
- **The time-orbit** uses the fixed look up to 40 Myr and then follows the thinning wind smoothly (median-based
  shift, running mean over 5 checkpoints), so its late brightness is normalised too.
- **No physics outside the box:** the outflow leaves the 10x10x20 kpc domain through the side walls; frames show
  the whole box (no cylindrical cut), so the wind simply runs off the frame edges.
- Both runs are adiabatic: no cool clouds condense out of the mixed gas, unlike in CGOLS IV/V (radiative
  cooling, 10^4 K floor), and our model C injects the CGOLS I loading (12 Msun/yr) rather than CGOLS IV's
  Starburst99 rates (~2 Msun/yr), so its wind is denser.
