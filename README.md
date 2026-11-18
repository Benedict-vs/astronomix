# astronomix - differentiable mhd for astrophysics in JAX

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/848159116.svg)](https://doi.org/10.5281/zenodo.15052815)

`astronomix` (formerly `jf1uids`) is a differentiable hydrodynamics and magnetohydrodynamics code
written in `JAX` with a focus on astrophysical applications. `astronomix` is easy to use, well-suited for
fast method development, scales to multiple GPUs and its differentiability 
opens the door for gradient-based inverse modeling and sampling as well 
as surrogate / solver-in-the-loop training.

## Features

- [x] 1D, 2D and 3D hydrodynamics (HD) and magnetohydrodynamics (MHD) simulations scaling to multiple GPUs and nodes
- [x] fifth order WENO-JS and WENO-Z finite difference schemes (FD) for both HD and MHD simulations (adiabtatic and isothermal)
- [x] for the fifth order finite difference scheme a constrained transport MHD approach following [HOW-MHD by Seo & Ryu 2023](https://arxiv.org/abs/2304.04360)
- [x] for the fifth order finite difference scheme a positivity-preserving reconstruction (based on [Zhang and Shu, 2012](https://www.sciencedirect.com/science/article/pii/S0021999111006759) and [Wu, 2018](https://arxiv.org/abs/1802.02278)) that keeps density and pressure positive for `C_cfl <= 0.75` without floors, additionally slight dissipation for the electric field can be activated for stability in extreme flows (where the cell-centered WENO dissipation does not translate into enough damping of the interface electric field at the Nyquist scale)
- [x] for the finite difference scheme a novel semi-discretely energy conserving self-gravity scheme
- [x] a re-implementation of [AthenaPK](https://github.com/parthenon-hpc-lab/athenapk)'s default second-order finite volume (FV) scheme (VL2 predictor-corrector, PLM, HLLD / HLLE / HLLC, GLM divergence cleaning, first-order flux correction), agreeing with AthenaPK to round-off ([validation](examples/scripts/validation/athenapk_vl2)), further supported Riemann solvers include Lax-Friedrichs, HLL, HLLC-LM ([Fleischmann et al., 2020](https://www.sciencedirect.com/science/article/pii/S0021999120305362)), HYBRID-HLLC & AM-HLLC ([Hu et al., 2025](https://www.sciencedirect.com/science/article/pii/S1007570425005891))
- [x] backwards and forwards differentiable with adaptive timestepping
- [x] physics modules: turbulent driving (white-in-time and Ornstein-Uhlenbeck, AthenaK / AthenaPK driving spectra), radiative cooling with an implicit solver, thermal conduction, viscosity and resistivity, passive scalars (e.g. composition tracers), dual energy, stellar winds
- [x] spherically symmetric simulations such that mass and energy are conserved based on the scheme of [Crittenden and Balachandar (2018)](https://doi.org/10.1007/s00193-017-0784-y)
- [x] Pallas (Triton) GPU kernels for the finite difference and the VL2 finite volume schemes, multi-GPU via `shard_map` halo exchange
- [x] easily extensible, all code is open source
- [x] **(this fork)** stiff astrochemistry + thermochemistry per cell via [carbox](https://github.com/GijsVermarien/carbox) reaction networks (Diffrax Kvaerno5, chunked, sharded, or a per-cell Rosenbrock23 Warp kernel built with [gradsolve](https://github.com/ECLIPSE-AI4Science/gradsolve)), and a **neural chemistry emulator** that replaces the stiff solve with a trained dense network (~45x lower wall time at 160^3), see [Chemistry and the neural emulator](#chemistry-and-the-neural-chemistry-emulator-this-fork)

## Contents

- [Features](#features)
- [Installation](#installation)
- [Hello World! Your first astronomix simulation](#hello-world-your-first-astronomix-simulation)
- [Examples for Getting Started](#examples-for-getting-started)
- [Showcase](#showcase)
- [Chemistry and the neural chemistry emulator (this fork)](#chemistry-and-the-neural-chemistry-emulator-this-fork)
- [Performance](#performance)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
  - [How to store intermediate information during the simulation?](#how-to-store-intermediate-information-during-the-simulation)
  - [How to fit larger simulations into GPU memory?](#how-to-fit-larger-simulations-into-gpu-memory)
  - [What if my simulation crashes?](#what-if-my-simulation-crashes)
- [Documentation](#documentation)
- [Citing astronomix](#citing-astronomix)

## Installation

`astronomix` can be installed via `pip`

```bash
pip install astronomix
```

Note that if `JAX` is not yet installed, only the CPU version of `JAX` will be installed
as a dependency. For a GPU-compatible installation of `JAX`, please refer to the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Hello World! Your first astronomix simulation

Below is a minimal example of a 1D hydrodynamics shock tube simulation using `astronomix`.

```python
import jax.numpy as jnp
from astronomix import (
    SimulationConfig, SimulationParams,
    get_helper_data, finalize_config,
    get_registered_variables, construct_primitive_state,
    time_integration
)

# the SimulationConfig holds static 
# configuration parameters
config = SimulationConfig(
    box_size = 1.0,
    num_cells = 101,
    progress_bar = True
)

# the SimulationParams can be changed
# without causing re-compilation
params = SimulationParams(
    t_end = 0.2,
)

# the variable registry allows for the principled
# access of simulation variables from the state array
registered_variables = get_registered_variables(config)

# next we set up the initial state using the helper data
helper_data = get_helper_data(config)
shock_pos = 0.5
r = helper_data.geometric_centers
rho = jnp.where(r < shock_pos, 1.0, 0.125)
u = jnp.zeros_like(r)
p = jnp.where(r < shock_pos, 1.0, 0.1)

# get initial state
initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = u,
    gas_pressure = p,
)

# finalize and check the config
config = finalize_config(config, initial_state.shape)

# now we run the simulation
final_state = time_integration(initial_state, config, params, registered_variables)

# the final_state holds the final primitive state, the 
# variables can be accessed via the registered_variables
rho_final = final_state[registered_variables.density_index]
u_final = final_state[registered_variables.velocity_index]
p_final = final_state[registered_variables.pressure_index]
```

You've just run your first `astronomix` simulation! You can continue with
the notebooks below and we have also prepared a more advanced use-case
(stellar wind in driven MHD tubulence) which you can
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Pg98IPGnoejaGvzmNNZiwmf1JnXwYAJH?usp=sharing).

## Examples for Getting Started

Runnable scripts live under `examples/scripts/`, notebooks under
`examples/notebooks/`.

- forward simulations
  - [1D Sod shock tube](examples/scripts/forward/hydro/shock_tube.py), [a minimal notebook](examples/notebooks/hydro/simple_example.ipynb)
  - [2D Kelvin-Helmholtz instability](examples/scripts/forward/hydro/khi.py) ([notebook](examples/notebooks/hydro/kelvin_helmholtz.ipynb))
  - [2D Orszag-Tang vortex](examples/scripts/forward/mhd/orszag_tang.py) ([notebook](examples/notebooks/mhd/orszag_tang_vortex.ipynb))
  - [3D MHD jet](examples/scripts/forward/mhd/jet.py) ([notebook](examples/notebooks/mhd/jet.ipynb))
  - [3D driven MHD turbulence and the small-scale dynamo](examples/scripts/forward/mhd/turbulence) (astronomix against AthenaPK)
  - [3D self-gravitating collapse](examples/scripts/forward/self_gravity/collapse.py) ([notebook](examples/notebooks/gravity/evrards_collapse.ipynb))
- differentiability
  - [Field-level inference](examples/scripts/differentiability/field_level_inference) ([notebook](examples/notebooks/differentiability/field_level_inference.ipynb))
  - [KHI eigenmode initialization](examples/scripts/differentiability/eigen_initialization.py)
  - [Solver-in-the-loop correction network](examples/scripts/differentiability/solver_in_the_loop.py)
- output options
  - [On-the-fly movie via callback](examples/scripts/output_options/callback.py)
  - [Orbax checkpointing and restart](examples/scripts/output_options/orbax_checkpointing.py)
  - [In-memory snapshot diagnostics](examples/scripts/output_options/return_snapshots.py)
- multi-GPU
  - [Sharded multi-GPU run](examples/scripts/multi_gpu/multi_gpu.py)
  - [Multi-node run](examples/scripts/multi_gpu/multi_node.py)
- validation
  - [AthenaPK's VL2 scheme in astronomix](examples/scripts/validation/athenapk_vl2)
  - [Positivity-preserving WENO](examples/scripts/validation/weno_stability)

More involved, physics-module showcase scripts — turbulent radiative mixing
layer, stellar wind, cooling — live in [`examples/gallery/`](examples/gallery).

## Showcase

| ![wind in driven turbulence](readme_resources/readme_figures/interm_driven_turb_wind4.png) |
|:---------------------------------------------------------------------------------:|
| Magnetohydrodynamics simulation with driven turbulence at a resolution of 512³ cells in a fifth order CT MHD scheme run on 4 H200 GPUs. |

| ![wind in driven turbulence](readme_resources/readme_figures/driven_turb_wind4.png) |
|:---------------------------------------------------------------------------------:|
| Magnetohydrodynamics simulation with driven turbulence and stellar wind at a resolution of 512³ cells in a fifth order CT MHD scheme run on 4 H200 GPUs. |

| ![3D MHD jet](readme_resources/readme_figures/mhd_jet.png) |
|:----------------------------------------------------------:|
| 3D MHD jet propagating into a magnetized medium. |

| ![Semi-Discretely Energy Conserving Self Gravity Scheme](readme_resources/readme_figures/collapse_energy_evolution_comparison.svg) |
|:-----------------------------------------------------------------------------------------------------------:|
| Novel semi-discretely energy conserving self-gravity scheme. |

## Chemistry and the neural chemistry emulator (this fork)

This fork (`lorenzobranca/astronomix`, branch `feature/multi-gpu-sharding`) adds a
chemistry module to the finite-volume solver and the multi-GPU plumbing it needs.
It was developed for a molecular-cloud-collision ground truth with MHD + self-gravity +
stiff chemistry; the design notes and every measured number live in that project's
`CONTEXT.md`.

### Chemistry quick start (carbox + gradsolve + the network)

Everything the chemistry needs is pinned from this repository: our carbox version as the git
submodule `external/carbox`, which also holds the network file, and gradsolve, Warp and JAX in
`requirements-chemistry.txt`. With Python 3.12:

```bash
git clone -b feature/multi-gpu-sharding https://github.com/lorenzobranca/astronomix.git
cd astronomix
git submodule update --init          # fetches external/carbox
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-chemistry.txt -e . -e external/carbox
# NVIDIA GPU (CUDA 12), in addition: pip install "jax[cuda12]==0.10.2"

python examples/chemistry_cells.py
python -m pytest pytests/chemistry/test_warp_stiff_solver.py
```

- **Start with `examples/chemistry_cells.py`.** It loads the network, sets up a row of cells
  from `n_H` = 10 to 1e4 cm^-3 without any hydrodynamics, advances them with the vmapped
  Kvaerno5 and with the per-cell Warp kernel that gradsolve generates from the same right-hand
  side, and prints both results and their difference. It takes about a minute on a CPU; the
  script is the same on a GPU.
- **The network file** is `external/carbox/data/simple_latent_tgas.csv`.
- Do not clone with `--recurse-submodules`: carbox has a submodule of its own (uclchem, not
  needed here) that is fetched over SSH and fails without a GitHub SSH key.
- Use the pinned carbox, not carbox `main`: see the next but one section for why.
- The same steps as a plain text hand-out, with the expected output, the parameters of our
  runs and common problems: [`CHEMISTRY_SETUP_INSTRUCTIONS.txt`](CHEMISTRY_SETUP_INSTRUCTIONS.txt).

### Stiff chemistry and thermochemistry

- A [carbox](https://github.com/GijsVermarien/carbox) reaction network is registered as a
  contiguous block of species number densities in the state array
  (`ChemistryConfig`/`ChemistryParams` in `astronomix/_modules/_chemistry/`).
- Every hydro step, `update_chemistry` advances each cell with Diffrax `Kvaerno5` (vmapped,
  optionally in sequential chunks via `reaction_chunk_size` to bound memory). With
  `thermochemistry=True` the temperature is integrated alongside the species (heating:
  cosmic rays, H2 formation; cooling: Glover & Abel H2, [C II], [O I], optional
  Neufeld & Kaufman CO table) and written back to the pressure.
- Setup in one call:

```python
from astronomix.setup_helpers import build_chemistry_from_network_file

chemistry_config, chemistry_params, species_names = build_chemistry_from_network_file(
    "network.csv", "latent_tgas",
    number_density_unit_cgs=..., temperature_unit_kelvin=..., time_unit_seconds=...,
    cosmic_ray_rate=3e-17, fuv_field=1e-4, visual_extinction=2.0,
    thermochemistry=True, reaction_chunk_size=131072,
    absolute_tolerance=1e-12, relative_tolerance=1e-6, max_steps=512,
)
config = SimulationConfig(..., chemistry_config=chemistry_config)
params = SimulationParams(..., chemistry_params=chemistry_params)
```

- `ChemistryParams.cooling_courant` clips the per-step fractional temperature change of the
  thermochemistry. **Do not use it as a stabiliser**: it silently sets the thermal state of the
  whole cloud (dense gas parked at hundreds of K, dt-dependent). Set it to a large value
  (e.g. 10) once the CFL estimate is nan-safe (below), which is what made it unnecessary.

### carbox: what it provides, which version, where the network is

The reaction network comes from [carbox](https://github.com/GijsVermarien/carbox)
(`GijsVermarien/carbox`). It is not a dependency of `astronomix` itself: only the setup helper
imports it, so `import astronomix` works without it.

- **What is coupled.** `build_chemistry_from_network_file` parses a network CSV with
  `carbox.parsers.parse_chemical_network`, builds a `carbox.network.Network` and stores its
  right-hand side (a `JNetwork`) in `ChemistryParams`. `update_chemistry` calls it as
  `network(t, n, T, cosmic_ray_rate, fuv_field, A_V, rate_modifier_a, rate_modifier_b)` with
  number densities in cm^-3. carbox supplies the reaction rates; the heating and cooling terms
  are astronomix's own (`astronomix/_modules/_chemistry/_thermochemistry.py`).
- **Version.** The module is written against carbox commit
  [`1578667`](https://github.com/GijsVermarien/carbox/commit/1578667aec5aaa1b039b448eaf761689470334e7)
  (branch `fix/vectorized-rate-order`, carbox 0.2.0), pinned in this repository as the git
  submodule `external/carbox`. This is our own carbox version: it is kept as a branch of the
  carbox repository and is not part of carbox `main`. It carries two corrections the 16-species
  network needs:
  the vectorised rates are put back into the column order of the incidence matrix (the network
  file interleaves mechanisms, and without this the rates of rows 9-21 of its 25 land on other
  reactions), and the `latent_tgas` parser builds H2 formation on grains as an `H2FormReaction`
  instead of a cosmic-ray reaction. carbox `main` (`745079d`, July 2026) does not work with this
  module: its network takes fractional abundances and a density, has no rate-modifier
  arguments, and its `latent_tgas` parser still maps `H2Form` to a cosmic-ray reaction.
- **Install.** See the quick start above (`pip install -e external/carbox`).
- **gradsolve.** The per-cell kernel of the next section is generated by
  [gradsolve](https://github.com/ECLIPSE-AI4Science/gradsolve) 0.2.1, pinned by commit in
  `requirements-chemistry.txt`. carbox has a branch with the same kernel built in
  (`feature/fused-warp-solver`, `carbox.fused`), but that branch is based on carbox `main` and
  so builds no H2 formation on grains from this network file; it is not the version to use
  with this network.
- **Network.** `external/carbox/data/simple_latent_tgas.csv` (format `"latent_tgas"`): 16
  species (C, C+, CO, CO+, e-, H, H+, H2, H2O, H2O+, H3O+, HCO+, O, O+, OH, OH+) and 25 rate
  terms (two-body reactions, cosmic-ray and FUV processes, H2 formation on grains). The file
  sits outside the carbox Python package, so installing carbox from its URL with `pip` does
  not ship it; the submodule or a clone does.
- **Licences.** carbox is GPL-3.0-or-later and astronomix is MIT, which is why carbox is
  referenced as a submodule and not copied into this repository.

### Networks in the KROME format

`build_chemistry_from_network_file(path, "krome", ...)` reads a network written for
[KROME](https://bitbucket.org/tgrassi/krome) (for instance its `networks/react_galaxy_ism`)
and integrates it with the solvers of this module, doing with it what KROME does: the rate
coefficients, the photo-reactions from a binned spectrum, the shielding, and KROME's heating
and cooling. carbox is not needed for it; the reader is
`astronomix/_modules/_chemistry/_krome_network.py`.

**How to run it.** The temperature is evolved with the abundances in everything below
(`thermochemistry=True`). You need the network file and the `data` folder of a KROME
installation; neither is shipped here.

1. *On a row of cells, without hydrodynamics* (both solvers and their difference):

   ```bash
   python examples/chemistry_krome_network.py \
       --network /path/to/krome/networks/react_galaxy_ism --krome-data /path/to/krome/data \
       --draine 1 --shield-habing-dust --no-metal-table
   ```

   `--draine 1` is the interstellar radiation field in ten energy bins and
   `--shield-habing-dust` its attenuation by dust. Without `--no-metal-table` the metals also
   cool above 1e4 K through KROME's table, in both solvers. `--user-variable NAME=VALUE` sets a
   `@common` variable of the file, `--help` lists the rest. The first build of the per-cell
   kernel takes a minute or two and is cached afterwards; `--no-kernel` skips it.
2. *In a simulation*: build the chemistry with the call below and hand it over as
   `SimulationConfig(..., chemistry_config=chemistry_config)` and
   `SimulationParams(..., chemistry_params=chemistry_params)`. Species `i` of
   `species_names` is the field `registered_variables.chemistry_species_index + i` of the
   state, a number density in units of `number_density_unit_cgs`.
3. *With the per-cell kernel (gradsolve)*: attach the kernel as in the section on the
   per-cell implicit solver further down; the function `build` of
   `examples/chemistry_krome_network.py` does exactly this.

```python
from astronomix._modules._chemistry._krome_photo import photo_spectrum

chemistry_config, chemistry_params, species_names = build_chemistry_from_network_file(
    "/path/to/krome/networks/react_galaxy_ism", "krome",
    number_density_unit_cgs=..., temperature_unit_kelvin=..., time_unit_seconds=...,
    cosmic_ray_rate=3e-17,                      # KROME's user_crate
    thermochemistry=True,                       # heating and cooling as in KROME
    krome_data_folder="/path/to/krome/data",    # metal lines, cooling table, cross sections
    krome_photo_bins=photo_spectrum("draine", 5.0, 13.6, 10),   # -photoBins=10, krome_set_photoBin_draineLin
    krome_h2_formation_on_dust=True,            # -useDustH2const
    krome_shield_habing_dust=True,              # -shieldHabingDust
    krome_shielding="R14",                      # -shielding=R14
    krome_column_density_method="DEFAULT",      # -columnDensityMethod
)
```

- **What is read.** Reaction lines under the default layout or a `@format:` line; temperature
  limits (`NONE`, `.LE.5.5e3`, `>5.5e3`, a bare number), outside which a coefficient is zero;
  `@var:` and `@common:` (the values of the user variables other than `user_crate` go in
  `krome_user_variables`); rate coefficients as Fortran expressions of `Tgas` and KROME's
  shortcuts (`Te`, `lnTe`, `invT`, `T32`, ...); recombination on dust of H+, He+, C+, O+ and
  Si+ and the Lyman-Werner rate `H2_solomonLW`; the photo block, the cosmic-ray block, and
  `@reactionModifier` blocks (`k(i) = ...`). Every reaction is checked for the conservation
  of nuclei and charge.
- **As KROME reads it.** A line whose number of fields does not match the layout in force
  (for instance after a commented `@format:` line) is left out with a warning, as KROME does,
  and the reactions are numbered in the order in which they are read: that number is the
  KROME reaction index, whatever the first column says. Any other directive that the reader
  does not know raises an error.
- **Radiation.** As in KROME the radiation field is a spectrum in energy bins,
  `krome_photo_bins` (edges `left`, `right` in eV and the `intensity` of every bin in
  eV s^-1 cm^-2 sr^-1 Hz^-1; `photo_spectrum` builds KROME's Draine, black-body and J21
  spectra on linear or logarithmic bins, with `krome_photoBin_scale` as `scale=` or a target
  Habing flux as `habing=`). From it follow, as in KROME's `calc_photoBins`, the rate of every
  photo-reaction, its heating, and the Habing flux of the dust terms (the flux between 6 and
  13.6 eV in units of 1.6e-3 erg cm^-2 s^-1; `fuv_field` is then not used). The cross
  sections are those of KROME: `auto` takes the fit of Verner et al. (1996) from KROME's
  database, `@xsecFile=SWRI` and `@xsecFile=LEIDEN` the tables that KROME derives from the
  SwRI and Leiden files of its database (or the tables of a KROME build folder,
  `krome_xsec_folder`), anything else is the expression of `energy_eV` in the file. Without
  bins the photo-reactions take their rates from `krome_photo_rates` (by KROME reaction
  index, zero when not given) and `fuv_field` is the Habing flux.
- **Shielding.** `krome_shield_habing_dust` attenuates the Habing flux by dust,
  `exp(-1.33e-21 D/D_sun N_H)`. `shield_dust(n,Tgas,gamma)` and `krome_fshield(n,Tgas)` can
  be used in the rate expressions, `@var` lines and reaction modifiers of a file, as KROME's
  own networks do for the photo-reactions; `krome_shielding` selects the H2 self-shielding
  that `krome_fshield` returns (`"DB96"`, `"WG11"`, `"R14"`), and as in KROME it has no
  effect on a file that never calls it. `krome_column_density_method` is the column density
  of both: `"DEFAULT"` (`1.87e21 (n / 1e3)^(2/3)` cm^-2) or `"JEANS"` (half a Jeans length).
- **Heating and cooling.** With `thermochemistry=True` a KROME network is heated and cooled
  as in KROME (`_krome_thermochemistry.py`), not by the terms of the section above. Cooling:
  H and He (Cen 1992: collisional ionisation and excitation, recombination, bremsstrahlung),
  the H2 lines (Glover & Abel 2008, Glover 2015), the lines of the metals below 1e4 K from
  two- and three-level systems, the metal cooling table of Shen et al. (2013) above, and
  Compton cooling. Heating: the photoelectric effect on dust net of the recombination
  cooling, with the dust abundance relative to solar (`dust_to_gas_ratio / 0.01`); the
  chemical term (formation and collisional dissociation of H2); the cosmic rays, through the
  reactions of the cosmic-ray block of the file (4.3 eV per ionisation of H and He, a fit in
  n(H2) for H2, 20 eV otherwise: a file without that block has no cosmic-ray heating, as in
  KROME); and the photo-reactions. As in KROME the cooling at the floor temperature
  (`krome_floor_temperature`, 2.73 K) is subtracted from the cooling. The metal line data and
  the table are read from the `data` folder of a KROME installation (`krome_data_folder`:
  `coolZ.dat`, `coolZ_CIE2012NOUV.dat`); without it the metals cool through [C II] and [O I]
  only.
- **Build options of KROME and their counterparts.** `-useDustH2const`:
  `krome_h2_formation_on_dust`; `-photoBins`: `krome_photo_bins`; `-shieldHabingDust`,
  `-shielding`, `-columnDensityMethod`: the three arguments above; `-gamma`: the adiabatic
  index of the simulation; `-RTOL`, `-ATOL`: `relative_tolerance`, `absolute_tolerance`.
  `-conserve` has no counterpart: the implicit solvers used here keep the nuclei and the
  charge to round-off. Not ported: dust grains as species, X-rays, the opacity that KROME
  returns to a radiative-transfer code, and the other cooling and heating options.
- **Solvers.** The vmapped Kvaerno5 runs with a Newton iteration for these networks: the
  default chord iteration stalls on the ionisation equilibrium of hot, dense gas. The
  per-cell Warp kernel needs `thermochemistry=True`. It looks the metal cooling table up
  like the vmapped solvers: the generated code cannot index a constant array, so the tables
  that a right-hand side indexes are handed to the kernel as one array (2026-11-18;
  `krome_metal_table="kernel"` is an alternative that needs no look-up at one product per
  table value in every call, `krome_metal_table=False` leaves the table out). For
  react_galaxy_ism_plus with the table, on one H200 that another job was using at the same
  time: the kernel builds in 98 s, one solver step costs about 0.2 microseconds per cell,
  and a step of 1e-3 Myr over gas from 30 K to 1e8 K at ionisation equilibrium agrees with
  Kvaerno5 to 2e-3 dex in the abundances and 6e-5 in the temperature
  (`pytests/chemistry/krome_cgols_conditions.py --kernel`). On eight cells from 1 to 1e4
  cm^-3 under the Draine field, with the dust shielding, the H2 self-shielding and cosmic
  rays, the kernel and Kvaerno5 agree to 4e-5 dex in the abundances and 6e-6 in the
  temperature (2e-4 dex and 1e-5 for the same network without its cosmic-ray block, where H+
  decays through the run). The first build of the kernel for a network of this size takes
  about a quarter of a minute on a CPU and a minute and a half on a GPU (it is cached
  afterwards); before 2026-11-18 it took about ten minutes on a CPU and did not finish on
  a GPU, because the Jacobian carried the zero tangents of every direction through each
  division and Warp compiled an adjoint that is never used.
- **Checked against KROME itself**, with the Fortran that KROME generates for
  `react_galaxy_ism` with the metal reactions and the cosmic-ray block on (87 reactions, 16
  species) and the options `-photoBins=10 -shielding=R14 -shieldHabingDust -useDustH2const`,
  for three spectra (Draine, a black body of 3e5 K, a J21 power law), four compositions (cold
  neutral, diffuse, molecular, ionised) and 17 temperatures from 10 K to 1e8 K. With the
  constants read as a Fortran compiler reads them (`single_precision_literals=True`; KROME's
  constants without a `d` exponent are single precision) the 87 rate coefficients agree to
  3e-10; the cross sections per bin, the photo-rates, their heating and the Habing flux to
  1e-14; the dust shielding, the H2 self-shielding and the column densities to 2e-13; the
  total cooling to 3e-8, the total heating and the temperature derivative to 5e-7.
  With the default (constants as the decimal numbers they stand for) the coefficients agree
  to 1e-5. Five cells from 1 to 1e4 cm^-3, starting between 35 K and 2.1e4 K under 1 to 100
  Draine fields and followed to 1e14 s with KROME's own solver (thermochemistry on), agree in
  temperature and in every abundance above 1e-12 n_H to 2e-7. The cross section tables
  derived here from the SwRI and Leiden files reproduce those that KROME writes to 5e-12.
  One place where KROME itself is undefined: its metal-line rates above 1e4 K are returned
  unset; here they are zero.
- `examples/chemistry_krome_network.py --network <file>` runs a network file through both
  solvers with the thermochemistry on (`--krome-data <folder>` for KROME's metal cooling,
  `--draine`, `--shield-habing-dust`, `--shielding` for the radiation and the shielding);
  `pytests/chemistry/test_krome_network.py` tests the reader, the photo bins, the shielding
  and the heating and cooling on small networks and data files written in the test, and a
  full file when `ASTRONOMIX_TEST_KROME_NETWORK` (and `ASTRONOMIX_TEST_KROME_DATA`) point at
  one. No KROME network or data file is shipped here (KROME is GPL-3).

### Per-cell implicit solver on the GPU (Warp kernel, via gradsolve)

- `astronomix/_modules/_chemistry/_warp_stiff.py` is an alternative to the vmapped Diffrax
  solve: an adaptive Rosenbrock23 integrator compiled as one
  [NVIDIA Warp](https://github.com/NVIDIA/warp) kernel with **one GPU thread per cell**, so
  every cell takes its own steps instead of the step count of the stiffest cell of its batch.
  The right-hand side and its analytic Jacobian are translated from the JAX function by
  [gradsolve](https://github.com/ECLIPSE-AI4Science/gradsolve) (jaxpr -> Warp), whose
  Rosenbrock23 scheme and step controller the kernel uses; the module extends the translator
  to the primitives of the carbox network and wraps the kernel with `warp.jax_kernel`, so
  that it is an ordinary JAX primitive inside the jitted, sharded time step.
- It needs `thermochemistry=True` and the optional packages `gradsolve` and `warp-lang`
  (`pip install 'gradsolve[warp]'`; developed against gradsolve 0.2.1 and warp-lang 1.18).
  The kernel closes over the concrete chemistry parameters, so it is built once at set-up
  time and carried in the config:

```python
from astronomix._modules._chemistry._chemistry import make_vector_field
from astronomix._modules._chemistry._warp_stiff import build_fused_core_solver

cc, cp = chemistry_config, chemistry_params
vector_field = make_vector_field(
    cp.network, cp.rate_modifier_a, cp.rate_modifier_b, cp.cosmic_ray_rate, cp.fuv_field,
    cp.visual_extinction, cc.thermochemistry, gamma,  # gamma: the adiabatic index of the run
    cp.dust_to_gas_ratio, cp.hydrogen_molecule_formation_rate_coefficient,
    cc.hydrogen_index, cc.molecular_hydrogen_index, cc.electron_index, cc.atomic_oxygen_index,
    cc.ionized_hydrogen_index, cc.helium_index, cc.ionized_carbon_index, cc.co_cooling,
    cc.carbon_monoxide_index, cp.co_cooling_table, cp.co_cooling_bounds,
)
solver = build_fused_core_solver(
    vector_field, cc.number_of_chemical_species + 1,
    rtol=float(cp.rtol), atol=float(cp.atol), max_steps=cc.max_steps,
)
chemistry_config = chemistry_config._replace(stiff_core_solver="warp", stiff_core_fn=solver)
```

- Without an emulator, `update_chemistry` then advances **every cell** with the kernel. With
  the emulator and `stiff_core_capacity > 0`, the kernel solves the gathered dense cells and
  the emulator the rest. A cell that returns a non-finite state keeps its previous one, as
  with Diffrax and `throw=False`.
- Measured on the 256^3 cloud collision (16 species + T, rtol 1e-6, atol 1e-12): a run with
  the kernel in every cell reproduces the Kvaerno5 reference at 3.65 Myr to a relative L2
  difference of 9e-6 in density, and one chemistry update of the whole grid takes 2.8-3.5 s
  on one H200 against 118-122 s for the vmapped Kvaerno5.
- Tests: `pytest pytests/chemistry/test_warp_stiff_solver.py` (on the CPU, about a minute):
  the kernel on the Robertson problem, the translated field and Jacobian of a small
  carbox-like network against JAX, and `update_chemistry` with the kernel against the
  vmapped Kvaerno5 on the carbox network of `external/carbox` (skipped when carbox is not
  installed or the network file is not found).

### Multi-GPU (sharded) finite volume

- Pass a `NamedSharding` of the state (X axis split) to `time_integration`; the FV stencils run
  inside a `shard_map` with rolls turned into halo exchanges (`_finite_volume/_sharding.py`).
- The MHD magnetic update's eigen-iteration reduces its convergence test across devices
  (`distributed_max`); without it the devices disagree on the trip count and deadlock in
  `kCollectivePermute`.
- On some nodes NCCL's NVLS multicast fails: set `NCCL_NVLS_ENABLE=0`.

### Robustness of the coupled run

- The FV CFL estimate ignores non-finite per-cell wave speeds and guards the final `dt`; before
  that, one cell with a negative/non-finite pressure made the global timestep nan and the
  whole grid was floored by the nan backstop in the next step.
- The thermochemistry nan backstop resets any non-finite cell to a floored rest state and
  prints `NANREPAIR bad_cells=N` when it fires (a steady trickle is local; a jump to the grid
  size is global).

### The neural chemistry emulator

The per-cell operator `(species, T, dt) -> (species, T)` can be replaced by a trained
network exported to an `npz` holding the standardisation of the 17 quantities
`[log10 x_i, log10 T]` and of `log10 n_H`, the time grid whose index fraction is the time
input, the species order, the activation and residual flag, and the weights of one of two
CODES architectures:

- `FullyConnected` / `FullyConnectedResidual`: dense layers `W<i>`/`b<i>` on
  `[state, tau, log10 n_H]` (`architecture` absent or `"fcnn"`);
- `MultiONet` / `MultiONetResidual`: a branch net `branch_W<i>`/`branch_b<i>` on
  `[state, log10 n_H]` and a trunk net `trunk_W<i>`/`trunk_b<i>` on `[tau]`, whose
  outputs are split per quantity and dotted (`architecture = "multionet"`).

The exporters live in the cloud-collision project (`export_fcnn_to_npz.py`,
`export_multionet_to_npz.py`), and `verify_multionet_coupling.py` there checks the JAX
path against the PyTorch model to float64 round-off:

```python
from astronomix.setup_helpers.chemistry_setup import attach_emulator

chemistry_config, chemistry_params = attach_emulator(
    chemistry_config, chemistry_params, "fcresidual.npz", project_conservation=True,
)
```

`update_chemistry` then dispatches to `_emulate_single_cell` under the same vmap/chunking/
writeback. Because an unbounded regressor drifts where it was not trained, the step is
guarded: every element is conserved per cell by rescaling its species to the budget the cell
entered with, electrons are reset to charge neutrality, the standardised inputs are clipped
to `emulator_input_clip_sigma` (5) and the temperature change per step is capped
(`emulator_max_log_temperature_change`, 3 dex). Without the element projection a run
diverged after ~400 steps from C/O species running away in a few void cells.

Measured at 160^3 (MHD + gravity + 16-species carbox network, 3.65 Myr, ~400 steps), warm-
started from one stiff segment: peak density within 1% of the stiff reference at every
segment, final density field within 1.3% (rel. L2), temperature field median difference
4e-5, H/H2 fractions within 5%, at 48x lower wall time (the run becomes hydro-bound).
Two caveats: the emulator must have been trained on the compositions the run visits
(a model trained on evolved states only could not start from a freshly seeded initial
condition and drifted in chemical age), and the first step from a pristine seed is still
best done with the stiff solver (`--restart` from a one-segment stiff checkpoint).

## Performance

Methods paper incoming :)

## Frequently Asked Questions (FAQ)

### How to store intermediate information during the simulation?

There are two main options for storing intermediate states:

- activate `return_snapshots` in the `SimulationConfig`, set either `num_snapshots` in the `SimulationConfig` for equidistant snapshots or `snapshot_timepoints` in the `SimulationParams` for custom snapshot timepoints, then activate specific variables to be stored via `snapshot_settings` in the `SimulationConfig` (e.g. the full states, the total energy, ..., see [here](https://astronomix-mhd.web.app/source/astronomix.option_classes.simulation_config.html#astronomix.option_classes.simulation_config.SnapshotSettings)) - these snapshots are stored on the GPU directly, which avoids host-device transfer and is e.g. very useful for losses over intermediate states
- activate `snapshot_callable` in the `SimulationConfig` and provide a callable to the `time_integration` function which is called at the same timepoints as the snapshots in the first option, this function can then offload specific data (possibly after processing) to the host via `jax.debug.callback` and save it appropriately (e.g. directly make plots, save to disk in the preferred format, ...)

### How to fit larger simulations into GPU memory?

Make sure that you only have the initial state in GPU memory when you start the simulation,
e.g. have a function which constructs the initial state (otherwise e.g. the intermediate helper data
you used to construct the state might still be in GPU memory). To further save storage,
you can donate the initial state to the time integration (activate `donate_state` in the `SimulationConfig`),
which allows `JAX` to reuse the same memory for the state throughout the simulation (but you can also no
longer access the initial state after the simulation has started).

### What if my simulation crashes?

First of all check if the initial conditions are valid. For the finite
difference scheme, turn on the positivity-preserving WENO reconstruction
(`weno_positivity_preserving=True` in the `SimulationConfig`), which keeps
density and pressure positive for `C_cfl <= 0.75`. The
`PositivityConfig` in the `SimulationConfig` additionally offers a per-step
density / pressure floor.

### The chemistry emulator run drifts away from the stiff reference. Why?

Check the emulator was trained on the states your run actually visits (density,
temperature AND chemical composition/age): an off-manifold input is answered with an
arbitrary output, and in an autoregressive run one such step is enough. Keep
`project_conservation=True`. Warm-start from a short stiff integration when the initial
composition is a synthetic seed.

## Documentation

See [here](https://astronomix-mhd.web.app/).

## Citing astronomix

If you use `astronomix` in your research, please cite via

```bibtex
@misc{storcks_astronomix_2025,
  doi = {10.5281/ZENODO.17782162},
  url = {https://zenodo.org/doi/10.5281/zenodo.17782162},
  author = {Storcks, Leonard},
  title = {astronomix - differentiable MHD in JAX},
  publisher = {Zenodo},
  year = {2025},
  copyright = {MIT License}
}
```

There is also a workshop paper on an earlier stage of the project:

[Storcks, L., & Buck, T. (2024). Differentiable Conservative Radially Symmetric Fluid Simulations and Stellar Winds--jf1uids. arXiv preprint arXiv:2410.23093.](https://arxiv.org/abs/2410.23093)