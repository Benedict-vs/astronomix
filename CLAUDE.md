# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`astronomix` (formerly `jf1uids`) is a differentiable (magneto)hydrodynamics code written in JAX for
astrophysical applications. It runs 1D/2D/3D hydro and MHD simulations, scales to multiple GPUs, and is
end-to-end differentiable (forward and backward) for gradient-based inverse modeling and solver-in-the-loop
training. Packaged with Poetry; current version is in `pyproject.toml`.

## Commands

Tests are standalone scripts, **not** a `pytest` suite despite the directory name — most do GPU selection at
import time and write figures/data as a side effect. Run an individual test directly:

```bash
python pytests/hydrodynamics/shock_tube1D.py
python pytests/mhd/alfven_wave3D.py --convergence --scaling   # many take CLI flags (--sp, --dp, --scaling, ...)
```

Everything test-related lives under `pytests/`, organized by physics (`hydrodynamics/`, `mhd/`, `self_gravity/`,
`viscosity/`, `differentiability/`), with committed reference data under `pytests/*/data/` and figures under
`pytests/*/figures/`. The top-level `tests/` directory is gone — upstream retired it, and this fork's cgols
research scripts now live in `pytests/self_gravity/cgols/` (see its `run_*.sh` for the HoreKa/dev batch jobs;
its `data/` is a symlink to scratch and its `cgols_snapshots*` / `cgols_logs` outputs are gitignored).

- Lint: `ruff check` (config in `.ruff.toml` / `pyproject.toml`; selects E,F,I,C,B,D,Q and ignores E402 because
  GPU selection via `autocvd` must run before imports).
- Performance probe: `python _bench_perf.py <config_name> <out_json>` — one config per process.
- `make version v=<x.y.z>` bumps the Poetry version, commits, tags, and pushes (release workflow).

GPU selection: scripts call `autocvd(num_gpus=N)` at the top before importing JAX, followed by `# ruff: noqa: E402`.
Follow this pattern when writing new runnable scripts.

## Core architecture

A simulation is driven by four objects, all passed explicitly through the API (no global state):

- **`SimulationConfig`** (`option_classes/simulation_config.py`) — a `NamedTuple` of **static** options that, when
  changed, trigger JAX recompilation (it is a `static_argnames` to the jitted core). This is the central control
  surface: backend, solver mode, geometry, dimensionality, boundaries, which physics modules and snapshot outputs
  are active. Run it through `finalize_config(config, state.shape)` after building the initial state.
- **`SimulationParams`** (`option_classes/simulation_params.py`) — runtime values (e.g. `t_end`, `gamma`, CFL number)
  that can change **without** recompilation.
- **`RegisteredVariables`** (`variable_registry/registered_variables.py`) — built by `get_registered_variables(config)`.
  The state is a single array whose first axis indexes physical variables; you index it via registry fields
  (`state[registered_variables.density_index]`, etc.). The set of variables depends on config (MHD adds B-fields,
  cosmic rays add a component, ...). Always go through the registry — never hardcode variable indices.
- **`HelperData`** (`data_classes/simulation_helper_data.py`) — geometry/grid arrays (cell centers, volumes, ...),
  built by `get_helper_data(config)`. `_helper_data_requirements(config)` builds only the fields the active
  subsystems need.

Typical flow (see README "Hello World"): build `config` → `get_registered_variables` → `get_helper_data` →
`construct_primitive_state(...)` → `finalize_config` → `time_integration(state, config, params, registered_variables)`.

`time_integration` (`time_stepping/time_integration.py`) is a thin wrapper that sets up sharding, optional state
donation, runtime checking (`checkify`), and memory analysis, then calls the jitted `_time_integration`. The generic
stepping loop lives in `time_stepping/_time_loop.py` (`FIXED_STEP`, `ADAPTIVE_WHILE`, `ADAPTIVE_CHECKPOINTED`).

### Two solver schemes (`solver_mode` in config)

- **`FINITE_VOLUME`** (`_finite_volume/`) — Riemann-solver based (LF, HLL, HLLC, HLLC-LM, hybrid/AM-HLLC). Entry point
  `_evolve_state_fv`.
- **`FINITE_DIFFERENCE`** (`_finite_difference/`) — 5th-order WENO with constrained transport for MHD (HOW-MHD). Entry
  point `_evolve_state_fd`.

Each scheme directory mirrors the same substructure: `_state_evolution`, `_timestep_estimation`, `_magnetic_update`,
and (FD) `_interface_fluxes` / (FV) `_riemann_solver`.

### Two compute backends (`config.backend_config.backend`)

Backend choice and the Pallas/Triton kernel knobs live in the nested `BackendConfig` sub-struct
(`SimulationConfig(backend_config=BackendConfig(backend=PALLAS, pallas_block_shape=(4, 4, 8), ...))`), alongside
`PositivityConfig` / `GravityConfig`.

- **`NATIVE_JAX`** — plain JAX/XLA.
- **`PALLAS`** — fused Pallas kernels, dramatically lower memory and faster (see tables in
  `pallas_backend_implementation_guide.md`).

The pattern: a `*_native` function written in idiomatic JAX is the source of truth; a `*_pallas` sibling is a hand-
or skill-translated kernel that must match it bit-for-bit (to rounding noise). **Do not hand-edit `_pallas` files** —
regenerate them from the native function using the `pallasify` skill, following
`pallas_backend_implementation_guide.md`. Boundary handling interacts with the backend: `GHOST_CELLS` vs
`PERIODIC_ROLL`, the latter avoiding padding for periodic domains.

### Physics modules (`_modules/`)

Optional subsystems gated by config flags: `_gravity` (self-gravity / external potential — note the novel
"conservative" self-gravity scheme), `_stellar_wind`, `_cooling`, `_cosmic_rays`, `_turbulent_forcing`, `_viscosity`,
plus neural correctors (`_cnn_mhd_corrector`, `_neural_net_force`). `_time_integrator_sources.py` and
`_iteration_level_updates.py` wire these into the step. Each module typically carries its own `*_options.py` config
sub-struct embedded in `SimulationConfig`.

### Snapshotting (`_snapshotting/`)

Two mechanisms (see README FAQ): on-device snapshots via `return_snapshots`/`SnapshotSettings` (the various
`return_*` flags select what to store — full states, energies, spectra, divergence, ...), or a user
`snapshot_callable` that offloads to host via `jax.debug.callback`.

## Conventions

- All physics code must be JAX-traceable (no Python side effects in hot paths; use `jax.debug.callback` to reach the
  host). The whole solver is jitted with `config` and `registered_variables` as static arguments.
- `equinox` and `beartype`/`jaxtyping` are used for typed array containers; shape annotations like
  `Float[Array, "num_vars num_cells_x num_cells_y num_cells_z"]` appear throughout.
- Reusable initial conditions live in `astronomix/test_setups/` (importable, e.g. `setup_sod_shock_tube`); test
  scripts compose these rather than re-deriving ICs.
- Documentation site: https://astronomix-mhd.web.app/ (Firebase-hosted, config in `firebase.json`/`.firebaserc`).
