"""Post-simulation analysis for the CGOLS replication.

Runs in its own process so the GPU is empty: cgols.py persists the initial and
final states to .npy, and this script loads them and produces the figures. Doing
the analysis inside cgols.py OOMs because XLA still holds the simulation's ~32 GB
device pool.

Usage:
    python cgols.py            # runs the sim, writes cgols_{initial,final}_state.npy
    python cgols_analyse.py    # reads them back, writes the figures
"""

import os

# The analysis only loads the saved .npy states and renders figures - it never
# runs the sharded solver, so it must never claim more than one GPU. cgols.py
# derives autocvd(num_gpus=...) from CGOLS_SHARD_SPLIT at import time (default
# (1, 2, 2, 1) -> 4 GPUs), so pin it to a single device before importing cgols.
# (setdefault lets an explicit CGOLS_SHARD_SPLIT override win, e.g. for debugging.)
os.environ.setdefault("CGOLS_SHARD_SPLIT", "(1, 1, 1, 1)")

import numpy as np
import jax.numpy as jnp

# Importing cgols runs autocvd + the module-level constants/functions (cheap), but
# not the simulation itself (guarded behind `if __name__ == "__main__"`).
from cgols import (
    _here,
    analyse_results,
    animate_wind_snapshots,
    build_config,
    plot_paper_slices,
    plot_wind_timeseries,
)

config, registered_variables = build_config()

# Read the states from the script directory (where cgols.py wrote them), not the
# current working directory, so analysis works regardless of where it is launched.
final_state = jnp.asarray(np.load(_here("cgols_final_state.npy")))

try:
    initial_state = jnp.asarray(np.load(_here("cgols_initial_state.npy")))
except FileNotFoundError:
    print("cgols_initial_state.npy not found - showing final-state diagnostics only.")
    initial_state = None

analyse_results(final_state, config, registered_variables, initial_state=initial_state)
print("Wrote cgols_static_check.png, cgols_vz_diagnostic.png, cgols_extras.png")

# Intermediate-snapshot products (the wind animation + the outflow time-series).
# These read the per-frame .npz files streamed to SNAPSHOTS_DIR/ during the run,
# so they work even if the final state itself is unusable, and even if the run
# blew up partway (the frames already on disk survive). Both functions print a
# message and return if no frames are present.
animate_wind_snapshots(config)
plot_wind_timeseries(config)
# Paper comparison: x-z density & temperature slices at 10/25/50/60 Myr (the last
# two fall back to the capped run's final frame).
plot_paper_slices(config, target_times_myr=(10.0, 25.0, 50.0, 60.0))
