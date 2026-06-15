"""Post-simulation analysis for the CGOLS replication.

Runs in its own process so the GPU is empty: cgols.py persists the initial and
final states to .npy, and this script loads them and produces the figures. Doing
the analysis inside cgols.py OOMs because XLA still holds the simulation's ~32 GB
device pool.

Usage:
    python cgols.py            # runs the sim, writes cgols_{initial,final}_state.npy
    python cgols_analyse.py    # reads them back, writes the figures
"""

import numpy as np
import jax.numpy as jnp

# Importing cgols runs autocvd + the module-level constants/functions (cheap), but
# not the simulation itself (guarded behind `if __name__ == "__main__"`).
from cgols import build_config, analyse_results

config, registered_variables = build_config()

final_state = jnp.asarray(np.load("cgols_final_state.npy"))

try:
    initial_state = jnp.asarray(np.load("cgols_initial_state.npy"))
except FileNotFoundError:
    print("cgols_initial_state.npy not found - showing final-state diagnostics only.")
    initial_state = None

analyse_results(final_state, config, registered_variables, initial_state=initial_state)
print("Wrote cgols_static_check.png, cgols_vz_diagnostic.png, cgols_extras.png")
