"""
Run cgols.py with a probe that reports which VL2 stage path was traced.

The VL2 Pallas stage falls back to the native path silently when
_vl2_pallas_supported() rejects the config (e.g. GHOST_CELLS on a mesh before
the sharded fix), which only shows up as ~2x memory. This prints the trace-time
counts at exit; a sharded run on the Pallas path has pallas_local_builds_on_shard > 0.

    python cgols_vl2_probe.py cgols.py
"""

# general
import atexit
import runpy
import sys

import astronomix._finite_volume._state_evolution._van_leer_integrator as vi
import astronomix._finite_volume._state_evolution._van_leer_pallas as vlp

counts = {"pallas_local_builds_on_shard": 0, "pallas_local_builds_unsharded": 0, "pallas_stage_calls": 0}
_local = vlp._vl2_stage_pallas_local
_stage = vlp._vl2_stage_pallas


def local(*args, **kwargs):
    """Count local Pallas stage builds, split by whether they sit on a shard."""
    layout = kwargs.get("shard_layout")
    on_shard = layout and any(entry is not None for entry in layout)
    counts["pallas_local_builds_on_shard" if on_shard else "pallas_local_builds_unsharded"] += 1
    return _local(*args, **kwargs)


def stage(*args, **kwargs):
    """Count calls of the Pallas stage entry point."""
    counts["pallas_stage_calls"] += 1
    return _stage(*args, **kwargs)


vlp._vl2_stage_pallas_local = local
vi._vl2_stage_pallas = stage
atexit.register(lambda: print("[probe] VL2 stage path (trace-time counts):", counts, flush=True))
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
