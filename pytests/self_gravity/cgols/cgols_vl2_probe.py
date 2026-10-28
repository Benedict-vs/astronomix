"""
Run cgols.py with a probe that reports which VL2 stage path was traced.

The VL2 Pallas stage falls back to the native path silently when
_vl2_pallas_supported() rejects the config (e.g. GHOST_CELLS on a mesh before
the sharded fix), which only shows up as ~2x memory. This prints the trace-time
counts at exit; a sharded run on the Pallas path has pallas_local_builds_on_shard > 0.

The patch is applied when cgols itself imports the VL2 integrator, not here:
importing astronomix initialises the jax backend, and a process that already
holds CUDA contexts on every card makes cgols' autocvd wait for free GPUs forever.

    python cgols_vl2_probe.py cgols.py
"""

# general
import atexit
import importlib.abc
import importlib.util
import runpy
import sys

TARGET = "astronomix._finite_volume._state_evolution._van_leer_integrator"
counts = {"pallas_local_builds_on_shard": 0, "pallas_local_builds_unsharded": 0, "pallas_stage_calls": 0}


def patch(vi):
    """Wrap the VL2 Pallas stage entry points with counters."""
    import astronomix._finite_volume._state_evolution._van_leer_pallas as vlp

    _local, _stage = vlp._vl2_stage_pallas_local, vlp._vl2_stage_pallas

    def local(*args, **kwargs):
        layout = kwargs.get("shard_layout")
        on_shard = layout and any(entry is not None for entry in layout)
        counts["pallas_local_builds_on_shard" if on_shard else "pallas_local_builds_unsharded"] += 1
        return _local(*args, **kwargs)

    def stage(*args, **kwargs):
        counts["pallas_stage_calls"] += 1
        return _stage(*args, **kwargs)

    vlp._vl2_stage_pallas_local = local
    vi._vl2_stage_pallas = stage


class PatchOnImport(importlib.abc.MetaPathFinder):
    """Patch the VL2 integrator right after its first import."""

    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


sys.meta_path.insert(0, PatchOnImport())
atexit.register(lambda: print("[probe] VL2 stage path (trace-time counts):", counts, flush=True))
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
