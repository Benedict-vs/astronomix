"""Regenerate every methods-paper figure from the ``examples/`` generators.

The faithful figure generators live under ``examples/scripts/`` (the same
scripts a user runs to reproduce a figure by hand). This module drives them:
each entry in :data:`PAPER_FIGURES` names one paper figure, the ordered
generator invocation(s) that build it, and the output file(s) it must produce.

Run the whole set (slow, needs a GPU)::

    pytest pytests/test_reproduce_paper.py --reproduce-paper

Without ``--reproduce-paper`` every case is skipped, so a normal ``pytest`` run
stays fast. Each generator is executed as a subprocess with the repository root
on ``PYTHONPATH`` (so it imports this worktree's ``astronomix``) and its own
directory as the working directory (so ``from _common import ...`` and the
relative ``data/`` / ``figures/`` directories resolve). ``autocvd`` inside each
generator selects a free GPU.

Hardware-gated cases are intentionally excluded from the automated set because
they need resources beyond a single node (run them by hand — see each module's
docstring):

* ``differentiability/kh_recon.py`` — heavy reconstruction optimisation campaign.
* ``scaling/strong_scaling_speedup.py`` — the combined hydro+MHD speedup figure
  is measured across separate 4-/8-GPU H100/H200 machines, then assembled with
  ``--plot``.
* ``scaling/scaling_campaign.py`` — single-GPU / block-shape / strong-scaling
  measurement driver (writes NPZ+JSON caches, not a single figure).
* ``scaling/weak_scaling_hydro.py`` — multi-node weak scaling launched under
  Slurm (up to 2048^3 on 16 GPUs / 4 nodes).

The single-node ``checkpoint_scaling`` and 2-GPU ``extended_strong_scaling``
figures ARE included above.
"""

# general
import os
import subprocess
import sys
from pathlib import Path

# typing
from typing import NamedTuple, Sequence, Tuple

# testing
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples" / "scripts"


class Step(NamedTuple):
    """One generator invocation: a script (relative to ``examples/scripts``)
    and its command-line arguments."""

    script: str
    args: Tuple[str, ...] = ()


class PaperFigure(NamedTuple):
    """One paper figure: the ordered generator steps that build it and the
    output files (relative to each step's directory) it must produce."""

    name: str
    steps: Sequence[Step]
    outputs: Sequence[str]


# The turbulence figures need the ISM and ICM regimes at both resolutions before
# the figure assemblers (make_fig14 / make_fig15) run; every other figure is a
# single generator call (some with a ``--double`` / ``--rerun`` flag).
_TURB = "forward/mhd/turbulence"
_TURB_STEPS = [
    Step(f"{_TURB}/paper_turbulence.py",
         ("--mturb", "10", "--beta", "0.1", "--eos", "iso", "--tcross", "5",
          "--N", str(N), "--tag", f"ISM_N{N}"))
    for N in (128, 256)
] + [
    Step(f"{_TURB}/paper_turbulence.py",
         ("--mturb", "0.5", "--beta", "1e6", "--eos", "iso", "--tcross", "30",
          "--N", str(N), "--tag", f"ICM_N{N}"))
    for N in (128, 256)
] + [
    Step(f"{_TURB}/make_fig14.py"),
    Step(f"{_TURB}/make_fig15.py"),
]


PAPER_FIGURES = [
    # --- differentiability ------------------------------------------------
    PaperFigure(
        "eigen_initialization",
        [Step("differentiability/eigen_initialization.py")],
        ["figures/eigenvalue_spectrum.png",       # paper: eigenvalue_spectrum_final
         "figures/eigenmode_final_states.png",
         "figures/eigenmode_transient_test.png"],
    ),
    PaperFigure(
        "sensitivity",
        [Step("differentiability/sensitivity.py")],
        ["figures/gradient_3d_gaussian.svg", "figures/gradient_convergence_test.svg"],
    ),
    PaperFigure(
        "shock_tube_sensitivity",
        [Step("differentiability/shock_tube_sensitivity.py")],
        ["figures/shock_tube_sensitivity.svg",
         "figures/shock_tube_sensitivity_step_sweep.svg"],
    ),
    # The paper ran this at --fine-N 128, whose naive full-resolution
    # value-and-grad needs ~181 GiB; --fine-N 64 reproduces the same 3-method
    # comparison within ~10 GiB (A100-40GB). Three steps: optimise, render the
    # panel snapshots, then assemble the figure.
    PaperFigure(
        "field_level_inference",
        [Step("differentiability/field_level_inference/run_inference.py",
              ("--fine-N", "64", "--coarse-N", "32")),
         Step("differentiability/field_level_inference/make_panel_snaps.py",
              ("--state", "data/best_state_full128.npy", "--resolution", "64",
               "--out", "panel_snaps.npz")),
         Step("differentiability/field_level_inference/make_panels_fig.py",
              ("--snaps", "panel_snaps.npz", "--out", "field_level_inference.png"))],
        ["field_level_inference.png"],
    ),
    # --- forward hydro ----------------------------------------------------
    PaperFigure(
        "double_blast",
        [Step("forward/hydro/double_blast.py", ("--rerun",))],
        ["figures/double_blast.pdf"],
    ),
    PaperFigure(
        "sedov_blast",
        [Step("forward/hydro/sedov_blast.py", ("--rerun",))],
        ["figures/sedov_blast_256.png"],
    ),
    PaperFigure(
        "shock_tube1D",
        [Step("forward/hydro/shock_tube1D.py")],
        ["figures/shock_tube1D_test.svg"],
    ),
    PaperFigure(
        "sound_wave_convergence",
        [Step("forward/hydro/sound_wave_convergence.py")],
        ["figures/sound_wave3D_convergence.svg"],
    ),
    # --- forward mhd ------------------------------------------------------
    PaperFigure(
        "mhd_blast",
        [Step("forward/mhd/mhd_blast.py", ("--rerun",))],
        ["figures/mhd_blast_test1_256cells.png",
         "figures/mhd_blast_oscillations_comparison.png"],
    ),
    PaperFigure(
        "mhd_jet",
        [Step("forward/mhd/mhd_jet.py", ("--rerun",))],
        ["figures/mhd_jet_fd_256.png"],
    ),
    PaperFigure(
        "orszag_tang",
        [Step("forward/mhd/orszag_tang.py", ("--rerun",))],
        ["figures/orszag_tang.svg"],
    ),
    PaperFigure(
        "alfven_convergence",
        [Step("forward/mhd/alfven_convergence.py")],
        ["figures/alfven_wave3D_dp_convergence.svg"],
    ),
    PaperFigure(
        "turbulence",
        _TURB_STEPS,
        ["figures/turbulence_slices.pdf", "figures/turbulence_spectra.svg"],
    ),
    # --- forward self-gravity --------------------------------------------
    PaperFigure(
        "jeans_convergence",
        [Step("forward/self_gravity/jeans_convergence.py")],
        ["figures/jeans_waves_error_convergence.svg"],
    ),
    PaperFigure(
        "slab_convergence",
        [Step("forward/self_gravity/slab_convergence.py")],
        ["figures/slab_error_convergence.svg"],
    ),
    PaperFigure(
        "energy_conservation_comparison",
        [Step("forward/self_gravity/energy_conservation_comparison.py",
              ("--double", "--rerun"))],
        ["figures/energy_conservation_comparison_fp64.svg"],
    ),
    PaperFigure(
        "evrard_energy_convergence",
        [Step("forward/self_gravity/evrard_energy_convergence.py",
              ("--double", "--rerun"))],
        ["figures/evrard_energy_convergence_fp64.svg"],
    ),
    PaperFigure(
        "evrard_timestep_convergence",
        [Step("forward/self_gravity/evrard_timestep_convergence.py",
              ("--double", "--rerun"))],
        ["figures/evrard_timestep_convergence_fp64.svg"],
    ),
    PaperFigure(
        "radial_profiles_comparison",
        [Step("forward/self_gravity/radial_profiles_comparison.py",
              ("--double", "--rerun"))],
        ["figures/collapse_radial_profiles_comparison_fp64.svg"],
    ),
    # --- scaling ----------------------------------------------------------
    PaperFigure(
        "checkpoint_scaling",
        [Step("scaling/checkpoint_scaling.py")],
        ["figures/checkpoint_scaling.svg"],
    ),
    # Single-node 2-GPU strong-scaling sweep (FV/FD-JAX/FD-Pallas up to N=256).
    PaperFigure(
        "extended_strong_scaling",
        [Step("scaling/extended_strong_scaling.py")],
        ["figures/sound_wave3D_extended_strong_scaling.svg"],
    ),
]


@pytest.mark.parametrize(
    "figure", PAPER_FIGURES, ids=lambda figure: figure.name
)
def test_reproduce_paper_figure(figure, request):
    """Run one paper figure's generator step(s) and assert its outputs appear.

    Skipped unless ``--reproduce-paper`` is passed, so a normal ``pytest`` run
    stays fast. Each step is executed with the repository root on PYTHONPATH and
    its own directory as cwd; afterwards every declared output file must exist.

    Args:
        figure: The :class:`PaperFigure` describing one paper figure.
        request: The pytest request, used to read the ``--reproduce-paper`` flag.
    """
    if not request.config.getoption("--reproduce-paper"):
        pytest.skip("pass --reproduce-paper to regenerate the paper figures")

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    for step in figure.steps:
        script = EXAMPLES / step.script
        assert script.exists(), f"generator missing: {script}"
        subprocess.run(
            [sys.executable, str(script), *step.args],
            cwd=str(script.parent),
            check=True,
            env=env,
        )

    # Outputs are resolved relative to the last step's directory.
    out_dir = (EXAMPLES / figure.steps[-1].script).parent
    for rel in figure.outputs:
        produced = out_dir / rel
        assert produced.exists(), f"{figure.name}: expected output missing: {produced}"
