#!/bin/bash
# One-time setup for the cgols campaign on HoreKa 2 (x86 side: Ruby gpu-h200,
# Teal gpu-h100). Run it on the x86 LOGIN node (hk2-x86.scc.kit.edu), not in a
# job - it needs internet for pip. Safe to re-run: every step checks first and
# only does what is missing, then prints a status report.
#
#   bash pytests/self_gravity/cgols/horeka2_setup.sh
#
# What it does:
#   1. micromamba in ~/.local/bin (if missing) + its bash hook in ~/.bashrc
#   2. env `astro`: python 3.12, jax[cuda12]==0.10.1, orbax-checkpoint==0.12.1,
#      ptxas 12.9 (nvidia-cuda-nvcc-cu12), astronomix editable from this repo
#      - the exact stack the 1024 run and the local verification used
#   3. workspace `cgols` (60 days) linked as pytests/self_gravity/cgols/data
#   4. report: versions, ptxas, workspace path/expiry/free space, accounts,
#      GPU partition limits
#
# Override with CGOLS_ENV=<name> / CGOLS_WS=<name> / CGOLS_WS_DAYS=<n>.
# HoreKa 2 $HOME is 50 GB / 2M inodes; the env is ~8 GB / ~60k files.

# No `set -u`: the micromamba hook and HoreKa's /etc/bashrc read unset vars.

ENV_NAME="${CGOLS_ENV:-astro}"
WS_NAME="${CGOLS_WS:-cgols}"
WS_DAYS="${CGOLS_WS_DAYS:-60}"
CGOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$CGOLS_DIR/../../.." && pwd)"

say() { printf '\n=== %s\n' "$*"; }

if [ "$(uname -m)" != "x86_64" ]; then
    echo "ERROR: this is $(uname -m). Run on hk2-x86.scc.kit.edu - the Jade (gpu-b200)" >&2
    echo "       ARM side needs a separate aarch64 env, which this script does not build." >&2
    exit 1
fi

# --- 1. micromamba ----------------------------------------------------------
MAMBA_EXE="${MAMBA_EXE:-$HOME/.local/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
if [ ! -x "$MAMBA_EXE" ]; then
    say "installing micromamba -> $MAMBA_EXE"
    mkdir -p "$(dirname "$MAMBA_EXE")"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C "$(dirname "$(dirname "$MAMBA_EXE")")" bin/micromamba || {
        echo "ERROR: micromamba download failed" >&2; exit 1; }
fi
if ! grep -q "micromamba shell init" ~/.bashrc 2>/dev/null \
        && ! grep -q "MAMBA_EXE" ~/.bashrc 2>/dev/null; then
    say "adding the micromamba hook to ~/.bashrc"
    "$MAMBA_EXE" shell init --shell bash --root-prefix "$MAMBA_ROOT_PREFIX"
fi
eval "$("$MAMBA_EXE" shell hook --shell bash)"

# --- 2. env -----------------------------------------------------------------
if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    say "creating env $ENV_NAME (python 3.12)"
    micromamba create -y -n "$ENV_NAME" -c conda-forge python=3.12 || exit 1
fi
micromamba activate "$ENV_NAME"

# `python -m pip`, never bare pip: outside the env pip falls back to the system
# python 3.9 "user installation", where jax caps at 0.4.30.
if ! python -c 'import jax, orbax.checkpoint as o, sys
sys.exit(not (jax.__version__ == "0.10.1" and o.__version__ == "0.12.1"))' 2>/dev/null; then
    say "installing jax 0.10.1 + orbax 0.12.1 (+ ptxas 12.9)"
    # orbax 0.12.1 is REQUIRED with jax 0.10 (older orbax cannot restore under
    # it - a failure that would surface at the first restart leg, hours in).
    # nvidia-cuda-nvcc 12.9: older ptxas cannot assemble the Hopper TMA
    # instructions the Pallas kernels emit, and XLA warns that <=12.6.2
    # miscompiles clamping edge cases.
    python -m pip install -q \
        "jax[cuda12]==0.10.1" \
        "orbax-checkpoint==0.12.1" \
        "nvidia-cuda-nvcc-cu12>=12.9,<13" \
        equinox beartype jaxtyping scipy astropy autocvd matplotlib || exit 1
fi
if ! python -c 'import astronomix, os, sys
sys.exit(not os.path.realpath(astronomix.__file__).startswith(os.path.realpath(sys.argv[1])))' "$REPO" 2>/dev/null; then
    say "installing astronomix editable from $REPO"
    # --no-deps: pyproject still pins orbax <0.11.11 for upstream's jax 0.6.2,
    # and resolving it would drag orbax back below what jax 0.10 can restore.
    # The real dependencies were installed explicitly above.
    python -m pip install -q --no-deps -e "$REPO" || exit 1
fi

# --- 3. workspace -----------------------------------------------------------
WS_PATH="$(ws_find "$WS_NAME" 2>/dev/null)"
if [ -z "$WS_PATH" ]; then
    say "allocating workspace $WS_NAME for $WS_DAYS days"
    ws_allocate "$WS_NAME" "$WS_DAYS" >/dev/null || exit 1
    WS_PATH="$(ws_find "$WS_NAME")"
fi
cd "$CGOLS_DIR" || exit 1
if [ -L data ]; then
    if [ "$(readlink -f data)" != "$(readlink -f "$WS_PATH")" ]; then
        echo "WARNING: ./data -> $(readlink data), NOT the workspace $WS_PATH (left as is)" >&2
    fi
elif [ -e data ]; then
    echo "ERROR: ./data exists and is not a symlink - move it away, then re-run" >&2
    exit 1
else
    ln -s "$WS_PATH" data
fi
mkdir -p cgols_logs

# --- 4. report --------------------------------------------------------------
say "versions (env $ENV_NAME)"
JAX_PLATFORMS=cpu python - <<'PY'
import sys, importlib
print(f"python      {sys.version.split()[0]}  {sys.executable}")
for name in ("jax", "jaxlib", "orbax.checkpoint", "equinox", "numpy"):
    try:
        print(f"{name:11s} {importlib.import_module(name).__version__}")
    except Exception as exc:
        print(f"{name:11s} MISSING ({exc})")
import astronomix
print(f"astronomix  {astronomix.__file__}")
PY
PTXAS="$(python -c 'import site,glob;print(next(iter(glob.glob(site.getsitepackages()[0]+"/nvidia/*/bin/ptxas")),""))')"
echo "ptxas       ${PTXAS:-NOT FOUND} $([ -n "$PTXAS" ] && "$PTXAS" --version | tail -1)"
echo "repo        $REPO @ $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"

say "workspace"
echo "data ->     $(readlink -f data)"
ws_list 2>/dev/null | grep -A3 "$WS_NAME" | head -6
df -h "$WS_PATH" | tail -1
# Per-user work quota is 250 TiB / 50M inodes (soft); $HOME is 50 GiB / 2M.
/usr/lpp/mmfs/bin/mmlsquota -u "$(whoami)" --block-size G -C hk2n.scc.kit.edu hfs2-work 2>/dev/null
/usr/lpp/mmfs/bin/mmlsquota -u "$(whoami)" --block-size G -C hk2n.scc.kit.edu hfs2-home:home 2>/dev/null
echo "(2048 needs ~1.7 TB steady state at CGOLS_CHECKPOINT_KEEP=4, 344 GB per extra full state)"

say "accounts (the runners use --account=hk-project-pai00101)"
sacctmgr -n -P show associations user="$USER" format=Account 2>/dev/null | sort -u

say "GPU partitions"
for p in gpu-h200 gpu-h100 dev-gpu-h100 gpu-b200; do
    scontrol show partition "$p" 2>/dev/null \
        | grep -oE "PartitionName=[^ ]+|State=[^ ]+|MaxNodes=[^ ]+|MaxTime=[^ ]+|TotalNodes=[^ ]+|OverSubscribe=[^ ]+" \
        | tr '\n' ' '
    echo
done
echo
echo "Setup done. Next: sbatch --test-only the job shapes, then the dev ladder."
