#!/bin/bash
# One CGOLS run with non-equilibrium chemistry (CGOLS_CHEMISTRY=1) on one local GPU.
#
#   run_chemistry_local.sh <gpu> <dim> <end_fraction> <num_snapshots> <run_tag> [VAR=VALUE ...]
#
#   run_chemistry_local.sh 7 256 1 60 _chem256                            # 75 Myr at 256 x 256 x 512
#   run_chemistry_local.sh 7 256 1 60 _chem256 CGOLS_RESTART_FROM=latest  # resume from its newest checkpoint
#   run_chemistry_local.sh 7 64 0.02 3 _chemtest64 CGOLS_CHECKPOINT_EVERY=0
#   # one radiation field everywhere, with the network file as KROME ships it (the first run, _chem256):
#   run_chemistry_local.sh 7 256 1 60 _chem256 CGOLS_CHEM_FUV=0 CGOLS_CHEM_NETWORK=<...>/react_galaxy_ism_plus_cr
#
# The GPU must be completely free (no process of anyone, lab policy); with ALLOW_OWN=1 it may hold
# other runs of the same user. The run needs what is not part of the repository:
#   CHEM_PYTHON            a Python with jax, diffrax, gradsolve and warp-lang (requirements-chemistry.txt)
#   CGOLS_CHEM_NETWORK     the network in the KROME format (react_galaxy_ism_plus, cosmic-ray block on, and for
#                          a radiation field per cell the H2 dissociation by the flux user_myfluxLW: see
#                          cgols_chemistry.build_chemistry and the README next to the network files)
#   CGOLS_CHEM_KROME_DATA  the data folder of a KROME installation
# Every 4th full state is kept (CGOLS_CHECKPOINT_ARCHIVE_EVERY=4: every 5 Myr of a 60-frame run, 16 x 5.2 GB
# at 256), on top of the rolling window of the newest four; pass CGOLS_CHECKPOINT_ARCHIVE_EVERY=1 for all.
# Start it detached for a long run:  setsid nohup run_chemistry_local.sh ... > run.out 2>&1 < /dev/null &
set -u
if [ $# -lt 5 ]; then sed -n '2,15p' "$0"; exit 2; fi
GPU=$1; DIM=$2; END_FRACTION=$3; NUM_SNAPSHOTS=$4; RUN_TAG=$5; shift 5

HERE=$(cd "$(dirname "$0")" && pwd)
REPOSITORY=$(cd "$HERE/../../.." && pwd)
CHEM_PYTHON=${CHEM_PYTHON:-/export/scratch/bschuber/envs/chemenv/bin/python}
export CGOLS_CHEM_NETWORK=${CGOLS_CHEM_NETWORK:-/export/scratch/bschuber/krome_networks/react_galaxy_ism_plus_cr_fuv}
export CGOLS_CHEM_KROME_DATA=${CGOLS_CHEM_KROME_DATA:-/export/scratch/lbranca/krome/data}

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
processes=0; foreign=0
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU"); do
    processes=$((processes + 1))
    [ "$(ps -o user= -p "$pid" 2>/dev/null)" = "$(id -un)" ] || foreign=$((foreign + 1))
done
if [ "${ALLOW_OWN:-0}" = "1" ]; then
    if [ "$foreign" -gt 0 ]; then echo "GPU $GPU holds $foreign processes of other users: not starting"; exit 3; fi
elif [ "$used" -gt 1000 ] || [ "$processes" -gt 0 ]; then
    echo "GPU $GPU is not free ($used MiB, $processes processes): not starting"; exit 3
fi

cd "$HERE" || exit 1
# autocvd_shim: the GPU is chosen here, so the script's own autocvd call must not wait for one.
# No preallocation: the Warp kernel of the chemistry allocates outside the XLA pool.
exec env CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="$HERE/gpuq/autocvd_shim:$REPOSITORY:$REPOSITORY/external/carbox" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    CGOLS_X64=1 CGOLS_CHEMISTRY=1 \
    CGOLS_DIM="$DIM" CGOLS_END_FRACTION="$END_FRACTION" CGOLS_NUM_SNAPSHOTS="$NUM_SNAPSHOTS" CGOLS_RUN_TAG="$RUN_TAG" \
    CGOLS_CHECKPOINT_ARCHIVE_EVERY=4 \
    "$@" "$CHEM_PYTHON" -u cgols.py
