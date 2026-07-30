#!/bin/bash
# Validate the Orbax TO_DISK checkpointing at production scale: 1024x1024x2048
# on 4 GPUs, production memory config (cuda_async allocator, 0.98 fraction, as
# in run_horeka_1024.sh). Leg 1 runs a short fresh window (4 checkpoints, keep 2);
# leg 2 resumes from checkpoint step 3 over the identical remaining segment;
# the resumed final state must be bit-identical to leg 1's. Peak GPU memory is
# tracked in cgols_logs/ckpt1024_peakmem.log. Cleans up its ~200 GB of test
# artifacts on PASS, keeps them for forensics on FAIL.
#
# Launched via: python gpu_reserve.py -n 4 --wait-file data/initial/.ic1024_done \
#                   -- bash ckpt1024_test.sh
set -uo pipefail
cd "$(dirname "$0")"

export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

TAG=_ckpt1024
COMMON=(CGOLS_DIM=1024 "CGOLS_SHARD_SPLIT=(1, 4, 1, 1)" CGOLS_END_FRACTION=0.01
        CGOLS_CHECKPOINT_KEEP=2 "CGOLS_RUN_TAG=$TAG")

# Safety net: the reserver already gates on the IC-build marker, but never
# start without the ICs (a bare invocation would otherwise die confusingly).
while [ ! -f data/initial/.ic1024_done ]; do
    echo "[ckpt1024] waiting for the 1024 IC build to finish..."
    sleep 60
done

mkdir -p cgols_logs
python - <<'EOF' &
import subprocess, time
peaks = {}
while True:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"]).decode()
    for line in out.strip().splitlines():
        i, m = (int(x) for x in line.split(","))
        peaks[i] = max(peaks.get(i, 0), m)
    with open("cgols_logs/ckpt1024_peakmem.log", "w") as f:
        for i in sorted(peaks):
            f.write(f"gpu{i} peak {peaks[i]} MiB\n")
    time.sleep(5)
EOF
POLLER=$!
trap 'kill $POLLER 2>/dev/null' EXIT

echo "[ckpt1024] leg 1: fresh run, 4 checkpoints"
env "${COMMON[@]}" CGOLS_NUM_SNAPSHOTS=4 python cgols.py \
    || { echo "[ckpt1024] LEG 1 FAILED"; exit 1; }
cp "data/final/cgols_final_state${TAG}.npy" \
   "data/final/cgols_final_state${TAG}_leg1.npy"

echo "[ckpt1024] leg 2: resume from checkpoint step 3"
env "${COMMON[@]}" CGOLS_NUM_SNAPSHOTS=1 CGOLS_RESTART_FROM=latest \
    CGOLS_RESTART_STEP=3 python cgols.py \
    || { echo "[ckpt1024] LEG 2 FAILED"; exit 1; }

python - <<'EOF'
import sys
import numpy as np
a = np.load("data/final/cgols_final_state_ckpt1024_leg1.npy", mmap_mode="r")
b = np.load("data/final/cgols_final_state_ckpt1024.npy", mmap_mode="r")
ok = a.shape == b.shape and np.array_equal(a, b)
print(f"[ckpt1024] resumed final == fresh final bitwise: {ok}")
sys.exit(0 if ok else 1)
EOF
RC=$?

echo "[ckpt1024] peak GPU memory during the test:"
cat cgols_logs/ckpt1024_peakmem.log

if [ "$RC" -eq 0 ]; then
    echo "[ckpt1024] PASS - removing test artifacts"
    rm -rf "data/cgols_checkpoints${TAG}" "cgols_snapshots${TAG}" \
           "data/final/cgols_final_state${TAG}.npy" \
           "data/final/cgols_final_state${TAG}_leg1.npy" \
           "cgols_logs/cgols_diag${TAG}.log"
else
    echo "[ckpt1024] FAIL - artifacts kept for forensics"
fi
exit "$RC"
