#!/bin/bash
# Print the indices of FULLY free GPUs: no compute process (of anyone) and
# < 1000 MiB used. --wait blocks until one stays free on two checks 2 min apart
# and prints that index (polls every 5 min; nvidia-smi only, local node).
list(){ local busy; busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | while IFS=', ' read -r i u m; do
    grep -q "$u" <<< "$busy" && continue; [ "$m" -lt 1000 ] && echo "$i"; done; }
[ "${1:-}" = "--wait" ] || { list; exit 0; }
while true; do
  a=$(list | head -1)
  if [ -n "$a" ]; then sleep 120; list | grep -qx "$a" && { echo "$a"; exit 0; }; fi
  sleep 300
done
