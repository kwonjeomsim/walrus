#!/bin/bash
# Usage: nohup setsid ./run_apps15.sh [repeats, default 10] > run.out 2>&1 &
set -eu
K=$(cd "$(dirname "$0")" && pwd)
W=$HOME/walrus-v6/out/walrus; REPS=${1:-10}
mkdir -p "$K/results"
taskset -c 2 python3 "$HOME/test_kit/tools/run_modes.py" "$K/benches_apps15.tsv" --out "$K/results/apps15_arm.json" --walrus "$W" \
  --modes interp,alljit,tierup,lazy,O1,O2,O3,oracle \
  --repeats "$REPS" --oracle-dir "$K/oracle" \
  --remap /home/soonbkwon/projects/walrus/results/apps15_kit="$K" --timeout-scale 4 2>&1 | tee -a "$K/results/apps15_arm.log"
