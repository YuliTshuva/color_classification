#!/bin/bash
# Build every dataset and scaling point one after another, so the recorded
# build times are not skewed by jobs competing for the CPU.
PY=${PY:-/c/Users/yulit/miniconda3/envs/color-classification/python.exe}
cd "$(dirname "$0")/.."
mkdir -p logs
run() {
  local tag="$1"; shift
  echo "$(date '+%H:%M:%S') START $tag"
  if "$PY" scripts/build_graph.py "$@" > "logs/build_$tag.log" 2>&1; then
    echo "$(date '+%H:%M:%S') DONE  $tag: $(grep '^Wrote' "logs/build_$tag.log" | sed 's/.*: N=/N=/')"
  else
    echo "$(date '+%H:%M:%S') FAIL  $tag: $(tail -1 "logs/build_$tag.log")"
  fi
}
run synthea_1k --dataset synthea_1k
for f in 0.10 0.25 0.50 1.0; do run jodie_wikipedia_$f --dataset jodie_wikipedia --fraction $f; done
run synthea_5k --dataset synthea_5k
for f in 0.10 0.25 0.50 1.0; do run jodie_reddit_$f --dataset jodie_reddit --fraction $f; done
run synthea_20k --dataset synthea_20k
echo "$(date '+%H:%M:%S') ALL DONE"
