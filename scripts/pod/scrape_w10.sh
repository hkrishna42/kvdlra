#!/bin/bash
# Scrape the base64-folded Week-10 result blocks from a finished pod's `vastai logs`
# (SSH is unusable, so results are emitted to the onstart log; see scripts/pod/w10_gpu.sh).
# The pod log persists until the instance is destroyed, so this is safely re-runnable.
#
# Usage:  bash scripts/pod/scrape_w10.sh <contract_id> <tag>      # tag e.g. 1b / 8b
# Writes: results/w10-gpu-<tag>-{frontier,ruler,longbench}.json
set -u
C="$1"; TAG="$2"
L="$(timeout 90 uvx vastai logs "$C" --tail 20000 2>/dev/null)"
mkdir -p results
for pair in PPL:frontier RULER:ruler LB:longbench; do
  M="${pair%%:*}"; name="${pair##*:}"
  out="results/w10-gpu-${TAG}-${name}.json"
  echo "$L" \
    | awk "/===${M}_RESULT_BEGIN===/{f=1;next} /===${M}_RESULT_END===/{f=0} f" \
    | tr -d ' \r\n' | base64 -d > "$out" 2>/dev/null
  if [ -s "$out" ] && python3 -c "import json,sys; json.load(open('$out'))" 2>/dev/null; then
    echo "OK   ${TAG}/${name}  ($(wc -c < "$out") bytes)"
  else
    echo "MISS ${TAG}/${name}  (block absent or truncated in the fetched tail)"
    rm -f "$out"
  fi
done
