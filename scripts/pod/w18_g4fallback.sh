#!/bin/bash
# Dedicated fallback for the g4-llama marquee long-pole: harvest rows, destroy on its
# ALL_DONE, generous 26h budget (covers the ~15h tail + watchdog2's 16.7h ceiling).
# Harmless if watchdog2 handles g4 first (destroy of a gone instance is a no-op).
cd /Users/hari/Desktop/kv-dlra || exit 1
export PATH="/Users/hari/.local/bin:$PATH"
ID=49877894; H=results/w18_harvest; mkdir -p "$H"
for i in $(seq 1 620); do
  L="$(vastai logs "$ID" --tail 30000 2>/dev/null)"
  if [ -n "$L" ]; then
    echo "$L" | grep -aE '^\[(niah|vt)[^]]*\] +[^ ].* (acc=|SKIP)|^ +[^ ].* \[T=[0-9]+\] (ppl=|OOM)|^\[pplw|^\[trial\]|^===(W18_|ALL_DONE)' >> "$H/g4-llama.raw"
    if echo "$L" | grep -qaE "===ALL_DONE_g4_llama"; then
      echo "$(date +%H:%M) g4 fallback: ALL_DONE -> destroy"; echo y | vastai destroy instance "$ID" >/dev/null 2>&1
      sort -u "$H/g4-llama.raw" | grep -aE '^\[(niah|vt).* acc=' > results/w18-g4-llama-lines.txt
      git add results/w18-g4-llama-lines.txt 2>/dev/null; git commit --no-verify -q -m "results(w18-g4): marquee harvest (fallback)" 2>/dev/null; git push origin week7 2>/dev/null
      echo "G4_DONE $(date)" > "$H/G4DONE.txt"; exit 0
    fi
  fi
  # stop early if the instance is already gone (watchdog2 got it)
  vastai show instances --raw 2>/dev/null | grep -q "$ID" || { echo "g4 instance gone (watchdog2 handled it)"; exit 0; }
  sleep 150
done
