#!/bin/bash
# Detached unattended watchdog for one pod name (restart-safe, file-driven).
#   usage: scripts/pod/watchdog.sh <pod>            # e.g. scripts/pod/watchdog.sh w18_g1
#   pods:  results/<pod>/pods.txt   one "label:id:mode:tag" per line, appended by
#          `scripts/pod.py launch` -- APPEND to add an instance mid-run (re-read every
#          iteration; no restart needed). The label is "<pod>-<instance id>", unique per
#          launch: done.txt below is keyed by it, so a label shared by two launches
#          would retire the second instance before it started.
#   done:  results/<pod>/done.txt   labels already harvested+destroyed (persisted, so a
#          restart never hangs on a destroyed pod). The run-status marker is status.txt
#          -- NOT DONE.txt, which is the same file as done.txt on macOS's APFS.
# Per instance per iteration: append the short rows to results/<pod>/<label>.raw (the
# vastai log buffer scrolls under per-[trial] emission, so the accumulated rows are the
# record), and on its own ===ALL_DONE_<mode> -- or ===RUN_FAILED_<mode>, or any of
# boot.sh's pre-run failures, which are the same signal for billing and are recorded as
# `status: RUN_FAILED` / `BOOT_FAILED` by the harvest: destroy it, dedupe the rows into
# <label>.log and turn them into records with `scripts/pod.py harvest` (which takes the
# pod name or the label, and refuses to shrink an existing harvest). It commits nothing
# and pushes nothing -- the orchestrator commits the harvest.
# Credit floor -> destroy everything. Run detached (python double-fork) + caffeinate.
#   BUDGET_ITERS (default 600 x 150s = 25h) -- give a long pole a bigger budget.
POD="${1:?usage: watchdog.sh <pod>}"
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
H="results/$POD"; mkdir -p "$H"; touch "$H/pods.txt" "$H/done.txt"
echo $$ > "$H/watchdog.pid"  # for caffeinate -w and for teardown checks
FLOOR="${FLOOR:-6.0}"; BUDGET_ITERS="${BUDGET_ITERS:-600}"
ROWS='^\[(niah|vt|persist|latency)[^]]*\] +[^ ].* (acc=|SKIP|bytes=|ms/tok=)|^ +[^ ].* \[T=[0-9]+\] (ppl=|OOM|error|mem alloc)|^\[pplw|^\[diag|^\[trial\]|^\[error\]|^===(ALL_DONE|RUN_FAILED|CLONE_FAILED|CHECKOUT_FAILED|DEPS_FAILED|MODEL_FAILED|POD_|RUN_SHA|ENV_|QUANTO|HQQ|MODEL_)|^run_sha=|^device=|^torch=|^transformers=|NVIDIA'
# boot.sh's pre-run failures. The instance is destroyed on any of them exactly as on
# ALL_DONE -- a pod that could not clone, check out, install, load the model or import
# its quant backend has nothing left to do but bill. `pod.py harvest` reads the same
# markers back and records `status: BOOT_FAILED`.
BOOT='CLONE_FAILED|CHECKOUT_FAILED|DEPS_FAILED|MODEL_FAILED|QUANTO_MISSING|HQQ_MISSING'
for iter in $(seq 1 "$BUDGET_ITERS"); do
  while IFS=: read -r lab id mode tag; do
    [ -z "$lab" ] && continue
    grep -qx "$lab" "$H/done.txt" && continue
    L="$(vastai logs "$id" --tail 30000 2>/dev/null)"
    # The 30000-line fetch returned EMPTY for three finished Week-20 pods (never
    # harvested, never destroyed, ~$27 idle overnight); a smaller tail worked. Fall
    # back before skipping.
    [ -z "$L" ] && L="$(vastai logs "$id" --tail 5000 2>/dev/null)"; [ -z "$L" ] && continue
    echo "$L" | grep -aE "$ROWS" >> "$H/${lab}.raw"
    if echo "$L" | grep -qaE "===(ALL_DONE|RUN_FAILED|${BOOT})_${mode}"; then
      end="ALL_DONE"; echo "$L" | grep -qaE "===RUN_FAILED_${mode}" && end="RUN_FAILED"
      echo "$L" | grep -qaE "===(${BOOT})_${mode}" && end="BOOT_FAILED"
      echo "$(date +%H:%M) $lab $end -> destroy"; echo y | vastai destroy instance "$id" >/dev/null 2>&1
      echo "$lab" >> "$H/done.txt"
      sort -u "$H/${lab}.raw" > "$H/${lab}.log"
      python scripts/pod.py harvest --pod "$POD" --log "$H/${lab}.log" || echo "HARVEST_FAILED $lab"
    fi
  done < "$H/pods.txt"
  cr="$(vastai show user --raw 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("credit",0))' 2>/dev/null)"
  nd=$(wc -l < "$H/done.txt" | tr -d ' '); tot=$(grep -c . "$H/pods.txt" | tr -d ' ')
  echo "$(date +%H:%M) iter=$iter done=$nd/$tot credit=\$$cr"
  if [ -n "$cr" ] && awk "BEGIN{exit !($cr < $FLOOR)}" 2>/dev/null; then
    echo "!!! CREDIT FLOOR <\$$FLOOR -- destroy all"
    while IFS=: read -r lab id mode tag; do [ -n "$id" ] && echo y | vastai destroy instance "$id" >/dev/null 2>&1; done < "$H/pods.txt"
    echo "FLOOR_STOP $(date)" > "$H/status.txt"; exit 0
  fi
  if [ "$tot" -gt 0 ] && [ "$nd" -ge "$tot" ] && [ -z "${KEEP_ALIVE:-}" ]; then
    echo "ALL_DONE $(date)" > "$H/status.txt"; echo "=== ALL INSTANCES DONE + HARVESTED ==="; exit 0
  fi
  sleep 150
done
echo "BUDGET_EXPIRED $(date)" > "$H/status.txt"
