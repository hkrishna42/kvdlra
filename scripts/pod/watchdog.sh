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
# Credit floor -> destroy everything; so does the BUDGET_ITERS expiry (default: the pod's
# gpu_budget_h plus boot.sh's 2 h grace in 150 s polls, floor 600 = 25 h; BUDGET_ITERS=
# overrides): a watchdog that stops polling must not leave a pod billing (D-011 addendum
# 2), and one that expires before the pod's own bar destroys a healthy run. Run detached,
# and under caffeinate -s -i so the Mac cannot sleep past a finished pod (D-011 addendum
# 8: ~$12 of idle billing):
#   caffeinate -s -i scripts/pod/watchdog.sh <pod>
# boot.sh's own budget markers (RUN_TIMEOUT, SELF_DESTRUCT_FAILED) are kept in ROWS so
# they reach the harvested log; the pod also self-destructs GRACE_S after its final marker.
POD="${1:?usage: watchdog.sh <pod>}"
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
# The harvest needs the repo's venv (the cycle pod's harvest died on a bare `python`).
PY=$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )
H="results/$POD"; mkdir -p "$H"; touch "$H/pods.txt" "$H/done.txt"
echo $$ > "$H/watchdog.pid"  # for caffeinate -w and for teardown checks
FLOOR="${FLOOR:-6.0}"
# Expiry in polls: the pod's own bar (gpu_budget_h, the `timeout` boot.sh enforces) plus
# its GRACE_S self-destruct (7200 s), over the 150 s sleep; never under 600 (25 h).
BUDGET_ITERS="${BUDGET_ITERS:-$(cat "configs/pods/$POD.yaml" 2>/dev/null | awk -F': *' '/^gpu_budget_h:/{n=int(($2*3600+7200)/150)+1} END{print (n<600)?600:n}')}"
# The row kinds kept from each log fetch. boot.sh's ENV block rows are in the set because
# `pod.py harvest` rebuilds results/<pod>/env.txt from them -- the file `pod.py run`
# writes stays on the destroyed instance, and a rebuilt one is what `check` reads. The
# `===ENV_` markers alone are not enough: `sort -u` scatters the block's contents, so
# every line the harvest needs has to match on its own. `[stage]` rows are the runner's
# load timings (model, corpora, haystacks): kept so a slow pod's log says where the
# hours went.
ROWS='^\[(niah|vt|persist|latency)[^]]*\] +[^ ].* (acc=|SKIP|bytes=|ms/tok=)|^ +[^ ].* \[T=[0-9]+\] (ppl=|OOM|error|mem alloc)|^\[pplw|^\[diag|^\[stage|^\[trial\]|^\[error\]|^===(ALL_DONE|RUN_FAILED|RUN_TIMEOUT|SELF_DESTRUCT|CLONE_FAILED|CHECKOUT_FAILED|DEPS_FAILED|MODEL_FAILED|POD_|RUN_SHA|ENV_|QUANTO|HQQ|MODEL_)|^run_sha=|^device=|^python=|^torch=|^triton=|^transformers=|NVIDIA'
# boot.sh's pre-run failures. The instance is destroyed on any of them exactly as on
# ALL_DONE -- a pod that could not clone, check out, install, load the model or import
# its quant backend has nothing left to do but bill. `pod.py harvest` reads the same
# markers back and records `status: BOOT_FAILED`.
BOOT='CLONE_FAILED|CHECKOUT_FAILED|DEPS_FAILED|MODEL_FAILED|QUANTO_MISSING|HQQ_MISSING'
destroy_all() {
  while IFS=: read -r lab id mode tag; do [ -n "$id" ] && echo y | vastai destroy instance "$id" >/dev/null 2>&1; done < "$H/pods.txt"
}
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
    sort -u "$H/${lab}.raw" -o "$H/${lab}.raw"  # the fetch re-sends the tail every poll
    if echo "$L" | grep -qaE "===(ALL_DONE|RUN_FAILED|${BOOT})_${mode}"; then
      end="ALL_DONE"; echo "$L" | grep -qaE "===RUN_FAILED_${mode}" && end="RUN_FAILED"
      echo "$L" | grep -qaE "===(${BOOT})_${mode}" && end="BOOT_FAILED"
      echo "$(date +%H:%M) $lab $end -> destroy"; echo y | vastai destroy instance "$id" >/dev/null 2>&1
      echo "$lab" >> "$H/done.txt"
      sort -u "$H/${lab}.raw" > "$H/${lab}.log"
      $PY scripts/pod.py harvest --pod "$POD" --log "$H/${lab}.log" || echo "HARVEST_FAILED $lab"
    fi
  done < "$H/pods.txt"
  cr="$(vastai show user --raw 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("credit",0))' 2>/dev/null)"
  nd=$(wc -l < "$H/done.txt" | tr -d ' '); tot=$(grep -c . "$H/pods.txt" | tr -d ' ')
  echo "$(date +%H:%M) iter=$iter done=$nd/$tot credit=\$$cr"
  if [ -n "$cr" ] && awk "BEGIN{exit !($cr < $FLOOR)}" 2>/dev/null; then
    echo "!!! CREDIT FLOOR <\$$FLOOR -- destroy all"
    destroy_all
    echo "FLOOR_STOP $(date)" > "$H/status.txt"; exit 0
  fi
  if [ "$tot" -gt 0 ] && [ "$nd" -ge "$tot" ] && [ -z "${KEEP_ALIVE:-}" ]; then
    echo "ALL_DONE $(date)" > "$H/status.txt"; echo "=== ALL INSTANCES DONE + HARVESTED ==="; exit 0
  fi
  sleep 150
done
echo "!!! BUDGET_ITERS=$BUDGET_ITERS expired -- destroy all"
destroy_all
echo "BUDGET_EXPIRED $(date)" > "$H/status.txt"
