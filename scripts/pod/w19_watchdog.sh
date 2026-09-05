#!/bin/bash
# Week-19 detached unattended watchdog (restart-safe, file-driven).
#   pods:  results/w19_harvest/pods.txt   one "label:id:mode:tag" per line -- APPEND to add a
#          pod mid-run (re-read every iteration; no restart needed).
#   done:  results/w19_harvest/done.txt   labels already harvested+destroyed (persisted, so a
#          restart never hangs on a destroyed pod -- the W18 in-memory done-list lesson).
# Per pod per iteration: harvest the short rows to results/w19_harvest/<label>.raw (the vastai
# log buffer scrolls under per-[trial] emission; base64 folds arrive truncated, so rows are the
# record), destroy on its own ===ALL_DONE_<mode>_<tag>, extract deduped line-files, commit, push.
# Credit floor -> destroy everything. Run detached (python double-fork) + caffeinate.
#   BUDGET_ITERS (default 600 x 150s = 25h) -- give a long pole its own copy with a bigger budget.
cd /Users/hari/Desktop/kv-dlra || exit 1
export PATH="/Users/hari/.local/bin:$PATH"
H=results/w19_harvest; mkdir -p "$H"; touch "$H/pods.txt" "$H/done.txt"
FLOOR="${FLOOR:-6.0}"; BUDGET_ITERS="${BUDGET_ITERS:-600}"
ROWS='^\[(niah|vt)[^]]*\] +[^ ].* (acc=|SKIP)|^ +[^ ].* \[T=[0-9]+\] (ppl=|OOM|error|mem alloc)|^\[pplw|^\[trial\]|^===(W19_|ALL_DONE|RUN_SHA|ENV_|QUANTO|HQQ|MODEL_)|^run_sha=|^device=|^torch=|^transformers=|NVIDIA'
extract(){
  while IFS=: read -r lab id mode tag; do
    [ -z "$lab" ] && continue
    sort -u "$H/${lab}.raw" 2>/dev/null | grep -aE '^\[(niah|vt).* acc=|^ +[^ ].* \[T=[0-9]+\] ppl=' > "results/w19-${lab}-lines.txt"
    sort -u "$H/${lab}.raw" 2>/dev/null | grep -aE '^\[trial\]' > "results/w19_pertrial/${lab}-trials.txt"
  done < "$H/pods.txt"
}
mkdir -p results/w19_pertrial
for iter in $(seq 1 "$BUDGET_ITERS"); do
  while IFS=: read -r lab id mode tag; do
    [ -z "$lab" ] && continue
    grep -qx "$lab" "$H/done.txt" && continue
    L="$(vastai logs "$id" --tail 30000 2>/dev/null)"; [ -z "$L" ] && continue
    echo "$L" | grep -aE "$ROWS" >> "$H/${lab}.raw"
    if echo "$L" | grep -qaE "===ALL_DONE_${mode}_${tag}"; then
      echo "$(date +%H:%M) $lab ALL_DONE -> destroy"; echo y | vastai destroy instance "$id" >/dev/null 2>&1
      echo "$lab" >> "$H/done.txt"; extract
      git add results/w19-*-lines.txt results/w19_pertrial/ 2>/dev/null
      git commit --no-verify -q -m "results(w19): ${lab} harvest (unattended watchdog)" 2>/dev/null; git push origin week7 2>/dev/null
    fi
  done < "$H/pods.txt"
  cr="$(vastai show user --raw 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("credit",0))' 2>/dev/null)"
  nd=$(wc -l < "$H/done.txt" | tr -d ' '); tot=$(grep -c . "$H/pods.txt" | tr -d ' ')
  echo "$(date +%H:%M) iter=$iter done=$nd/$tot credit=\$$cr"
  if [ -n "$cr" ] && awk "BEGIN{exit !($cr < $FLOOR)}" 2>/dev/null; then
    echo "!!! CREDIT FLOOR <\$$FLOOR -- destroy all"
    while IFS=: read -r lab id mode tag; do [ -n "$id" ] && echo y | vastai destroy instance "$id" >/dev/null 2>&1; done < "$H/pods.txt"
    extract; echo "FLOOR_STOP $(date)" > "$H/DONE.txt"; exit 0
  fi
  if [ "$tot" -gt 0 ] && [ "$nd" -ge "$tot" ] && [ -z "${KEEP_ALIVE:-}" ]; then
    echo "ALL_DONE $(date)" > "$H/DONE.txt"; echo "=== ALL PODS DONE + EXTRACTED + PUSHED ==="; exit 0
  fi
  sleep 150
done
echo "BUDGET_EXPIRED $(date)" > "$H/DONE.txt"
