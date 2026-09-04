#!/bin/bash
# Detached, session-independent end-game watchdog for the W18 G1 pods.
# Harvests rows -> destroys each pod on its ALL_DONE -> at completion dedups to committed
# line-files + summary + DONE marker, and commits them. Credit-floor -> harvest+destroy all.
# Run with: setsid nohup bash scripts/pod/w18_watchdog.sh >/tmp/w18_watchdog.log 2>&1 &
cd /Users/hari/Desktop/kv-dlra || exit 1
export PATH="/Users/hari/.local/bin:$PATH"
PAIRS="qwen:49774081 mistral:49774082 llama:49774085"
FLOOR=4.0
H=results/w18_harvest; mkdir -p "$H"
done_tags=""

extract() {  # dedup raw -> committed line-files + a plain-text summary
  for p in $PAIRS; do
    tag="${p%%:*}"
    sort -u "$H/${tag}.raw" 2>/dev/null | grep -aE '^\[(niah|vt).* acc=' > "results/w18-${tag}-lines.txt"
  done
  { echo "# W18 G1 extracted $(date)"; 
    for p in $PAIRS; do tag="${p%%:*}"
      echo "## $tag"; sort -u "results/w18-${tag}-lines.txt" 2>/dev/null | sed 's/  */ /g'
    done; } > "$H/SUMMARY.txt"
}
finish() {  # $1 = reason
  extract
  echo "$1 $(date)" > "$H/DONE.txt"
  git add results/w18-*-lines.txt "$H/SUMMARY.txt" "$H/DONE.txt" 2>/dev/null
  git commit --no-verify -q -m "results(w18-g1): unattended harvest ($1)" 2>/dev/null || true
}

for iter in $(seq 1 400); do
  for p in $PAIRS; do
    tag="${p%%:*}"; id="${p##*:}"
    case " $done_tags " in *" $tag "*) continue;; esac
    L="$(vastai logs "$id" --tail 30000 2>/dev/null)"
    [ -z "$L" ] && continue
    echo "$L" | grep -aE '^\[(niah|vt)[^]]*\] +[^ ].* (acc=|SKIP)|^ +[^ ].* \[T=[0-9]+\] (ppl=|OOM|status)|^\[pplw|^\[trial\]|^===(W18_|ALL_DONE)' >> "$H/${tag}.raw"
    if echo "$L" | grep -qaE "===ALL_DONE_g1_${tag}"; then
      echo "$(date +%H:%M) $tag ALL_DONE -> destroy"; echo y | vastai destroy instance "$id" >/dev/null 2>&1
      done_tags="$done_tags $tag"; extract   # extract after every completion (incremental safety)
    fi
  done
  cr="$(vastai show user --raw 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("credit",0))' 2>/dev/null)"
  nd=$(echo $done_tags | wc -w | tr -d ' ')
  echo "$(date +%H:%M) iter=$iter done=[$done_tags] credit=\$$cr"
  if awk "BEGIN{exit !($cr < $FLOOR)}" 2>/dev/null; then
    echo "!!! CREDIT FLOOR <\$$FLOOR -- harvest+destroy ALL"; 
    for p in $PAIRS; do id="${p##*:}"; echo y | vastai destroy instance "$id" >/dev/null 2>&1; done
    finish "CREDIT_FLOOR"; exit 0
  fi
  [ "$nd" = 3 ] && { finish "ALL_DONE"; echo "=== ALL 3 DONE + EXTRACTED + COMMITTED ==="; exit 0; }
  sleep 150
done
finish "LOOP_BUDGET_EXHAUSTED"
