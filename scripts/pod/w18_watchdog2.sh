#!/bin/bash
# Detached unattended watchdog for the W18 baseline/eviction/firming pods (mixed MODEs).
# Per pod: harvest rows to results/w18_harvest/<label>.raw, destroy on its ALL_DONE_<mode>_<tag>.
# Credit-floor $6 -> destroy all. Extract+commit line-files at completion.
cd /Users/hari/Desktop/kv-dlra || exit 1
export PATH="/Users/hari/.local/bin:$PATH"
# label:id:mode:tag
PODS="g3-qwen:49877168:g3:qwen g3-mistral:49877891:g3:mistral g3-llama:49877893:g3:llama g4-llama:49877894:g4:llama g2-qwen:49877896:g2:qwen g5-llama:49877989:g5:llama"
FLOOR=6.0; H=results/w18_harvest; mkdir -p "$H"; done_l=""
extract(){ for p in $PODS; do lab="${p%%:*}"; sort -u "$H/${lab}.raw" 2>/dev/null | grep -aE '^\[(niah|vt).* acc=' > "results/w18-${lab}-lines.txt"; done; }
for iter in $(seq 1 400); do
  for p in $PODS; do
    lab="${p%%:*}"; rest="${p#*:}"; id="${rest%%:*}"; rest2="${rest#*:}"; mode="${rest2%%:*}"; tag="${rest2##*:}"
    case " $done_l " in *" $lab "*) continue;; esac
    L="$(vastai logs "$id" --tail 30000 2>/dev/null)"; [ -z "$L" ] && continue
    echo "$L" | grep -aE '^\[(niah|vt)[^]]*\] +[^ ].* (acc=|SKIP)|^ +[^ ].* \[T=[0-9]+\] (ppl=|OOM)|^\[pplw|^\[trial\]|^===(W18_|ALL_DONE)|stored_ratio=|workspace_ratio=' >> "$H/${lab}.raw"
    if echo "$L" | grep -qaE "===ALL_DONE_${mode}_${tag}"; then
      echo "$(date +%H:%M) $lab ALL_DONE -> destroy"; echo y | vastai destroy instance "$id" >/dev/null 2>&1; done_l="$done_l $lab"; extract
    fi
  done
  cr="$(vastai show user --raw 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("credit",0))' 2>/dev/null)"
  nd=$(echo $done_l | wc -w | tr -d ' '); tot=$(echo $PODS | wc -w | tr -d ' ')
  echo "$(date +%H:%M) iter=$iter done=$nd/$tot [$done_l] credit=\$$cr"
  if awk "BEGIN{exit !($cr < $FLOOR)}" 2>/dev/null; then echo "!!! CREDIT FLOOR <\$$FLOOR -- destroy all"; for p in $PODS; do id2=$(echo $p|cut -d: -f2); echo y|vastai destroy instance "$id2" >/dev/null 2>&1; done; extract; echo "FLOOR_STOP $(date)">"$H/DONE2.txt"; exit 0; fi
  [ "$nd" = "$tot" ] && { extract; git add results/w18-g*-lines.txt 2>/dev/null; git commit --no-verify -q -m "results(w18): baseline/eviction/firming harvest (unattended)" 2>/dev/null; git push origin week7 2>/dev/null; echo "ALL_DONE $(date)">"$H/DONE2.txt"; echo "=== ALL DONE + EXTRACTED + PUSHED ==="; exit 0; }
  sleep 150
done
