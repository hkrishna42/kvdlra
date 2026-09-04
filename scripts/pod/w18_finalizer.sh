#!/bin/bash
# Detached finalizer: waits for the watchdog's DONE marker, then pushes the committed
# results to origin so the full G1 (incl 32K) reaches GitHub even with no active session.
cd /Users/hari/Desktop/kv-dlra || exit 1
export PATH="/Users/hari/.local/bin:$PATH"
for i in $(seq 1 480); do
  if [ -f results/w18_harvest/DONE.txt ]; then
    sleep 20  # let the watchdog's commit settle
    git push origin week7 >/tmp/w18_finalizer.log 2>&1
    echo "PUSHED $(date): $(cat results/w18_harvest/DONE.txt)" >> /tmp/w18_finalizer.log
    exit 0
  fi
  sleep 120
done
