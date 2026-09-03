#!/bin/bash
# Week-18 MODE driver. SOURCED by scripts/pod/w18_boot.sh from /root/kvdlra after a
# SHA-pinned clone, with env (MODE/MODEL/TAG/CHUNK/DTYPE) and emit() already defined.
# Being cloned at $SHA, this file is itself SHA-pinned -- edit it here, push, then the
# pod runs exactly this. Never launched directly; run bash -n on it for syntax only.
#
# Pre-registered Week-18 GPU matrix (see docs/week18-kickoff.md Phase 2 / the plan):
#   g1  quantization baseline (quant-2bit/4bit + flagship + q4-compose)   ~$60  BLOCKING
#   g2  realistic filler + depth grid; official-benchmark anchor          ~$65  BLOCKING
#   g3  eviction-in-grid (ea-k0.1 + snapkv-k0.1 + bugEVICT), mk included  ~$25
#   g4  firming (32K n->12; mk on Qwen/Mistral; marquee n=16; r256 law)   ~$30-40
#   g5  persistence / cold-load bench                                     ~$10
# Fund bars are pre-registered in the plan; every "beats" needs McNemar p<0.05 on the
# per-trial [trial] lines. NEVER drop --chunk (bugslash/warmup-seed require chunk>0).
set -x
cd /root/kvdlra || exit 1

# --chunk is unconditional (the w17 lesson): single-shot prefill bypasses the exact
# tier and warmup-seed. Every out-JSON is folded back through the log via emit().
RULER(){ PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda \
           --dtype "$DTYPE" --chunk "$CHUNK" "$@" 2>&1; }
PPL(){ PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda \
         --dtype "$DTYPE" --chunk "$CHUNK" --window 512 --n-samples 8 --no-ruler "$@" 2>&1; }
PPL4(){ PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda \
          --dtype "$DTYPE" --chunk "$CHUNK" --window 512 --n-samples 4 --no-ruler "$@" 2>&1; }

# --------------------------------------------------------------- MODE bodies
# NOTE: Phase-1 synthesis fills these from the merged workstream findings (the quant
# arm names, --filler/--depths flags, and per-model cells do not all exist yet). Each
# is a real, syntactically-complete stub that fails loud rather than pretending to run.

g1(){ echo "===W18_G1_TODO_PHASE1_${TAG}==="; echo "quant baseline: fill from WS1/WS2"; return 1; }
g2(){ echo "===W18_G2_TODO_PHASE1_${TAG}==="; echo "realistic filler + anchor: fill from WS3/WS6"; return 1; }
g3(){ echo "===W18_G3_TODO_PHASE1_${TAG}==="; echo "eviction grid: fill from WS4"; return 1; }
g4(){ echo "===W18_G4_TODO_PHASE1_${TAG}==="; echo "firming: fill from WS7"; return 1; }
g5(){ echo "===W18_G5_TODO_PHASE1_${TAG}==="; echo "persistence bench: fill from WS5"; return 1; }

case "$MODE" in
  g1) g1 ;;
  g2) g2 ;;
  g3) g3 ;;
  g4) g4 ;;
  g5) g5 ;;
  *) echo "===UNKNOWN_MODE_${MODE}==="; exit 1 ;;
esac
echo "===W18_DONE_${MODE}_${TAG}==="
