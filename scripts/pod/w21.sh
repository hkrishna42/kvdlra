#!/bin/bash
# Week-21 MODE driver. SOURCED by scripts/pod/w18_boot.sh (DRIVER=scripts/pod/w21.sh) from
# /root/kvdlra after a SHA-pinned clone, with env (MODE/MODEL/TAG/CHUNK/DTYPE) and emit()
# already defined. Being cloned at $SHA, this file is itself SHA-pinned -- edit here, push,
# then the pod runs exactly this. Never launched directly; `bash -n` for syntax only.
#
# Pre-registered Week-21 matrix (lane L2 item 1; gate G2 first checkbox):
#   filler  the FILLER-REALISM DIAGNOSTIC -- prereg/filler_realism.md, DECISIONS D-005.
#           Llama-3.1-8B, 16K, the four in-house RULER tasks, n=12, five arms, ONE
#           variable changed against the committed archive: the haystack corpus.
#
# Launch (the prereg commit must be an ANCESTOR of $SHA):
#   SHA=$(git rev-parse HEAD)
#   uvx vastai create instance $OFFER \
#     --image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel --disk 80 \
#     --env "-e MODE=filler -e MODEL=meta-llama/Llama-3.1-8B-Instruct -e TAG=llama \
#            -e SHA=$SHA -e DRIVER=scripts/pod/w21.sh" \
#     --onstart scripts/pod/w18_boot.sh --label kvdlra-w21-llama
#   pods.txt line: filler-llama:<id>:filler:llama
set -x
cd /root/kvdlra || exit 1

RULER(){ PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda \
           --dtype "$DTYPE" --chunk "$CHUNK" "$@" 2>&1; }

T4="--tasks niah_single niah_multikey niah_multivalue vt"
KIVI="--methods quant --quant-scheme kivi --quant-nbits 2"
RH="--ranks 64 --hh-budgets 256 --hh-neighbor 1 --warmup-seed"
QC="$RH --bug-quant-bits 4 --bug-quant-budget 512"

# filler: the Day-1 filler-realism diagnostic (prereg/filler_realism.md). The in-house
# generator cycles TEN fixed sentences (scripts/w4_needle.py:46-57) ~1400x at 16K -- a
# near-rank-deficient haystack, exactly the structure a rank-64 gist absorbs best -- and
# the same q4 cell that scores 1.00/1.00/1.00 here scores 0.00 on all nine official RULER
# tasks (results/w20-close-report.md §4). This swaps `cycle` for a seed-shuffled
# WikiText-2 TEST sentence pool (--filler wikitext) and changes NOTHING else: same model,
# ctx, tasks, n, seeds, chunk, arm flags as the rows it is compared against.
#
# Exactly the five command lines this function runs (one per arm; $CHUNK=4096, $DTYPE=
# bfloat16, $MODEL=meta-llama/Llama-3.1-8B-Instruct, $TAG=llama):
#
#  1. PYTHONPATH=src python -u scripts/w10_ruler.py --model $MODEL --device cuda \
#       --dtype $DTYPE --chunk $CHUNK --context-lens 16384 \
#       --tasks niah_single niah_multikey niah_multivalue vt --filler wikitext \
#       --methods full --n-trials 6 --seeds 0 1 \
#       --out-json results/w21-$TAG-filler-full.json
#  2. ... --methods bugslash --ranks 64 --hh-budgets 256 --hh-neighbor 1 --warmup-seed \
#       --n-trials 6 --seeds 0 1 --out-json results/w21-$TAG-filler-r64.json
#  3. ... --methods bugslash --ranks 64 --hh-budgets 256 --hh-neighbor 1 --warmup-seed \
#       --bug-quant-bits 4 --bug-quant-budget 512 --n-trials 6 --seeds 0 1 \
#       --out-json results/w21-$TAG-filler-q4.json
#  4. ... --methods quant --quant-scheme kivi --quant-nbits 2 --n-trials 6 --seeds 0 1 \
#       --out-json results/w21-$TAG-filler-kivi2.json
#  5. ... --methods quant --quant-scheme kivi --quant-nbits 2 --n-trials 6 --seeds 0 1 \
#       --chunk 0 --out-json results/w21-$TAG-filler-kivi2ss.json
#
# Invariants that are NOT free to change:
#  * All four tasks in ONE call per arm. w10_ruler.py:390 sets max_new=12 for a
#    niah_single-only call and 40 otherwise, and every reference row came from a
#    four-task call. w18.sh g2 ran the r64 arm on niah_single alone WITH a --depths grid
#    and its baselines on four tasks WITHOUT one; that asymmetry is not repeated here.
#  * No --depths: the archive used the generator's default placement, so the filler is
#    the only variable.
#  * --chunk stays > 0 for bugslash (single-shot prefill bypasses the exact tier;
#    build_arms raises). Only the arm-5 control passes --chunk 0, and it is a quant arm.
#  * Order is cheap-control-first then the two arms the decision rule names, so a pod
#    that dies early still lands an interpretable result.
# No perplexity line: w10_frontier.py has no --filler (perplexity scores a separate
# corpus sweep, never the haystack), so a PPL4 call here would re-measure the committed
# 16K value at ~1h of pod time and say nothing about filler realism.
FILL="--context-lens 16384 $T4 --filler wikitext --n-trials 6 --seeds 0 1"
filler(){
  echo "===W21_FILLER_FULL_BEGIN_${TAG}==="
  RULER $FILL --methods full \
    --out-json "results/w21-${TAG}-filler-full.json"; emit "FILLER_FULL" "results/w21-${TAG}-filler-full.json"
  echo "===W21_FILLER_R64_BEGIN_${TAG}==="
  RULER $FILL --methods bugslash $RH \
    --out-json "results/w21-${TAG}-filler-r64.json"; emit "FILLER_R64" "results/w21-${TAG}-filler-r64.json"
  echo "===W21_FILLER_Q4_BEGIN_${TAG}==="
  RULER $FILL --methods bugslash $QC \
    --out-json "results/w21-${TAG}-filler-q4.json"; emit "FILLER_Q4" "results/w21-${TAG}-filler-q4.json"
  echo "===W21_FILLER_KIVI2_BEGIN_${TAG}==="
  RULER $FILL $KIVI \
    --out-json "results/w21-${TAG}-filler-kivi2.json"; emit "FILLER_KIVI2" "results/w21-${TAG}-filler-kivi2.json"
  echo "===W21_FILLER_KIVI2SS_BEGIN_${TAG}==="
  RULER $FILL $KIVI --chunk 0 \
    --out-json "results/w21-${TAG}-filler-kivi2ss.json"; emit "FILLER_KIVI2SS" "results/w21-${TAG}-filler-kivi2ss.json"
}

case "$MODE" in
  filler) filler ;;
  *) echo "===UNKNOWN_MODE_${MODE}==="; exit 1 ;;
esac
echo "===W21_DONE_${MODE}_${TAG}==="
