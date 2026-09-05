#!/bin/bash
# Week-19 MODE driver. SOURCED by scripts/pod/w18_boot.sh (DRIVER=scripts/pod/w19.sh) from
# /root/kvdlra after a SHA-pinned clone, with env (MODE/MODEL/TAG/CHUNK/DTYPE) and emit()
# already defined. Being cloned at $SHA, this file is itself SHA-pinned -- edit here, push,
# then the pod runs exactly this. Never launched directly; `bash -n` for syntax only.
#
# Pre-registered Week-19 GPU matrix (docs/week19-kickoff.md Phase A):
#   a1diag  ONE devel pod, Qwen 16K niah_single, n=4: validates the fair-KIVI path before
#           any fan-out. The decisive signal is a `quant-2bit-kivi acc=` ROW (not a boot
#           marker). Arms: full + quant-4bit (token = the W18 zero, replicated under chunked
#           prefill) + quant-{2,4}bit-kivi (quanto) + quant-{8,2}bit-kivi-hqq (8-bit = the
#           decode-path control; 2-bit = cross-implementation) + kivi ppl@16K (the W18 OOM).
#   a1      per family: quant-{2,4}bit-kivi x {16K,32K} x RULER{single,mk,mv,vt} + ppl (n=12),
#           + the 8-bit control on all four tasks at 16K (n=4).
# Fund bar (pre-registered): if the flagship holds single+mv >= 0.7 Wilson-lo where 2-bit
# KIVI <= 0.3, the exclusive band is claimed; otherwise the band claim is retired and
# §subcliff/§limits are reworded to the narrow story. Report WHATEVER it shows.
# Quant arms need the *-devel* image (quanto JIT-builds its CUDA kernel; hqq is pure torch).
set -x
cd /root/kvdlra || exit 1

RULER(){ PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda \
           --dtype "$DTYPE" --chunk "$CHUNK" "$@" 2>&1; }
PPL4(){ PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda \
          --dtype "$DTYPE" --chunk "$CHUNK" --window 512 --n-samples 4 --no-ruler "$@" 2>&1; }

T4="--tasks niah_single niah_multikey niah_multivalue vt"
KIVI="--methods quant --quant-scheme kivi"

a1diag(){
  echo "===W19_A1DIAG_BEGIN_${TAG}==="
  RULER --context-lens 16384 --tasks niah_single --methods full quant --quant-nbits 4 \
    --quant-scheme token --n-trials 2 --seeds 0 1 \
    --out-json "results/w19-${TAG}-a1diag-token.json"; emit "A1DIAG_TOKEN" "results/w19-${TAG}-a1diag-token.json"
  RULER --context-lens 16384 --tasks niah_single $KIVI --quant-nbits 2 4 --n-trials 2 --seeds 0 1 \
    --out-json "results/w19-${TAG}-a1diag-kivi.json"; emit "A1DIAG_KIVI" "results/w19-${TAG}-a1diag-kivi.json"
  RULER --context-lens 16384 --tasks niah_single $KIVI --quant-backend hqq --quant-nbits 8 2 \
    --n-trials 2 --seeds 0 1 \
    --out-json "results/w19-${TAG}-a1diag-hqq.json"; emit "A1DIAG_HQQ" "results/w19-${TAG}-a1diag-hqq.json"
  echo "===W19_A1DIAG_PPL_BEGIN_${TAG}==="
  PPL4 --T 16384 $KIVI --quant-nbits 2 4 \
    --out-json "results/w19-${TAG}-a1diag-ppl16384.json"; emit "A1DIAG_PPL" "results/w19-${TAG}-a1diag-ppl16384.json"
  # the W18 32K OOM cell (even on 80GB): does chunked quant prefill fit on this card?
  PPL4 --T 32768 $KIVI --quant-nbits 2 \
    --out-json "results/w19-${TAG}-a1diag-ppl32768.json"; emit "A1DIAG_PPL32" "results/w19-${TAG}-a1diag-ppl32768.json"
}

a1(){
  for T in 16384 32768; do
    echo "===W19_A1_R${T}_BEGIN_${TAG}==="
    RULER --context-lens "$T" $T4 $KIVI --quant-nbits 2 4 --n-trials 6 --seeds 0 1 \
      --out-json "results/w19-${TAG}-a1kivi-r${T}.json"; emit "A1KIVI_R${T}" "results/w19-${TAG}-a1kivi-r${T}.json"
    echo "===W19_A1_PPL${T}_BEGIN_${TAG}==="
    PPL4 --T "$T" $KIVI --quant-nbits 2 4 \
      --out-json "results/w19-${TAG}-a1kivi-ppl${T}.json"; emit "A1KIVI_PPL${T}" "results/w19-${TAG}-a1kivi-ppl${T}.json"
  done
  echo "===W19_A1_HQQ8_BEGIN_${TAG}==="
  RULER --context-lens 16384 $T4 $KIVI --quant-backend hqq --quant-nbits 8 --n-trials 2 --seeds 0 1 \
    --out-json "results/w19-${TAG}-a1hqq8-r16384.json"; emit "A1HQQ8_R16384" "results/w19-${TAG}-a1hqq8-r16384.json"
}

case "$MODE" in
  a1diag) a1diag ;;
  a1) a1 ;;
  *) echo "===UNKNOWN_MODE_${MODE}==="; exit 1 ;;
esac
echo "===W19_DONE_${MODE}_${TAG}==="
