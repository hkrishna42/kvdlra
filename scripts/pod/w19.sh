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

# A4: the 1/T asymptotic with data -- a 64K point (Llama only: the one family with a native
# 128K window; Qwen/Mistral are 32K-native). Gist is O(rn+hn), constant in T, so the stored
# ratio falls as 1/T (0.149->0.139 / 0.085->0.075 from 16K->32K): the ONE frontier claim no
# fixed-bit quantizer can match. Flagship n=8 (the long pole: ~2x a 32K trial), the eviction
# and fair-quant comparators n=12 (fast), + same-pod ppl. Needs an 80GB card.
RH="--ranks 64 --hh-budgets 256 --hh-neighbor 1 --warmup-seed"
a4(){
  T=65536
  echo "===W19_A4_R${T}_BEGIN_${TAG}==="
  RULER --context-lens "$T" $T4 --methods ea --evict-keeps 0.1 --n-trials 6 --seeds 0 1 \
    --out-json "results/w19-${TAG}-a4ea-r${T}.json"; emit "A4EA_R${T}" "results/w19-${TAG}-a4ea-r${T}.json"
  RULER --context-lens "$T" $T4 $KIVI --quant-nbits 2 4 --n-trials 6 --seeds 0 1 \
    --out-json "results/w19-${TAG}-a4kivi-r${T}.json"; emit "A4KIVI_R${T}" "results/w19-${TAG}-a4kivi-r${T}.json"
  RULER --context-lens "$T" $T4 --methods bugslash $RH --n-trials 4 --seeds 0 1 \
    --out-json "results/w19-${TAG}-a4flag-r${T}.json"; emit "A4FLAG_R${T}" "results/w19-${TAG}-a4flag-r${T}.json"
  echo "===W19_A4_PPL${T}_BEGIN_${TAG}==="
  PPL4 --T "$T" --methods full ea --evict-keeps 0.1 \
    --out-json "results/w19-${TAG}-a4ppl-base.json"; emit "A4PPL_BASE" "results/w19-${TAG}-a4ppl-base.json"
  PPL4 --T "$T" $KIVI --quant-nbits 2 4 \
    --out-json "results/w19-${TAG}-a4ppl-kivi.json"; emit "A4PPL_KIVI" "results/w19-${TAG}-a4ppl-kivi.json"
  PPL4 --T "$T" --methods bugslash $RH \
    --out-json "results/w19-${TAG}-a4ppl-flag.json"; emit "A4PPL_FLAG" "results/w19-${TAG}-a4ppl-flag.json"
}

# A2: the official-benchmark anchor. The NVIDIA RULER generator (pinned commit) builds the
# prompts ON the pod (their essay/noise haystacks, needle types, templates; 12 samples per
# task at 16K, seed 42) and scripts/w19_official_ruler.py runs the SAME arms through them
# (their tokens_to_generate + string_match_all). Closes the "self-authored benchmark" gap.
RULER_SHA=c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
TASKS9="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt"
OFF(){ PYTHONPATH=src python -u scripts/w19_official_ruler.py --model "$MODEL" --device cuda \
         --dtype "$DTYPE" --chunk "$CHUNK" --data-dir /root/ruler_data --context-len 16384 "$@" 2>&1; }
a2(){
  echo "===W19_A2_PREP_BEGIN_${TAG}==="
  pip install -q wonderwords tenacity nltk html2text beautifulsoup4 2>&1 | tail -1
  python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
  rm -rf /root/RULER; git clone -q https://github.com/NVIDIA/RULER.git /root/RULER || { echo "===RULER_CLONE_FAILED==="; return 1; }
  git -C /root/RULER checkout -q "$RULER_SHA" || { echo "===RULER_CHECKOUT_FAILED==="; return 1; }
  echo "===RULER_SHA_$(git -C /root/RULER rev-parse HEAD)==="
  ( cd /root/RULER/scripts/data/synthetic/json && python download_paulgraham_essay.py 2>&1 | tail -2 )
  [ -s /root/RULER/scripts/data/synthetic/json/PaulGrahamEssays.json ] || echo "===ESSAYS_MISSING==="
  for task in $TASKS9; do
    ( cd /root/RULER/scripts/data && python prepare.py --save_dir /root/ruler_data --benchmark synthetic \
        --task "$task" --tokenizer_path "$MODEL" --tokenizer_type hf --max_seq_length 16384 \
        --num_samples 12 --random_seed 42 --model_template_type base 2>&1 | tail -1 )
    echo "===A2_PREP_${task}_$(wc -l < /root/ruler_data/${task}/validation.jsonl 2>/dev/null || echo 0)==="
  done
  echo "===W19_A2_RUN_BEGIN_${TAG}==="
  OFF --tasks $TASKS9 --methods full think palu ea --think-ratios 0.5 --palu-ranks 0.5 --evict-keeps 0.1 \
    --out-json "results/w19-${TAG}-a2base.json"; emit "A2BASE" "results/w19-${TAG}-a2base.json"
  OFF --tasks $TASKS9 $KIVI --quant-nbits 2 4 \
    --out-json "results/w19-${TAG}-a2kivi.json"; emit "A2KIVI" "results/w19-${TAG}-a2kivi.json"
  OFF --tasks $TASKS9 --methods bugslash $RH \
    --out-json "results/w19-${TAG}-a2flag.json"; emit "A2FLAG" "results/w19-${TAG}-a2flag.json"
}

# A3: the realized systems win -- persisted-cache cold start (serialize -> reload -> H2D ->
# attend-ready wall-clock) for full KV vs the flagship vs the fair 2-bit baseline, CUDA,
# Llama 16K/32K. Turns the measured byte ratio (g5: 0.150x/0.139x) into a deployment
# existence proof. Rows: ^\[persist  (harvested like acc= rows).
a3(){
  echo "===W19_A3_PERSIST_BEGIN_${TAG}==="
  PYTHONPATH=src python -u scripts/w19_persist.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --chunk "$CHUNK" --context-lens 16384 32768 --repeats 5 --tmp /root/persist \
    --methods full bugslash quant $RH --quant-nbits 2 --quant-scheme kivi \
    --out-json "results/w19-${TAG}-a3persist.json" 2>&1
  emit "A3PERSIST" "results/w19-${TAG}-a3persist.json"
}

case "$MODE" in
  a1diag) a1diag ;;
  a1) a1 ;;
  a2) a2 ;;
  a3) a3 ;;
  a4) a4 ;;
  *) echo "===UNKNOWN_MODE_${MODE}==="; exit 1 ;;
esac
echo "===W19_DONE_${MODE}_${TAG}==="
