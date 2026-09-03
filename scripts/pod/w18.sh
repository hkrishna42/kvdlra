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
# The flagship extreme-compression arm (r64-h256, warmup-seeded): the W17 config.
RH="--ranks 64 --hh-budgets 256 --hh-neighbor 1 --warmup-seed"
# n=12 = 6 trials x 2 seeds; n=16 = 8 x 2. Baselines for every RULER cell.
BASE="think palu ea"

# G1 (blocking): 2/4-bit KV quantization baseline + the flagship + the bugS-q4 sub-cliff
# compose arm (no seed -- seed+quant is fenced). RULER{single,mk,mv,vt} + ppl, 16K & 32K.
g1(){
  for T in 16384 32768; do
    echo "===W18_G1_R${T}_BEGIN_${TAG}==="
    RULER --context-lens "$T" --tasks niah_single niah_multikey niah_multivalue vt \
      --methods quant --quant-nbits 2 4 --n-trials 6 --seeds 0 1 \
      --out-json "results/w18-${TAG}-g1quant-r${T}.json"; emit "G1QUANT_R${T}" "results/w18-${TAG}-g1quant-r${T}.json"
    RULER --context-lens "$T" --tasks niah_single niah_multikey niah_multivalue vt \
      --methods bugslash $RH --n-trials 6 --seeds 0 1 \
      --out-json "results/w18-${TAG}-g1flag-r${T}.json"; emit "G1FLAG_R${T}" "results/w18-${TAG}-g1flag-r${T}.json"
    # sub-cliff compose (bugS-r64-h256-q4, no seed): quantize the coord tier to 4 bits.
    RULER --context-lens "$T" --tasks niah_single niah_multikey niah_multivalue vt \
      --methods bugslash --ranks 64 --hh-budgets 256 --hh-neighbor 1 \
      --bug-quant-bits 4 --bug-quant-budget 512 --n-trials 6 --seeds 0 1 \
      --out-json "results/w18-${TAG}-g1q4-r${T}.json"; emit "G1Q4_R${T}" "results/w18-${TAG}-g1q4-r${T}.json"
    echo "===W18_G1_PPL${T}_BEGIN_${TAG}==="
    PPL4 --T "$T" --methods quant --quant-nbits 2 4 \
      --out-json "results/w18-${TAG}-g1quant-ppl${T}.json"; emit "G1QUANT_PPL${T}" "results/w18-${TAG}-g1quant-ppl${T}.json"
    PPL4 --T "$T" --methods bugslash $RH \
      --out-json "results/w18-${TAG}-g1flag-ppl${T}.json"; emit "G1FLAG_PPL${T}" "results/w18-${TAG}-g1flag-ppl${T}.json"
  done
}

# G2 (blocking): realistic-filler + depth grid vs the archived cyclic filler; flagship +
# baselines + ea-k0.1, 16K n=12. (Official-RULER anchor is a separate heavier run.)
g2(){
  echo "===W18_G2_FILLER_BEGIN_${TAG}==="
  RULER --context-lens 16384 --tasks niah_single --filler wikitext --depths 0.1 0.3 0.5 0.7 0.9 \
    --methods bugslash $RH --n-trials 6 --seeds 0 1 \
    --out-json "results/w18-${TAG}-g2flag.json"; emit "G2FLAG" "results/w18-${TAG}-g2flag.json"
  RULER --context-lens 16384 --tasks niah_single niah_multikey niah_multivalue vt --filler wikitext \
    --methods think palu ea --evict-keeps 0.1 --think-ratios 0.5 --palu-ranks 0.5 \
    --n-trials 6 --seeds 0 1 \
    --out-json "results/w18-${TAG}-g2base.json"; emit "G2BASE" "results/w18-${TAG}-g2base.json"
}

# G3: eviction IN the grid (the "cannot enter" reword needs measured points), mk included.
g3(){
  for T in 16384 32768; do
    echo "===W18_G3_R${T}_BEGIN_${TAG}==="
    RULER --context-lens "$T" --tasks niah_single niah_multikey niah_multivalue vt \
      --methods ea snapkv --evict-keeps 0.1 --n-trials 6 --seeds 0 1 \
      --out-json "results/w18-${TAG}-g3evict-r${T}.json"; emit "G3EVICT_R${T}" "results/w18-${TAG}-g3evict-r${T}.json"
    RULER --context-lens "$T" --tasks niah_single niah_multikey niah_multivalue vt \
      --methods bugevict --hh-budgets 256 --n-trials 6 --seeds 0 1 \
      --out-json "results/w18-${TAG}-g3bugevict-r${T}.json"; emit "G3BUGEVICT_R${T}" "results/w18-${TAG}-g3bugevict-r${T}.json"
  done
}

# G4: firming -- the marquee at n=16 (single+mk added), the r/n=0.25 r256 control, and the
# marquee ppl on the SAME pod (kills the cross-week splice). Llama-focused (TAG gates cells).
g4(){
  echo "===W18_G4_MARQUEE_BEGIN_${TAG}==="
  RULER --context-lens 32768 --tasks niah_single niah_multikey niah_multivalue vt \
    --methods bugslash --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed --score-rank 32 \
    --n-trials 8 --seeds 0 1 \
    --out-json "results/w18-${TAG}-g4marq.json"; emit "G4MARQ" "results/w18-${TAG}-g4marq.json"
  RULER --context-lens 32768 --tasks niah_single niah_multikey niah_multivalue vt \
    --methods think palu --think-ratios 0.5 --palu-ranks 0.5 --n-trials 8 --seeds 0 1 \
    --out-json "results/w18-${TAG}-g4base.json"; emit "G4BASE" "results/w18-${TAG}-g4base.json"
  # r/n=0.25 wall control: seeded r256 (should collapse on the hard tasks).
  RULER --context-lens 32768 --tasks niah_single niah_multikey niah_multivalue vt \
    --methods bugslash --ranks 256 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed \
    --n-trials 6 --seeds 0 1 \
    --out-json "results/w18-${TAG}-g4r256.json"; emit "G4R256" "results/w18-${TAG}-g4r256.json"
  # marquee ppl same-pod (was spliced from Week-15): full + baselines + the marquee arm.
  echo "===W18_G4_PPL_BEGIN_${TAG}==="
  PPL --T 32768 --methods full think palu --think-ratios 0.5 --palu-ranks 0.5 \
    --out-json "results/w18-${TAG}-g4ppl-base.json"; emit "G4PPL_BASE" "results/w18-${TAG}-g4ppl-base.json"
  PPL --T 32768 --methods bugslash --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed --score-rank 32 \
    --out-json "results/w18-${TAG}-g4ppl-marq.json"; emit "G4PPL_MARQ" "results/w18-${TAG}-g4ppl-marq.json"
}

# G5: persistence / storage on CUDA -- workspace + peak resident + cold-load size for the
# reconstruct-then-attend arms (full vs bug vs the flagship bugslash; w16_storage measures
# streaming caches, not presses -- the ~1.06x-vs-full resident number the panel wants).
g5(){
  echo "===W18_G5_STORAGE_BEGIN_${TAG}==="
  PYTHONPATH=src python -u scripts/w16_storage.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --context-lens 16384 32768 --chunk "$CHUNK" --methods full bug bugslash \
    --ranks 64 --hh-budgets 256 --warmup-seed \
    --out-json "results/w18-${TAG}-g5storage.json" 2>&1
  emit "G5STORAGE" "results/w18-${TAG}-g5storage.json"
}

# g1quant: the KIVI quant baseline ONLY (RULER + ppl, 2/4-bit). Split out because
# optimum-quanto needs the CUDA-toolkit (-devel) image to build its kernel, while the
# flagship/eviction arms run fine on the lighter -runtime image -- so the quant baseline
# rides its own devel pod and its line-files merge with the rest (quant acc/ppl are
# image-independent: same model, same QuantizedCache config).
g1quant(){
  for T in 16384 32768; do
    echo "===W18_G1QUANT_R${T}_BEGIN_${TAG}==="
    RULER --context-lens "$T" --tasks niah_single niah_multikey niah_multivalue vt \
      --methods quant --quant-nbits 2 4 --n-trials 6 --seeds 0 1 \
      --out-json "results/w18-${TAG}-g1quant-r${T}.json"; emit "G1QUANT_R${T}" "results/w18-${TAG}-g1quant-r${T}.json"
    echo "===W18_G1QUANT_PPL${T}_BEGIN_${TAG}==="
    PPL4 --T "$T" --methods quant --quant-nbits 2 4 \
      --out-json "results/w18-${TAG}-g1quant-ppl${T}.json"; emit "G1QUANT_PPL${T}" "results/w18-${TAG}-g1quant-ppl${T}.json"
  done
}

# g1diag: is the quant baseline's zero retrieval a decode bug or real? full (control) +
# quant-8bit (near-lossless: if IT scores 0 on a single needle, the decode path is broken)
# + 4/2-bit, one task, n=4 -- fast + definitive.
g1diag(){
  echo "===W18_G1DIAG_BEGIN_${TAG}==="
  RULER --context-lens 16384 --tasks niah_single \
    --methods full quant --quant-nbits 8 4 2 --n-trials 2 --seeds 0 1 \
    --out-json "results/w18-${TAG}-g1diag.json"; emit "G1DIAG" "results/w18-${TAG}-g1diag.json"
}

case "$MODE" in
  g1) g1 ;;
  g1quant) g1quant ;;
  g1diag) g1diag ;;
  g2) g2 ;;
  g3) g3 ;;
  g4) g4 ;;
  g5) g5 ;;
  *) echo "===UNKNOWN_MODE_${MODE}==="; exit 1 ;;
esac
echo "===W18_DONE_${MODE}_${TAG}==="
