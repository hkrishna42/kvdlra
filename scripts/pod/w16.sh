#!/bin/bash
# Week-16 NeurIPS-readiness pods (onstart-batch, SSH-less). Two MODEs:
#   MODE=tier2  generality on a NEW family ($MODEL): a stage-0 GPU probe gate
#               (w16_tier2_probe -- full must recover the needle or KILL), then the
#               both-axes test: 16K RULER {single,mv,vt} bugslash-r128 +/-s32 (warmup-seed)
#               + think/palu + 16K ppl; then 32K {vt,mv} n=4 + 32K ppl.
#   MODE=tier1  Llama-3.1-8B firm: add the MISSING think-c0.5 and firm bugSseed/palu
#               vt+mv at n=8 (16K) / n=4 (32K) for the "beats Palu / ties ThinK" claim.
# Launch per pod by sed/env of MODE, MODEL, TAG. Harvest ^===/^\[niah/^\[vt/^\[pplw from
# `vastai logs`; destroy from the launcher. Does NOT self-destruct.
set -x
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_HUB_DISABLE_XET=1
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false
export PIP_BREAK_SYSTEM_PACKAGES=1

MODE="${MODE:-tier2}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TAG="${TAG:-qwen}"
CHUNK="${CHUNK:-4096}"
DTYPE="${DTYPE:-bfloat16}"
PRANKS="${PRANKS:-16 32 64 128 256}"   # ppl sweep (spans the instability onset; near-full is slow+moot)
RRANKS="${RRANKS:-16 32 64 128}"        # retrieval sweep (hh-tier extreme-compression niche)

echo "===MODE_${MODE}_${TAG}==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd /root || exit 1
for attempt in 1 2 3 4 5; do
  rm -rf kvdlra
  git clone --branch week7 https://github.com/hkrishna42/kvdlra.git 2>&1 | tail -3
  [ -d kvdlra/scripts ] && break
  echo "===CLONE_RETRY_${attempt}==="; sleep 5
done
cd kvdlra || { echo "===CLONE_FAILED==="; exit 1; }
pip install -q hf_transfer hf_xet numpy matplotlib "kvpress==0.5.1" 2>&1 | tail -5
pip install -q 'transformers==5.8.0' 'datasets==2.21.0' 2>&1 | tail -5
echo "===DEPS_DONE==="
python -c "import torch,transformers,kvpress; print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || echo "===DEPS_FAILED==="

RULER(){ PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" --chunk "$CHUNK" "$@" 2>&1; }
PPL(){ PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
        --chunk "$CHUNK" --window 512 --n-samples 8 --no-ruler "$@" 2>&1; }

ppl_block(){ # $1=T $2=tag
  PPL --T "$1" --methods full think palu --think-ratios 0.5 --palu-ranks 0.5 \
      --out-json "results/w16-${TAG}-ppl$2-base.json"
  PPL --T "$1" --methods bugslash --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed \
      --out-json "results/w16-${TAG}-ppl$2-bug.json"
  PPL --T "$1" --methods bugslash --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed \
      --score-rank 32 --out-json "results/w16-${TAG}-ppl$2-s32.json"
}

tier2(){
  echo "===T2_PROBE_BEGIN_${TAG}==="
  PYTHONPATH=src python -u scripts/w16_tier2_probe.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --ctx 2048 2>&1 | tee /root/probe_${TAG}.log
  if ! grep -q "FUND -- authorize GPU" /root/probe_${TAG}.log; then
    echo "===T2_GATE_KILL_${TAG}==="; return 1
  fi
  echo "===T2_GATE_FUND_${TAG}==="

  echo "===T2_16K_UNCAPPED_BEGIN_${TAG}==="
  RULER --context-lens 16384 --tasks niah_single niah_multivalue vt \
    --methods bugslash think palu --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed \
    --think-ratios 0.5 --palu-ranks 0.5 --n-trials 4 --seeds 0 1 \
    --out-json "results/w16-${TAG}-16k-uncapped.json"
  echo "===T2_16K_S32_BEGIN_${TAG}==="
  RULER --context-lens 16384 --tasks niah_multivalue vt \
    --methods bugslash --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed --score-rank 32 \
    --n-trials 4 --seeds 0 1 --out-json "results/w16-${TAG}-16k-s32.json"
  echo "===T2_16K_PPL_BEGIN_${TAG}==="
  ppl_block 16384 16
  echo "===T2_16K_DONE_${TAG}==="

  echo "===T2_32K_UNCAPPED_BEGIN_${TAG}==="
  RULER --context-lens 32768 --tasks vt niah_multivalue \
    --methods bugslash think palu --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed \
    --think-ratios 0.5 --palu-ranks 0.5 --n-trials 2 --seeds 0 1 \
    --out-json "results/w16-${TAG}-32k-uncapped.json"
  echo "===T2_32K_S32_BEGIN_${TAG}==="
  RULER --context-lens 32768 --tasks vt niah_multivalue \
    --methods bugslash --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed --score-rank 32 \
    --n-trials 2 --seeds 0 1 --out-json "results/w16-${TAG}-32k-s32.json"
  echo "===T2_32K_PPL_BEGIN_${TAG}==="
  ppl_block 32768 32
  echo "===T2_32K_DONE_${TAG}==="
}

tier1(){ # Llama-3.1-8B: add think-c0.5 (missing) + firm bugSseed/palu vt+mv; ppl already banked.
  echo "===T1_16K_UNCAPPED_BEGIN==="
  RULER --context-lens 16384 --tasks niah_multivalue vt \
    --methods bugslash think palu --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed \
    --think-ratios 0.5 --palu-ranks 0.5 --n-trials 4 --seeds 0 1 \
    --out-json "results/w16-${TAG}-16k-uncapped.json"
  echo "===T1_16K_S32_BEGIN==="
  RULER --context-lens 16384 --tasks niah_multivalue vt \
    --methods bugslash --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed --score-rank 32 \
    --n-trials 4 --seeds 0 1 --out-json "results/w16-${TAG}-16k-s32.json"
  echo "===T1_32K_UNCAPPED_BEGIN==="
  RULER --context-lens 32768 --tasks vt niah_multivalue \
    --methods bugslash think palu --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed \
    --think-ratios 0.5 --palu-ranks 0.5 --n-trials 2 --seeds 0 1 \
    --out-json "results/w16-${TAG}-32k-uncapped.json"
  echo "===T1_32K_S32_BEGIN==="
  RULER --context-lens 32768 --tasks vt niah_multivalue \
    --methods bugslash --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed --score-rank 32 \
    --n-trials 2 --seeds 0 1 --out-json "results/w16-${TAG}-32k-s32.json"
}

sweep(){ # r128 fails on non-Llama (bugSseed ppl 7.3->47->467). Map (memory, ppl, retrieval) vs
         # rank per model. Retrieval & ppl want OPPOSITE ranks: the hh256 exact tier drives
         # retrieval (favours LOW rank -- a big gist blinds selection), fluency needs HIGH rank.
         # So ppl sweeps to full rank (PRANKS), retrieval sweeps the low-mid niche (RRANKS).
         # memory = ratio_fp16 in each row; think/palu baselines come from the tier2 logs.
  # RULER FIRST (fast: short decode) -- the generality question. ppl second (slow: long
  # scoring -> n-samples 4). NB at r16/r32 ppl is FINE (Qwen 9.3/8.2, Mistral 7.0/6.7) but
  # r128 blows up -> integrator instability at high rank, not a compressibility limit.
  echo "===SWEEP_RULER_BEGIN_${TAG}==="
  RULER --context-lens 16384 --tasks niah_single niah_multivalue vt \
    --methods bugslash --ranks $RRANKS --hh-budgets 256 --hh-neighbor 1 --warmup-seed \
    --n-trials 2 --seeds 0 1 --out-json "results/w16-${TAG}-sweepruler.json"
  echo "===SWEEP_PPL_BEGIN_${TAG}==="
  PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --T 16384 --chunk "$CHUNK" --window 512 --n-samples 4 --no-ruler \
    --methods bug bugslash --ranks $PRANKS --hh-budgets 256 --hh-neighbor 1 --warmup-seed \
    --out-json "results/w16-${TAG}-sweepppl.json" 2>&1
  echo "===SWEEP_DONE_${TAG}==="
}

case "$MODE" in
  tier2) tier2 ;;
  tier1) tier1 ;;
  sweep) sweep ;;
  *) echo "===UNKNOWN_MODE_${MODE}==="; exit 1 ;;
esac
echo "===ALL_DONE_${MODE}_${TAG}==="
