#!/bin/bash
# Week-11 DECISION TABLE: BUG vs bugS (SurpriseSLASH) vs bugEVICT vs ea/full,
# all 4 RULER tasks + perplexity, at 16K AND 32K. Cheap arms first (16K RULER of
# the new arms, then ppl) so the high-value data lands even if credit runs tight;
# the expensive 32K 4-task RULER runs last. Baselines (morph/snapkv/think/palu/
# shadow) reuse the GOAL-A 16K 4-task data -- not re-run here. Destroy the pod after.
set -x
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_HUB_DISABLE_XET=1
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false
export PIP_BREAK_SYSTEM_PACKAGES=1

MODEL="${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
RANKS="${RANKS:-32}"
HH="${HH:-256 1024}"
EVICT="${EVICT:-0.1}"
TASKS="${TASKS:-niah_single niah_multikey niah_multivalue vt}"
CHUNK="${CHUNK:-4096}"
DTYPE="${DTYPE:-bfloat16}"

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
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL'); print('===MODEL_OK===')" 2>&1 | tail -3 || echo "===MODEL_FAILED==="
emit(){ echo "===${1}_RESULT_BEGIN==="; base64 -w0 "$2" 2>/dev/null | fold -w 400; echo ""; echo "===${1}_RESULT_END==="; }

# (1) 16K RULER -- only the NEW arms (baselines already have GOAL-A 16K 4-task).
echo "===R16_BEGIN==="
PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
  --context-lens 16384 --tasks $TASKS --methods bugslash bugevict --ranks $RANKS \
  --hh-budgets $HH --hh-neighbor 1 --chunk "$CHUNK" --n-trials 3 --seeds 0 \
  --out-json results/w11-table-r16.json 2>&1
emit R16 results/w11-table-r16.json
echo "===R16_DONE==="

# (2) Perplexity 16K + 32K for the decision arms (fast, high value).
echo "===PPL_BEGIN==="
PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
  --T 16384 32768 --chunk "$CHUNK" --window 512 --n-samples 2 \
  --methods full bug bugslash bugevict ea --ranks $RANKS \
  --hh-budgets $HH --hh-neighbor 1 --evict-keeps $EVICT --no-ruler \
  --out-json results/w11-table-ppl.json 2>&1
emit PPL results/w11-table-ppl.json
echo "===PPL_DONE==="

# (3) 32K RULER -- the full decision set, all 4 tasks (the expensive block, last).
echo "===R32_BEGIN==="
PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
  --context-lens 32768 --tasks $TASKS --methods full bug bugslash bugevict ea --ranks $RANKS \
  --hh-budgets $HH --hh-neighbor 1 --evict-keeps $EVICT --chunk "$CHUNK" --n-trials 2 --seeds 0 \
  --out-json results/w11-table-r32.json 2>&1
emit R32 results/w11-table-r32.json
echo "===R32_DONE==="
echo "===ALL_DONE==="
