#!/bin/bash
# Complete the Week-11 decision table: the BASELINE methods (MorphKV / SnapKV /
# ThinK / Palu / ShadowKV) at the cells the earlier runs missed --
#   (1) 32K RULER, all 4 tasks (Week-10 only ran the single needle for these, and
#       MorphKV/SnapKV had nothing at 32K due to the now-fixed harness bug);
#   (2) 16K perplexity (only the BUG variants had 16K ppl).
# The BUG variants + ea + full already have every cell. Destroy the pod after.
set -x
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_HUB_DISABLE_XET=1
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false
export PIP_BREAK_SYSTEM_PACKAGES=1

MODEL="${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
TASKS="${TASKS:-niah_single niah_multikey niah_multivalue vt}"
MORPH="${MORPH:-0.1 0.25 0.5}"
EVICT="${EVICT:-0.1 0.25 0.5}"
THINK="${THINK:-0.3 0.5 0.7}"
PALU="${PALU:-0.25 0.5}"
SHADOW="${SHADOW:-64 128}"
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

# (1) 16K perplexity for the baselines (fast, high value first).
echo "===BPPL_BEGIN==="
PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
  --T 16384 --chunk "$CHUNK" --window 512 --n-samples 2 --methods morph snapkv think palu shadow \
  --morph-keeps $MORPH --evict-keeps $EVICT --think-ratios $THINK --palu-ranks $PALU --shadow-ranks $SHADOW \
  --no-ruler --out-json results/w11-base-ppl16.json 2>&1
emit BPPL results/w11-base-ppl16.json
echo "===BPPL_DONE==="

# (2) 32K RULER, all 4 tasks, for the baselines (the big block).
echo "===BR32_BEGIN==="
PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
  --context-lens 32768 --tasks $TASKS --methods morph snapkv think palu shadow \
  --morph-keeps $MORPH --evict-keeps $EVICT --think-ratios $THINK --palu-ranks $PALU --shadow-ranks $SHADOW \
  --chunk "$CHUNK" --n-trials 2 --seeds 0 --out-json results/w11-base-r32.json 2>&1
emit BR32 results/w11-base-r32.json
echo "===BR32_DONE==="
echo "===ALL_DONE==="
