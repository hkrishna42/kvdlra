#!/bin/bash
# Onstart batch for the Week-10 GPU long-context frontier: the three axes
# (perplexity / RULER accuracy / LongBench F1) vs honestly-counted memory, at
# 32K/64K, for every method (BUG ranks, MorphKV, SnapKV, ExpectedAttention, ThinK,
# Palu, and ShadowKV if integrated). Chunked ingest (Phase 3) keeps the streaming
# caches OOM-safe; presses run single-shot on the big card.
#
# Recipe follows the Week-5/7 pod lessons: SSH is unusable -> git-clone the pushed
# branch, `python -u` (unbuffered -> streams to `vastai logs`), pinned kvpress/
# transformers, and results base64-FOLDED between markers (vast logs truncate ~500-
# char lines, so raw json.dumps would be lost). Scrape each RESULT block from
# `vastai logs`, base64 -d it to recover the JSON.
#
# Env: MODEL (default 1B unsloth mirror -- ungated, no HF token needed), TLIST
# (context lengths), CHUNK (chunked-prefill block), NSAMP/NTRIALS/NEX (trial counts).
set -x
export HF_HUB_ENABLE_HF_TRANSFER=1
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false
export PIP_BREAK_SYSTEM_PACKAGES=1

MODEL="${MODEL:-unsloth/Llama-3.2-1B-Instruct}"
TLIST="${TLIST:-32768 65536}"
RULER_CTX="${RULER_CTX:-32768}"
LB_MAXLEN="${LB_MAXLEN:-16384}"
CHUNK="${CHUNK:-4096}"
NSAMP="${NSAMP:-3}"
NTRIALS="${NTRIALS:-4}"
NEX="${NEX:-20}"
DTYPE="${DTYPE:-bfloat16}"

cd /root || exit 1
git clone --depth 1 --branch week7 https://github.com/hkrishna42/kvdlra.git 2>&1 | tail -3
cd kvdlra || exit 1

pip install -q hf_transfer numpy matplotlib "kvpress==0.5.1" 2>&1 | tail -5
pip install -q 'transformers==5.8.0' 'datasets==2.21.0' 2>&1 | tail -5
echo "===DEPS_DONE==="
python -c "import torch,transformers,kvpress; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)" || echo "===DEPS_FAILED==="

emit() {  # emit <marker> <json-path> : base64-fold the result so vast logs keep it
  echo "===${1}_RESULT_BEGIN==="
  base64 -w0 "$2" 2>/dev/null | fold -w 400
  echo ""
  echo "===${1}_RESULT_END==="
}

# (1) Perplexity frontier (supporting axis) -- chunked ingest at 32K/64K.
echo "===PPL_BEGIN==="
PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
  --T $TLIST --chunk "$CHUNK" --window 512 --n-samples "$NSAMP" \
  --out-json results/w10-frontier.json 2>&1
emit PPL results/w10-frontier.json
echo "===PPL_DONE==="

# (2) RULER accuracy (PRIMARY) -- compress-then-query, eviction's weak spot.
echo "===RULER_BEGIN==="
PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
  --context-lens $RULER_CTX --chunk "$CHUNK" --n-trials "$NTRIALS" --seeds 0 1 \
  --out-json results/w10-ruler.json 2>&1
emit RULER results/w10-ruler.json
echo "===RULER_DONE==="

# (3) LongBench F1 (PRIMARY) -- realistic QA, query in-prompt (fairer to eviction).
echo "===LB_BEGIN==="
PYTHONPATH=src python -u scripts/w10_longbench.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
  --tasks qasper multifieldqa_en hotpotqa 2wikimqa --max-len "$LB_MAXLEN" --chunk "$CHUNK" \
  --n-examples "$NEX" --out-json results/w10-longbench.json 2>&1
emit LB results/w10-longbench.json
echo "===LB_DONE==="

echo "===ALL_DONE==="
