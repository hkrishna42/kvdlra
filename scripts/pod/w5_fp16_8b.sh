#!/bin/bash
# Onstart batch for the Week-5 CLEAN fp16 (no-TurboQuant) 8B runs at 32K + 64K:
# (1) fp16 perplexity frontier (pure BUG vs SnapKV/EA vs BUG-hybrid) on PG19, and
# (2) fp16 multi-key RULER retrieval. Proven recipe: `python -u` (stream results to
# `vastai logs`), `kvpress==0.5.1`. Run on an RTX 6000 Ada (ran clean; the A6000
# hit a cusolver SVD stall -- mitigated further here by BUG block_size=512).
set -x
export HF_HUB_ENABLE_HF_TRANSFER=1
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false

cd /root || exit 1
git clone --depth 1 --branch week3 https://github.com/hkrishna42/kvdlra.git 2>&1 | tail -3
cd kvdlra || exit 1

pip install -q hf_transfer numpy matplotlib "kvpress==0.5.1" 2>&1 | tail -5
pip install -q 'transformers==5.8.0' 'datasets==2.21.0' 2>&1 | tail -5
echo "===DEPS_DONE==="
python -c "import torch,transformers; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)"

MODEL=unsloth/Meta-Llama-3.1-8B-Instruct

# (1) fp16 perplexity frontier on PG19 (long books -> clean windows at 32K/64K).
echo "===FP16_BEGIN==="
PYTHONPATH=src python -u scripts/w5_fp16_longctx.py --model "$MODEL" --device cuda --dtype bfloat16 \
  --context-lens 32768 65536 --corpus wikitext-103 --max-tokens 3000000 \
  --ranks 64 128 256 --evict-ratios 0.75 0.875 0.94 --exact-fracs 0.03 --n-windows 8 \
  --out-json results/w5-fp16-longctx-8b.json 2>&1
echo "===FP16_DONE==="

# (2) fp16 multi-key retrieval (BUG quant_bits=None): does BUG's summary-of-everything
# hold the queried key where fp16 eviction (keeping only ~6-15% of tokens) drops it?
echo "===RULER_BEGIN==="
PYTHONPATH=src python -u scripts/w5_ruler.py --model "$MODEL" --device cuda --dtype bfloat16 \
  --no-quant --context-lens 32768 65536 --n-keys 12 --n-queries 3 --seeds 0 1 \
  --out-json results/w5-ruler-fp16-8b.json 2>&1
echo "===RULER_DONE==="
echo "===ALL_DONE==="
