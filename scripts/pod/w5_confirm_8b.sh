#!/bin/bash
# Onstart batch for the Week-5 8B CONFIRMATION runs: (1) 4-bit-exact hybrid fair
# perplexity sweep and (2) RULER-lite multi-key retrieval (the discriminating test).
# Fixes from the first attempt baked in: `python -u` (unbuffered -> results stream
# to `vastai logs`) and `kvpress==0.5.1` (no transformers conflict). Runs on an
# RTX 6000 Ada (ran cleanly before; the A6000 hit a cusolver SVD stall).
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

# (1) 4-bit-exact hybrid vs pure BUG vs eviction (fair; kept tokens now 4-bit).
echo "===HYBRID_BEGIN==="
PYTHONPATH=src python -u scripts/w5_hybrid.py --model "$MODEL" --device cuda --dtype bfloat16 \
  --context-lens 1024 8192 32768 --ranks 128 256 --exact-fracs 0.03 0.10 \
  --evict-ratios 0.5 0.7 0.85 --bits 4 --n-windows 3 \
  --out-json results/w5-hybrid-8b.json 2>&1
echo "===HYBRID_DONE==="

# (2) RULER-lite multi-key retrieval: retrieve 1 of 12 keys among distractors at
# aggressive memory. At 8B long ctx, BUG r128 amortizes to ~SnapKV keep-15% memory,
# so this is a fair matched-memory retrieval test -- where eviction should drop the
# un-cued key while BUG's low-rank summary keeps it.
echo "===RULER_BEGIN==="
PYTHONPATH=src python -u scripts/w5_ruler.py --model "$MODEL" --device cuda --dtype bfloat16 \
  --context-lens 4096 16384 --n-keys 12 --n-queries 3 --seeds 0 1 --bits 4 \
  --out-json results/w5-ruler-8b.json 2>&1
echo "===RULER_DONE==="
echo "===ALL_DONE==="
