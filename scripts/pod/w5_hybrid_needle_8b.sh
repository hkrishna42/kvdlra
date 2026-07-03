#!/bin/bash
# Onstart batch for the Week-5 8B runs: (1) hybrid fair perplexity sweep and
# (2) long-context needle retrieval (Experiment B). Robust-pod pattern
# ([[vastai-pod-flakiness-jul2026]]): clone the public repo, install deps WITHOUT
# touching the image's CUDA torch, run both scripts, and let each echo its results
# JSON to stdout (scraped from `vastai logs`, never SSH). Markers bracket phases.
set -x
export HF_HUB_ENABLE_HF_TRANSFER=1          # fast 8B download (ungated unsloth mirror; no token)
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

# (1) Hybrid vs pure BUG vs eviction, fair perplexity-vs-memory. Does keeping a few
# high-norm tokens exact let BUG pass SnapKV where pure BUG could not?
echo "===HYBRID_BEGIN==="
PYTHONPATH=src python -u scripts/w5_hybrid.py --model "$MODEL" --device cuda --dtype bfloat16 \
  --context-lens 1024 8192 32768 --ranks 128 256 --exact-fracs 0.05 \
  --evict-ratios 0.5 0.7 0.85 --bits 4 --n-windows 3 \
  --out-json results/w5-hybrid-8b.json 2>&1
echo "===HYBRID_DONE==="

# (2) Experiment B: long-context needle retrieval. BUG should hold its accuracy as
# context grows while eviction drops the (un-cued) needle. (32K dropped to bound time.)
echo "===NEEDLE_BEGIN==="
PYTHONPATH=src python -u scripts/w5_needle.py --model "$MODEL" --device cuda --dtype bfloat16 \
  --context-lens 4096 16384 --depths 0.25 0.5 0.75 --passcodes 48213 70561 \
  --out-json results/w5-needle-8b.json 2>&1
echo "===NEEDLE_DONE==="
echo "===ALL_DONE==="
