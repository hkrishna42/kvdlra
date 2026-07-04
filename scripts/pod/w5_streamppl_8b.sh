#!/bin/bash
# Onstart batch for the Week-5 Axis-B benchmark at 8B: streaming perplexity under
# constant-memory decode caches (BUG streaming cache vs MorphKV vs SnapKV-decode vs
# StreamingLLM vs full cache) at matched stored memory, 8192 scored decode steps.
# Proven recipe (`[[vastai-pod-flakiness-jul2026]]`): `python -u` so results stream
# to `vastai logs`, kvpress==0.5.1 + transformers==5.8.0 + datasets==2.21.0 on the
# image's CUDA torch, results JSON between ===MARKERS===. Run on an RTX 6000 Ada
# (ran clean; an A6000 previously hit a cusolver SVD stall -- decode-time BUG SVDs
# here are tiny (r+b)^2, but stay with the proven card). No HF token needed (the
# unsloth 8B mirror is ungated).
set -x
export HF_HUB_ENABLE_HF_TRANSFER=1
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false
# pytorch/pytorch:2.11 images ship a PEP-668 "externally managed" python:
# bare `pip install` refuses and the whole batch silently runs dep-less.
export PIP_BREAK_SYSTEM_PACKAGES=1

cd /root || exit 1
git clone --depth 1 --branch week3 https://github.com/hkrishna42/kvdlra.git 2>&1 | tail -3
cd kvdlra || exit 1

pip install -q hf_transfer numpy matplotlib "kvpress==0.5.1" 2>&1 | tail -5
pip install -q 'transformers==5.8.0' 'datasets==2.21.0' 2>&1 | tail -5
# Fail FAST and LOUD if deps are missing -- never blow through to ===ALL_DONE===
# with a dep-less python (that pattern burned pod 43751804).
python -c "import torch, transformers, kvpress, datasets, matplotlib" \
  || { echo "===DEPS_FAILED==="; exit 1; }
echo "===DEPS_DONE==="
python -c "import torch,transformers; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)"

MODEL=unsloth/Meta-Llama-3.1-8B-Instruct

# Main tier: BUG r128/W2048/w64 (~499 token-equivalents per layer at n=1024);
# MorphKV capacity + StreamingLLM window are solved to the same budget in-script.
# 3 docs x 8192 scored decode steps -> degradation curves 16x past the budget.
echo "===STREAMPPL_BEGIN==="
PYTHONPATH=src python -u scripts/w5_streamppl.py --model "$MODEL" --device cuda --dtype bfloat16 \
  --corpus wikitext-103 --prefill 1024 --g-tokens 8192 --bin-size 512 \
  --n-docs 3 --doc-stride 60000 \
  --rank 128 --coord-budget 2048 --recent-window 64 --absorb-block 32 --morph-recent 32 \
  --out-json results/w5-streamppl-8b.json --fig figures/week5/streamppl_8b.png 2>&1
echo "===STREAMPPL_DONE==="

# Aggressive tier: half the budget (rank 64 / W 1024 / w 32) -- does the ranking
# change when every method is squeezed harder?
echo "===STREAMPPL_TIER2_BEGIN==="
PYTHONPATH=src python -u scripts/w5_streamppl.py --model "$MODEL" --device cuda --dtype bfloat16 \
  --corpus wikitext-103 --prefill 1024 --g-tokens 8192 --bin-size 512 \
  --n-docs 3 --doc-stride 60000 \
  --rank 64 --coord-budget 1024 --recent-window 32 --absorb-block 16 --morph-recent 32 \
  --out-json results/w5-streamppl-8b-tier2.json --fig figures/week5/streamppl_8b_tier2.png 2>&1
echo "===STREAMPPL_TIER2_DONE==="
echo "===ALL_DONE==="
