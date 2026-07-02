#!/bin/bash
# Onstart batch for the Week-5 Axis-C (long-context amortization) run on 8B.
# Robust-pod pattern (see [[vastai-pod-flakiness-jul2026]]): clone the public repo,
# install deps WITHOUT touching the pod image's CUDA torch, run w5_longctx.py, and
# let it echo the results JSON to stdout (scraped from `vastai logs`, never SSH).
# Markers below bracket the phases so a `vastai logs` scrape can find them.
set -x
export HF_HUB_ENABLE_HF_TRANSFER=1          # fast 8B download (ungated unsloth mirror; no token)
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false

cd /root || exit 1
git clone --depth 1 --branch week3 https://github.com/hkrishna42/kvdlra.git 2>&1 | tail -3
cd kvdlra || exit 1

# Install everything EXCEPT torch (keep the image's CUDA build). kvpress first, then
# pin transformers/datasets last so their versions win over kvpress's looser pins.
pip install -q hf_transfer numpy matplotlib kvpress 2>&1 | tail -5
pip install -q 'transformers==5.8.0' 'datasets==2.21.0' 2>&1 | tail -5
echo "===DEPS_DONE==="
python -c "import torch,transformers; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)"

# The amortization test. n_windows kept modest to bound wall-clock; the harness
# runs each ctx in try/except so a 32K OOM still yields shorter-ctx results, and
# prints the full JSON between ===W5_LONGCTX_JSON_BEGIN/END=== for scraping.
echo "===RUN_BEGIN==="
PYTHONPATH=src python scripts/w5_longctx.py \
  --model unsloth/Meta-Llama-3.1-8B-Instruct --device cuda --dtype bfloat16 \
  --context-lens 1024 4096 8192 16384 32768 \
  --ranks 64 128 256 --evict-ratios 0.5 0.7 0.85 --bits 4 --n-windows 4 \
  --out-json results/w5-longctx-8b.json 2>&1
echo "===RUN_DONE==="
