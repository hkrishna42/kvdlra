#!/bin/bash
# Onstart batch for the Week-7 Axis-B confirmation at 8B: does adaptive
# coordinate retention (bugA) close BUG's deep-horizon gap to eviction, where
# Week 6 found the crossover (7.74 morph vs 7.87 BUG tier-1; 8.39 vs 9.17
# tier-2)? Methods: full, bug (FIFO baseline), bugA (attention retention),
# morph, snapkvD -- the Week-6 set + the one surviving Week-7 variant (D and E
# were decisive negatives at 1B, not worth pod time). Two budget tiers.
#
# Proven recipe (`[[vastai-pod-flakiness-jul2026]]`): `python -u` so results
# stream to `vastai logs`; PEP-668 pip fix; hard DEPS check; results JSON as
# base64 folded to short lines between markers (vast truncates ~500-char lines).
# RTX 6000 Ada ran clean twice for Week 6. No HF token (unsloth mirror ungated).
#
# IMPORTANT: this clones branch **week7** -- push it to origin before renting.
set -x
export HF_HUB_ENABLE_HF_TRANSFER=1
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false
export PIP_BREAK_SYSTEM_PACKAGES=1

cd /root || exit 1
git clone --depth 1 --branch week7 https://github.com/hkrishna42/kvdlra.git 2>&1 | tail -3
cd kvdlra || exit 1

pip install -q hf_transfer numpy matplotlib "kvpress==0.5.1" 2>&1 | tail -5
pip install -q 'transformers==5.8.0' 'datasets==2.21.0' 2>&1 | tail -5
python -c "import torch, transformers, kvpress, datasets, matplotlib" \
  || { echo "===DEPS_FAILED==="; exit 1; }
echo "===DEPS_DONE==="
python -c "import torch,transformers; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)"

MODEL=unsloth/Meta-Llama-3.1-8B-Instruct
METHODS=full,bug,bugA,morph,snapkvD

# Main tier: BUG r128/W2048/w64 (~499 token-equivalents per layer at n=1024);
# every bounded method is solved to the same budget in-script. 3 docs x 8192
# scored decode steps -> degradation curves 16x past the budget.
echo "===STREAMPPL_BEGIN==="
PYTHONPATH=src python -u scripts/w5_streamppl.py --model "$MODEL" --device cuda --dtype bfloat16 \
  --corpus wikitext-103 --prefill 1024 --g-tokens 8192 --bin-size 512 \
  --n-docs 3 --doc-stride 60000 \
  --rank 128 --coord-budget 2048 --recent-window 64 --absorb-block 32 --morph-recent 32 \
  --methods "$METHODS" \
  --out-json results/w7-streamppl-8b.json --fig figures/week7/streamppl_8b.png 2>&1
echo "===STREAMPPL_DONE==="
echo "===JSON_B64_T1_BEGIN==="
base64 -w 120 results/w7-streamppl-8b.json
echo "===JSON_B64_T1_END==="

# Aggressive tier: half the budget (rank 64 / W 1024 / w 32) -- Week 6 found the
# crossover most decisive here (morph 8.39 vs BUG 9.17). Does retention help?
echo "===STREAMPPL_TIER2_BEGIN==="
PYTHONPATH=src python -u scripts/w5_streamppl.py --model "$MODEL" --device cuda --dtype bfloat16 \
  --corpus wikitext-103 --prefill 1024 --g-tokens 8192 --bin-size 512 \
  --n-docs 3 --doc-stride 60000 \
  --rank 64 --coord-budget 1024 --recent-window 32 --absorb-block 16 --morph-recent 32 \
  --methods "$METHODS" \
  --out-json results/w7-streamppl-8b-tier2.json --fig figures/week7/streamppl_8b_tier2.png 2>&1
echo "===STREAMPPL_TIER2_DONE==="
echo "===JSON_B64_T2_BEGIN==="
base64 -w 120 results/w7-streamppl-8b-tier2.json
echo "===JSON_B64_T2_END==="
echo "===ALL_DONE==="
