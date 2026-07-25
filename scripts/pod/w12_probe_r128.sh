#!/bin/bash
# Week-12 T2b: trial-matched r128 probe. The W11 probe found the r128 exact tier
# STARVED (0/8 codes at hh<=256) but ran on probe-default (trial,seed); the RULER
# wins came from t{0,1}x s{0,1}. This closes the sampling gap: same task builder,
# the EXACT four (trial,seed) combos, rank 128, the two deployed hh budgets only
# (the probe re-ingests the whole cache per budget -- 6 budgets at r128@32K would
# cost ~$3-4 alone and W11 already established budget-independence). 32K first
# (the wins live there), 16K second. Harvest = the short printed per-hh lines
# ONLY (vastai logs truncates at 500 chars; the JSON block is unusable).
set -x
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_HUB_DISABLE_XET=1
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false
export PIP_BREAK_SYSTEM_PACKAGES=1

MODEL="${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
CHUNK="${CHUNK:-4096}"

cd /root || exit 1
for attempt in 1 2 3 4 5; do
  rm -rf kvdlra
  git clone --branch week7 https://github.com/hkrishna42/kvdlra.git 2>&1 | tail -3
  [ -d kvdlra/scripts ] && break
  echo "===CLONE_RETRY_${attempt}==="; sleep 5
done
cd kvdlra || { echo "===CLONE_FAILED==="; exit 1; }
grep -q "niah_multikey" scripts/w11_probe.py || { echo "===STALE_CLONE_NO_MULTIKEY==="; exit 1; }
pip install -q hf_transfer hf_xet numpy matplotlib "kvpress==0.5.1" 2>&1 | tail -5
pip install -q 'transformers==5.8.0' 'datasets==2.21.0' 2>&1 | tail -5
echo "===DEPS_DONE==="
python -c "import torch,transformers,kvpress; print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || echo "===DEPS_FAILED==="
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL'); print('===MODEL_OK===')" 2>&1 | tail -3 || echo "===MODEL_FAILED==="

echo "===PROBE_R128_BEGIN==="
for ctx in 32768 16384; do
  for ts in "0 0" "0 1" "1 0" "1 1"; do
    set -- $ts
    echo "===PROBE_c${ctx}_t${1}s${2}_BEGIN==="
    PYTHONPATH=src python -u scripts/w11_probe.py --model "$MODEL" --device cuda \
      --dtype bfloat16 --task niah_multikey --ctx "$ctx" --rank 128 \
      --hh-budgets 256 1024 --chunk "$CHUNK" --hh-neighbor 1 \
      --trial "$1" --seed "$2" \
      --out-json "results/w12-probe8b-mk-r128-c${ctx}-t${1}s${2}.json" 2>&1
    echo "===PROBE_c${ctx}_t${1}s${2}_DONE==="
  done
done
echo "===PROBE_R128_DONE==="
echo "===ALL_DONE==="
