#!/bin/bash
# Week-11 Q1 probe pod: 8B MULTIKEY surprise-capture at 16K vs 32K. The 1B sweep
# showed budget-INDEPENDENT capture improving with ctx (6/8@4K -> 7/8@8K -> 8/8@16K);
# this confirms/refutes at 8B across the 16K->32K boundary where the decision table
# saw the retrieval jump. Also one r128 run per ctx (the balanced config's gist).
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
emit(){ echo "===${1}_RESULT_BEGIN==="; base64 -w0 "$2" 2>/dev/null | fold -w 400; echo ""; echo "===${1}_RESULT_END==="; }

echo "===PROBE8B_BEGIN==="
for ctx in 16384 32768; do
  for ts in "0 0" "1 1"; do
    set -- $ts
    PYTHONPATH=src python -u scripts/w11_probe.py --model "$MODEL" --device cuda \
      --dtype bfloat16 --task niah_multikey --ctx "$ctx" --rank 32 \
      --hh-budgets 64 128 256 512 1024 2048 --chunk "$CHUNK" --hh-neighbor 1 \
      --trial "$1" --seed "$2" --out-json "results/w11-probe8b-mk-c${ctx}-t${1}s${2}.json" 2>&1
    emit "PROBE8B_${ctx}_t${1}s${2}" "results/w11-probe8b-mk-c${ctx}-t${1}s${2}.json"
  done
  PYTHONPATH=src python -u scripts/w11_probe.py --model "$MODEL" --device cuda \
    --dtype bfloat16 --task niah_multikey --ctx "$ctx" --rank 128 \
    --hh-budgets 64 128 256 512 1024 2048 --chunk "$CHUNK" --hh-neighbor 1 \
    --trial 0 --seed 0 --out-json "results/w11-probe8b-mk-r128-c${ctx}.json" 2>&1
  emit "PROBE8B_R128_${ctx}" "results/w11-probe8b-mk-r128-c${ctx}.json"
done
echo "===PROBE8B_DONE==="
echo "===ALL_DONE==="
