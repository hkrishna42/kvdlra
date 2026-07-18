#!/bin/bash
# Week-11 balanced-config + firming + Q1-probe run. One script, three MODEs so the
# workloads launch as PARALLEL pods (sed the MODE= line per pod at launch):
#   MODE=16k    (1) RULER 16K bugslash r128/r256 x hh{256,1024}  (Q2 retrieval)
#               (2) ppl 16K for the same new arms
#               (3) Q1 probe: surprise-rank of the needle at 16K AND 32K
#               (4) RULER 16K firming: bugS-r32 / bugEVICT / ea  (Q1 n>=5 rerun)
#   MODE=32k    (1) RULER 32K bugslash r128/r256 x hh{256,1024}  (Q2 retrieval)
#               (2) ppl 32K for the same new arms
#   MODE=firm32 (1) RULER 32K firming: bugS-r32 / bugEVICT / ea, n=6/cell
# Every cell lands at n = n-trials x 2 seeds = 6 (>= the n>=4 / n>=5 asks).
# Cheap/high-value blocks first so partial results survive a dead pod.
set -x
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_HUB_DISABLE_XET=1
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false
export PIP_BREAK_SYSTEM_PACKAGES=1

MODE="${MODE:-16k}"
MODEL="${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
NEW_RANKS="${NEW_RANKS:-128 256}"
FIRM_RANKS="${FIRM_RANKS:-32}"
HH="${HH:-256 1024}"
EVICT="${EVICT:-0.1}"
TASKS="${TASKS:-niah_single niah_multikey niah_multivalue vt}"
CHUNK="${CHUNK:-4096}"
DTYPE="${DTYPE:-bfloat16}"
NTRIALS="${NTRIALS:-3}"
SEEDS="${SEEDS:-0 1}"

echo "===MODE_${MODE}==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

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

ruler(){ # ruler <tag> <ctx> <out> <extra args...>
  local tag="$1" ctx="$2" out="$3"; shift 3
  echo "===${tag}_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --context-lens "$ctx" --tasks $TASKS --hh-budgets $HH --hh-neighbor 1 \
    --chunk "$CHUNK" --n-trials "$NTRIALS" --seeds $SEEDS "$@" \
    --out-json "$out" 2>&1
  emit "$tag" "$out"
  echo "===${tag}_DONE==="
}

if [ "$MODE" = "16k" ]; then
  ruler R16NEW 16384 results/w11-r128-r16.json --methods bugslash --ranks $NEW_RANKS

  echo "===PPL16_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --T 16384 --chunk "$CHUNK" --window 512 --n-samples 2 \
    --methods bugslash --ranks $NEW_RANKS --hh-budgets $HH --hh-neighbor 1 --no-ruler \
    --out-json results/w11-r128-ppl16.json 2>&1
  emit PPL16 results/w11-r128-ppl16.json
  echo "===PPL16_DONE==="

  # Q1 probe: does the needle's surprise rank rise 16K -> 32K? 2 (trial,seed)
  # combos per ctx; JSON also lands between ===W11_PROBE_JSON_BEGIN/END=== lines.
  echo "===PROBE_BEGIN==="
  for ctx in 16384 32768; do
    for ts in "0 0" "1 1"; do
      set -- $ts
      PYTHONPATH=src python -u scripts/w11_probe.py --model "$MODEL" --device cuda \
        --dtype "$DTYPE" --ctx "$ctx" --rank 32 --hh-budgets 64 128 256 512 1024 2048 \
        --chunk "$CHUNK" --hh-neighbor 1 --trial "$1" --seed "$2" \
        --out-json "results/w11-probe-c${ctx}-t${1}s${2}.json" 2>&1
      emit "PROBE_${ctx}_t${1}s${2}" "results/w11-probe-c${ctx}-t${1}s${2}.json"
    done
  done
  echo "===PROBE_DONE==="

  ruler R16FIRM 16384 results/w11-firm-r16.json \
    --methods bugslash bugevict ea --ranks $FIRM_RANKS --evict-keeps $EVICT

elif [ "$MODE" = "32k" ]; then
  ruler R32NEW 32768 results/w11-r128-r32.json --methods bugslash --ranks $NEW_RANKS

  echo "===PPL32_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --T 32768 --chunk "$CHUNK" --window 512 --n-samples 2 \
    --methods bugslash --ranks $NEW_RANKS --hh-budgets $HH --hh-neighbor 1 --no-ruler \
    --out-json results/w11-r128-ppl32.json 2>&1
  emit PPL32 results/w11-r128-ppl32.json
  echo "===PPL32_DONE==="

elif [ "$MODE" = "firm32" ]; then
  ruler R32FIRM 32768 results/w11-firm-r32.json \
    --methods bugslash bugevict ea --ranks $FIRM_RANKS --evict-keeps $EVICT
fi
echo "===ALL_DONE==="
