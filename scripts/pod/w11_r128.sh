#!/bin/bash
# Week-11 balanced-config + firming runs, v2 -- TRIMMED after measuring the real
# per-trial cost (~8-13 min/trial at 16-32K, not the ~2 min first assumed; the v1
# full grid projected ~55 pod-hours). One script, four MODEs launched as PARALLEL
# pods (sed the MODE= line per pod at launch):
#   MODE=new16  RULER 16K bugslash r128 x hh{256,1024}, 4 tasks, n=2x2seeds (Q2)
#               then ppl 16K for the r{128,256} x hh{256,1024} grid
#   MODE=new32  same at 32K
#   MODE=firm16 RULER 16K firming: bugS-r32-h256 / bugEVICT-h256 / ea on the 3
#               HARD tasks (needle saturates at 100 for these arms), n=2x2seeds --
#               POOLED with the existing n=3 (16K) / n=2 (32K) rows -> n>=6/cell
#   MODE=firm32 same at 32K
# r256 RULER dropped (ppl-only): bugS-r128 is the balanced-config headline; r256
# (~0.26-0.28x) sits outside the sub-0.1x retrieval story.
set -x
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_HUB_DISABLE_XET=1
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false
export PIP_BREAK_SYSTEM_PACKAGES=1

MODE="${MODE:-new16}"
MODEL="${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
HH="${HH:-256 1024}"
EVICT="${EVICT:-0.1}"
CHUNK="${CHUNK:-4096}"
DTYPE="${DTYPE:-bfloat16}"

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

new(){ # new <tag> <ctx>
  local tag="$1" ctx="$2"
  echo "===${tag}_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --context-lens "$ctx" --tasks niah_single niah_multikey niah_multivalue vt \
    --methods bugslash --ranks 128 --hh-budgets $HH --hh-neighbor 1 \
    --chunk "$CHUNK" --n-trials 2 --seeds 0 1 \
    --out-json "results/w11-new-r$((ctx/1024)).json" 2>&1
  emit "$tag" "results/w11-new-r$((ctx/1024)).json"
  echo "===${tag}_DONE==="
  echo "===PPL${ctx}_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --T "$ctx" --chunk "$CHUNK" --window 512 --n-samples 2 \
    --methods bugslash --ranks 128 256 --hh-budgets $HH --hh-neighbor 1 --no-ruler \
    --out-json "results/w11-new-ppl$((ctx/1024)).json" 2>&1
  emit "PPL$((ctx/1024))" "results/w11-new-ppl$((ctx/1024)).json"
  echo "===PPL${ctx}_DONE==="
}

firm(){ # firm <tag> <ctx>
  local tag="$1" ctx="$2"
  echo "===${tag}_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --context-lens "$ctx" --tasks niah_multikey niah_multivalue vt \
    --methods bugslash bugevict ea --ranks 32 --hh-budgets 256 --hh-neighbor 1 \
    --evict-keeps $EVICT --chunk "$CHUNK" --n-trials 2 --seeds 0 1 \
    --out-json "results/w11-firm-r$((ctx/1024)).json" 2>&1
  emit "$tag" "results/w11-firm-r$((ctx/1024)).json"
  echo "===${tag}_DONE==="
}

ppl(){ # ppl <ctx> -- standalone ppl block (hedge: new32's RULER host is slow and
  # its inline ppl runs last; a separate fast pod lands the Q2 ppl data early)
  local ctx="$1"
  echo "===PPLONLY${ctx}_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --T "$ctx" --chunk "$CHUNK" --window 512 --n-samples 2 \
    --methods bugslash --ranks 128 256 --hh-budgets $HH --hh-neighbor 1 --no-ruler \
    --out-json "results/w11-new-ppl$((ctx/1024)).json" 2>&1
  emit "PPL$((ctx/1024))" "results/w11-new-ppl$((ctx/1024)).json"
  echo "===PPLONLY${ctx}_DONE==="
}

R256_HH="${R256_HH:-1024}"
r256(){ # r256 <tag> <ctx> -- r256 retrieval: one hh arm (R256_HH), 3 HARD tasks
  # (needle saturates for the bugS family), n=2x2.
  local tag="$1" ctx="$2"
  echo "===${tag}_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --context-lens "$ctx" --tasks niah_multikey niah_multivalue vt \
    --methods bugslash --ranks 256 --hh-budgets $R256_HH --hh-neighbor 1 \
    --chunk "$CHUNK" --n-trials 2 --seeds 0 1 \
    --out-json "results/w11-r256-h${R256_HH}-r$((ctx/1024)).json" 2>&1
  emit "$tag" "results/w11-r256-h${R256_HH}-r$((ctx/1024)).json"
  echo "===${tag}_DONE==="
}

drop(){ # drop <tag> <ctx> -- Week-12 T1: the H1 ablation. Headline bugSdrop
  # (select-and-discard: withholding identical to bugS, pool invisible to
  # attention) FIRST so it lands even if the pod dies; then the bug-r128
  # gap-fill (its 32K hard-task cells are unmeasured). Hard tasks only.
  local tag="$1" ctx="$2"
  echo "===${tag}_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --context-lens "$ctx" --tasks niah_multikey niah_multivalue vt \
    --methods bugslash --ranks 128 --hh-budgets 1024 --hh-neighbor 1 --hh-discard \
    --chunk "$CHUNK" --n-trials 2 --seeds 0 1 \
    --out-json "results/w12-drop-r$((ctx/1024)).json" 2>&1
  echo "===${tag}_DONE==="
  echo "===${tag}_GAPFILL_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --context-lens "$ctx" --tasks niah_multikey niah_multivalue vt \
    --methods bug --ranks 128 --chunk "$CHUNK" --n-trials 1 --seeds 0 1 \
    --out-json "results/w12-bugr128-r$((ctx/1024)).json" 2>&1
  echo "===${tag}_GAPFILL_DONE==="
}

r192(){ # r192 <tag> <ctx> -- Week-12 T2: how narrow is the r128 sweet spot?
  # RULER hard tasks at ctx, then ppl at BOTH 16K and 32K (one invocation).
  local tag="$1" ctx="$2"
  echo "===${tag}_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --context-lens "$ctx" --tasks niah_multikey niah_multivalue vt \
    --methods bugslash --ranks 192 --hh-budgets 1024 --hh-neighbor 1 \
    --chunk "$CHUNK" --n-trials 2 --seeds 0 1 \
    --out-json "results/w12-r192-r$((ctx/1024)).json" 2>&1
  echo "===${tag}_DONE==="
  echo "===R192PPL_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --T 16384 32768 --chunk "$CHUNK" --window 512 --n-samples 2 \
    --methods bugslash --ranks 192 --hh-budgets 1024 --hh-neighbor 1 --no-ruler \
    --out-json "results/w12-r192-ppl.json" 2>&1
  echo "===R192PPL_DONE==="
}

mk64(){ # Week-12 T3: the 64K prediction test (needs an 80GB card -- 48GB OOMs
  # at 64K, docs/week5.md). bugS-r32-h256 + ea control interleaved in ONE
  # invocation per task block (a per-trial exception SKIP-logs, so an ea OOM
  # cannot kill the bugS rows). mk at n=2x2 (the prediction), mv/vt at n=1x2.
  echo "===MK64_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --context-lens 65536 --tasks niah_multikey \
    --methods bugslash ea --ranks 32 --hh-budgets 256 --hh-neighbor 1 \
    --evict-keeps $EVICT --chunk "$CHUNK" --n-trials 2 --seeds 0 1 \
    --out-json "results/w12-mk64.json" 2>&1
  echo "===MK64_DONE==="
  echo "===MVVT64_BEGIN==="
  PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
    --context-lens 65536 --tasks niah_multivalue vt \
    --methods bugslash ea --ranks 32 --hh-budgets 256 --hh-neighbor 1 \
    --evict-keeps $EVICT --chunk "$CHUNK" --n-trials 1 --seeds 0 1 \
    --out-json "results/w12-mvvt64.json" 2>&1
  echo "===MVVT64_DONE==="
}

case "$MODE" in
  new16)   new NEW16 16384 ;;
  new32)   new NEW32 32768 ;;
  firm16)  firm FIRM16 16384 ;;
  firm32)  firm FIRM32 32768 ;;
  ppl32)   ppl 32768 ;;
  r256_16) r256 R256_16 16384 ;;
  r256_32) r256 R256_32 32768 ;;
  drop32)  drop DROP32 32768 ;;
  r192_32) r192 R192_32 32768 ;;
  mk64)    mk64 ;;
esac
echo "===ALL_DONE==="
