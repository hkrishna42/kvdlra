#!/bin/bash
# Onstart batch for the Week-11 GOAL-B run: can BUG retrieve a needle at 32K, at
# <= the memory ExpectedAttention needs? Runs (1) the Phase-0 needle-surprise
# probe and (2) the SurpriseSLASH retrieval sweep + the bugEVICT attribution
# control + the EA bar, at 8B/32K (and 16K), niah_single then niah_multikey.
#
# Same pod recipe as w10_gpu.sh (SSH unusable -> git-clone the pushed branch,
# python -u, pinned deps, HF_HUB_DISABLE_XET, results base64-folded between
# markers). ALWAYS destroy the pod after. Uses existing keys (no rotation).
#
# Env: MODEL (default 8B), CTX (context lengths for the sweep), PROBE_CTX,
# RANK (BUG gist rank), HH (exact-tier sizes), NEIGHBOR (span window),
# EVICT (EA keep fractions = the bar), NTRIALS/SEEDS, CHUNK, DTYPE.
set -x
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false
export PIP_BREAK_SYSTEM_PACKAGES=1

MODEL="${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
CTX="${CTX:-32768}"
PROBE_CTX="${PROBE_CTX:-16384 32768}"
RANK="${RANK:-32}"
HH="${HH:-256 512 1024 2048}"
NEIGHBOR="${NEIGHBOR:-1}"
EVICT="${EVICT:-0.1}"          # EA bar: ea-k0.1 = 100% @ 0.10x at 8B/32K
RULER_TASKS="${RULER_TASKS:-niah_single niah_multikey}"
NTRIALS="${NTRIALS:-6}"
SEEDS="${SEEDS:-0 1}"
NSAMP="${NSAMP:-3}"
CHUNK="${CHUNK:-4096}"
DTYPE="${DTYPE:-bfloat16}"

cd /root || exit 1
git clone --depth 1 --branch week7 https://github.com/hkrishna42/kvdlra.git 2>&1 | tail -3
cd kvdlra || exit 1

pip install -q hf_transfer hf_xet numpy matplotlib "kvpress==0.5.1" 2>&1 | tail -5
pip install -q 'transformers==5.8.0' 'datasets==2.21.0' 2>&1 | tail -5
echo "===DEPS_DONE==="
python -c "import torch,transformers,kvpress; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)" || echo "===DEPS_FAILED==="
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL'); print('===MODEL_OK===')" 2>&1 | tail -3 || echo "===MODEL_FAILED==="

emit() {  # emit <marker> <json-path> : base64-fold so vast logs keep it
  echo "===${1}_RESULT_BEGIN==="
  base64 -w0 "$2" 2>/dev/null | fold -w 400
  echo ""
  echo "===${1}_RESULT_END==="
}

# (1) Phase-0 go/no-go probe: does surprise-selection land the needle in the exact
# tier at 8B/32K, and does +-1 span expansion capture the whole span? Cheap (no
# generation) -- the diagnostic that gates the retrieval verdict.
if [ "${RUN_PROBE:-1}" = "1" ]; then
for pc in $PROBE_CTX; do
for nb in 0 $NEIGHBOR; do
echo "===PROBE_BEGIN_c${pc}_n${nb}==="
PYTHONPATH=src python -u scripts/w11_probe.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
  --ctx "$pc" --rank "$RANK" --hh-budgets 64 128 256 512 1024 2048 --hh-neighbor "$nb" \
  --chunk "$CHUNK" --trial 0 --out-json "results/w11-probe-c${pc}-n${nb}.json" 2>&1
emit "PROBE_c${pc}_n${nb}" "results/w11-probe-c${pc}-n${nb}.json"
done
done
echo "===PROBE_DONE==="
fi

# (2) The retrieval frontier: SurpriseSLASH (bugslash) + the bugEVICT attribution
# control + the EA bar, at CTX. bugslash/bugevict REQUIRE --chunk>0 (single-shot
# bypasses the exact tier). full = the 100% reference.
if [ "${RUN_SLASH:-1}" = "1" ]; then
echo "===SLASH_BEGIN==="
PYTHONPATH=src python -u scripts/w10_ruler.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
  --context-lens $CTX --tasks $RULER_TASKS \
  --methods full bugslash bugevict ea --ranks "$RANK" \
  --hh-budgets $HH --hh-neighbor "$NEIGHBOR" --evict-keeps $EVICT \
  --chunk "$CHUNK" --n-trials "$NTRIALS" --seeds $SEEDS \
  --out-json results/w11-ruler-slash.json 2>&1
emit SLASH results/w11-ruler-slash.json
echo "===SLASH_DONE==="
fi

# (3) Joint quality axis: perplexity of the SAME arms (bugslash / bugEVICT / ea /
# full) at CTX -> the retrieval+quality frontier at matched memory. This is where
# BUG's low-rank gist earns its keep: bugEVICT (rank-1, no gist) may tie bugslash
# on needle retrieval but has no context quality, so its ppl should be far worse.
if [ "${RUN_PPL:-1}" = "1" ]; then
echo "===PPL_BEGIN==="
PYTHONPATH=src python -u scripts/w10_frontier.py --model "$MODEL" --device cuda --dtype "$DTYPE" \
  --T $CTX --chunk "$CHUNK" --window 512 --n-samples "$NSAMP" \
  --methods full bugslash bugevict ea --ranks "$RANK" \
  --hh-budgets $HH --hh-neighbor "$NEIGHBOR" --evict-keeps $EVICT \
  --no-ruler --out-json results/w11-ppl.json 2>&1
emit PPL results/w11-ppl.json
echo "===PPL_DONE==="
fi

echo "===ALL_DONE==="
