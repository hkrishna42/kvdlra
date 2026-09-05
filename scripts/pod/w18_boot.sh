#!/bin/bash
# Week-18 pod bootstrap (onstart-batch, SSH-less) -- the ONLY file uploaded via
# --onstart (kept tiny; the 16KB cap bit w16.sh). It pins the run to an exact SHA,
# stamps a reproducibility header INTO the harvested log (SHA + nvidia-smi + python/
# torch/CUDA/transformers), defines emit() (base64 JSON fold, so out-JSONs survive
# `vastai logs` truncation), then sources the committed MODE driver scripts/pod/w18.sh
# -- which is unbounded in size and, being cloned at $SHA, is itself SHA-pinned.
#
# Launch:
#   SHA=$(git rev-parse HEAD)   # the exact commit to evaluate; push it first
#   uvx vastai create instance $OFFER \
#     --image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --disk 80 \
#     --env "-e MODE=g1 -e MODEL=Qwen/Qwen2.5-7B-Instruct -e TAG=qwen -e SHA=$SHA" \
#     --onstart scripts/pod/w18_boot.sh --label kvdlra-w18
# Harvest ^===/^\[trial/^\[niah/^\[vt/^\[pplw and the *_RESULT_BEGIN/END folds from
# `vastai logs <id> --tail 20000`; DESTROY from the launcher. Does NOT self-destruct.
set -x
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_HUB_DISABLE_XET=1
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false
export PIP_BREAK_SYSTEM_PACKAGES=1
# optimum-quanto builds a CUDA kernel (quanto_cuda.so) on first quant use via torch's
# cpp_extension, which needs the CUDA toolkit (nvcc) + CUDA_HOME. Requires the pytorch
# *-devel* image (the -runtime image has no nvcc). Harmless when unset/unused.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

export MODE="${MODE:-g1}"
export MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export TAG="${TAG:-qwen}"
export CHUNK="${CHUNK:-4096}"
export DTYPE="${DTYPE:-bfloat16}"
export SHA="${SHA:-week7}"          # exact commit; falls back to the branch tip
export DRIVER="${DRIVER:-scripts/pod/w18.sh}"

echo "===MODE_${MODE}_${TAG}==="

cd /root || exit 1
for attempt in 1 2 3 4 5; do
  rm -rf kvdlra
  git clone https://github.com/hkrishna42/kvdlra.git 2>&1 | tail -3
  [ -d kvdlra/scripts ] && break
  echo "===CLONE_RETRY_${attempt}==="; sleep 5
done
cd kvdlra || { echo "===CLONE_FAILED==="; exit 1; }
# SHA pin: check out the exact commit and FAIL LOUD if it isn't what was asked for.
git checkout -q "$SHA" 2>&1 | tail -2 || { echo "===CHECKOUT_FAILED_${SHA}==="; exit 1; }
RUN_SHA="$(git rev-parse HEAD)"
echo "===RUN_SHA_${RUN_SHA}==="

pip install -q hf_transfer hf_xet numpy scipy matplotlib "kvpress==0.5.1" 2>&1 | tail -5
pip install -q 'transformers==5.8.0' 'datasets==2.21.0' "optimum-quanto>=0.2.7" 'hqq==0.2.8.post1' 2>&1 | tail -5
echo "===DEPS_DONE==="
# Fail loud if the quant baseline backend is missing (else G1's quant arms silently SKIP).
python -c "import optimum.quanto" 2>/dev/null && echo "===QUANTO_OK===" || echo "===QUANTO_MISSING==="
python -c "import hqq" 2>/dev/null && echo "===HQQ_OK===" || echo "===HQQ_MISSING==="

# Reproducibility header, INSIDE the log block (Week-18 evidentiary chain). Everything
# a camera-ready compute-disclosure needs: commit, card, driver/CUDA, and library set.
echo "===ENV_BEGIN==="
echo "run_sha=${RUN_SHA}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python - <<'PY' || echo "===DEPS_FAILED==="
import sys, importlib.metadata as md, torch, transformers, kvpress  # noqa: F401
def _ver(pkg):
    try: return md.version(pkg)
    except Exception: return "?"
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__} cuda_build={torch.version.cuda} cuda_avail={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__} kvpress={_ver('kvpress')}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)}")
PY
# Fail loud on a bad $MODEL before any long harness call (restored from w10_gpu.sh).
python -c "from transformers import AutoTokenizer as T; T.from_pretrained('$MODEL'); print('===MODEL_OK===')" \
  || { echo "===MODEL_FAILED_${MODEL}==="; exit 1; }
echo "===ENV_END==="

# emit <marker> <json-path>: base64-fold a result JSON through the log so `vastai logs`
# truncation (~500 chars/line) can't lose it; scrape with scripts/pod/scrape_w10.sh's
# awk flip-flop + `tr -d ' \r\n' | base64 -d`. Exported so the sourced driver sees it.
emit() {
  echo "===${1}_RESULT_BEGIN==="
  base64 -w0 "$2" 2>/dev/null | fold -w 400
  echo ""
  echo "===${1}_RESULT_END==="
}
export -f emit

# Hand off to the committed, SHA-pinned MODE driver (sourced so emit()/env persist).
# shellcheck disable=SC1090
source "$DRIVER"
echo "===ALL_DONE_${MODE}_${TAG}_${RUN_SHA}==="
