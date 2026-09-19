#!/bin/bash
# Pod bootstrap (onstart-batch, SSH-less) -- the ONLY file uploaded via --onstart
# (kept tiny; the 16KB cap bit an earlier driver). It pins the run to an exact SHA,
# stamps a reproducibility header INTO the harvested log (SHA + nvidia-smi + python/
# torch/CUDA/transformers), defines emit() (base64 JSON fold, so out-JSONs survive
# `vastai logs` truncation), then hands off to the committed entrypoint
# `scripts/pod.py run --pod "$POD"` -- which, being cloned at $SHA, is itself SHA-pinned.
# The pod config (arms, tasks, model, image) is configs/pods/$POD.yaml.
#
# Launch (this is exactly what `scripts/pod.py launch --pod <name> --offer <id>` builds):
#   SHA=$(git rev-parse HEAD)   # the exact commit to evaluate; push it first
#   vastai create instance $OFFER \
#     --image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel --disk 80 \
#     --env "-e POD=w18_g1 -e SHA=$SHA -e MODEL=<hf-id> -e DTYPE=bfloat16" \
#     --onstart scripts/pod/boot.sh --label kvdlra-w18_g1
# Harvest with `scripts/pod.py harvest --pod <name>` (scripts/pod/watchdog.sh does it
# unattended and destroys the instance on ALL_DONE). Does NOT self-destruct.
#
# EVERY marker this script prints carries the pod name: `===<MARKER>_${POD}...`. The
# watchdog matches `===(ALL_DONE|RUN_FAILED|<boot failure>)_<pod>` and destroys the
# instance on any of them, so a marker without the suffix is a pod that fails and then
# bills until the credit floor.
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

export POD="${POD:-w18_g1}"
export MODEL="${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
export DTYPE="${DTYPE:-bfloat16}"
export SHA="${SHA:-week7}"          # exact commit; falls back to the branch tip

echo "===POD_${POD}==="

cd /root || exit 1
for attempt in 1 2 3 4 5; do
  rm -rf kvdlra
  git clone https://github.com/hkrishna42/kvdlra.git 2>&1 | tail -3
  [ -d kvdlra/scripts ] && break
  echo "===CLONE_RETRY_${attempt}==="; sleep 5
done
cd kvdlra || { echo "===CLONE_FAILED_${POD}==="; exit 1; }
# SHA pin: check out the exact commit and FAIL LOUD if it isn't what was asked for.
git checkout -q "$SHA" >/dev/null 2>&1 || { echo "===CHECKOUT_FAILED_${POD}_${SHA}==="; exit 1; }  # no pipe: the exit code is the guard
RUN_SHA="$(git rev-parse HEAD)"
echo "===RUN_SHA_${RUN_SHA}==="

pip install -q hf_transfer hf_xet ninja numpy scipy matplotlib "kvpress==0.5.1" 2>&1 | tail -5
pip install -q 'transformers==5.8.0' 'datasets==2.21.0' "optimum-quanto>=0.2.7" 'hqq==0.2.8.post1' 'omegaconf>=2.3' 2>&1 | tail -5
echo "===DEPS_DONE==="
# Fail loud if the quant baseline backend is missing (else the quant arms silently SKIP).
python -c "import optimum.quanto" 2>/dev/null && echo "===QUANTO_OK===" || echo "===QUANTO_MISSING_${POD}==="
python -c "import hqq" 2>/dev/null && echo "===HQQ_OK===" || echo "===HQQ_MISSING_${POD}==="

# Reproducibility header, INSIDE the log block (the evidentiary chain). Everything a
# camera-ready compute-disclosure needs: commit, card, driver/CUDA, and library set.
# `scripts/pod.py run` writes the same set to results/$POD/env.txt as name==version;
# `pod.py harvest` rebuilds that file FROM these lines, since the pod-side one is
# destroyed with the instance. A package dropped from this block is `unrecorded` there.
echo "===ENV_BEGIN==="
echo "run_sha=${RUN_SHA}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python - <<'PY' || echo "===DEPS_FAILED_${POD}==="
import sys, importlib.metadata as md, torch, transformers, kvpress  # noqa: F401
def _ver(pkg):
    try: return md.version(pkg)
    except Exception: return "?"
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__} cuda_build={torch.version.cuda} cuda_avail={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__} kvpress={_ver('kvpress')} optimum-quanto={_ver('optimum-quanto')} hqq={_ver('hqq')}")
print(f"triton={_ver('triton')} omegaconf={_ver('omegaconf')} datasets={_ver('datasets')} numpy={_ver('numpy')} scipy={_ver('scipy')}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)}")
PY
# Fail loud on a bad $MODEL before any long harness call.
python -c "from transformers import AutoTokenizer as T; T.from_pretrained('$MODEL'); print('===MODEL_OK===')" \
  || { echo "===MODEL_FAILED_${POD}_${MODEL}==="; exit 1; }
echo "===ENV_END==="

# emit <marker> <json-path>: base64-fold a result JSON through the log so `vastai logs`
# truncation (~500 chars/line) can't lose it; scrape with an awk flip-flop over the
# BEGIN/END markers + `tr -d ' \r\n' | base64 -d`. Exported so a child process sees it.
emit() {
  echo "===${1}_RESULT_BEGIN==="
  base64 -w0 "$2" 2>/dev/null | fold -w 400
  echo ""
  echo "===${1}_RESULT_END==="
}
export -f emit

# Hand off to the committed, SHA-pinned entrypoint. Every knob is in the pod YAML.
# ALL_DONE is what the watchdog destroys on, so it must NOT be printed after a failed
# run -- an unconditional echo turns a crash into a clean-looking pod. RUN_FAILED is
# the same signal for the watchdog (destroy: no idle billing) and a visible status in
# the harvested manifest.
python scripts/pod.py run --pod "$POD" 2>&1 || { echo "===RUN_FAILED_${POD}_${RUN_SHA}==="; exit 1; }
echo "===ALL_DONE_${POD}_${RUN_SHA}==="
