#!/usr/bin/env bash
# One-time setup on an AutoDL GPU instance (single RTX 4090/4090D). Run from the repo root:
#   bash scripts/autodl_setup.sh
# Uses data/rl if it was uploaded (scripts/sync_to_autodl.sh); otherwise rebuilds it.
# Optional: INSTALL_FLASH_ATTN=1 (slow source build; otherwise train with ATTN=sdpa),
#           INDEX_URL=<PyPI mirror> if the default mirror is unreliable.
# Safe to rerun after an interruption: finished steps are skipped or served from cache.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$HOME/autodl-tmp}
VENV=${VENV:-$WORK/venv}
MODELS=$WORK/models

# The system disk is small (30GB on AutoDL); keep every download cache on the data disk.
export PIP_CACHE_DIR=$WORK/.cache/pip
export MODELSCOPE_CACHE=$WORK/.cache/modelscope
export HF_HOME=$WORK/.cache/huggingface
mkdir -p "$PIP_CACHE_DIR" "$MODELSCOPE_CACHE" "$HF_HOME"

df -h / "$WORK"
FREE_GB=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
if (( FREE_GB < 35 )); then
  echo "Only ${FREE_GB}GB free on $WORK; venv (~10GB), models (~4GB) and checkpoints (~20GB) need about 35GB."
  exit 1
fi

# Every download here uses domestic mirrors (PyPI mirror, ModelScope, hf-mirror), so AutoDL's
# academic proxy is not needed; it stalled large mirror downloads, so it is opt-in only.
if [[ "${USE_NETWORK_TURBO:-0}" == "1" && -f /etc/network_turbo ]]; then
  source /etc/network_turbo
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
fi

echo "== 1/5 GPU and driver"
nvidia-smi
CUDA_MAX=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)
# verl 0.9.1 pins torch 2.11, whose PyPI wheels are built for CUDA 13.
if ! python3 -c "import sys; sys.exit(0 if float('$CUDA_MAX') >= 13.0 else 1)"; then
  echo "Driver supports CUDA $CUDA_MAX, but torch 2.11 needs CUDA >= 13.0."
  echo "Rent a host whose '最高支持CUDA版本' is >= 13.0 (driver >= 580) and rerun."
  exit 1
fi

echo "== 2/5 Python environment ($VENV)"
python3 -c "import sys; assert sys.version_info >= (3, 10), sys.version"
# Override with e.g. INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple if a mirror keeps dropping.
INDEX_URL=${INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}
export UV_CACHE_DIR=$WORK/.cache/uv UV_HTTP_TIMEOUT=300 UV_INDEX_URL=$INDEX_URL
retry() {  # rerun a flaky download step; caches keep what already finished
  local attempt=1
  until "$@"; do
    if (( attempt >= ${RETRIES:-4} )); then echo "failed after $attempt attempts: $*"; return 1; fi
    echo "attempt $attempt failed, retrying in 15s: $*"
    attempt=$((attempt + 1))
    sleep 15
  done
}
[[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
retry pip install --retries 10 --timeout 60 -i "$INDEX_URL" -U pip uv
# uv downloads in parallel and resolves in seconds; rerunning the script resumes from its cache.
if python -c "import verl, vllm, sys; sys.exit(0 if verl.__version__ == '0.9.1' else 1)" 2>/dev/null; then
  echo "verl 0.9.1 and vLLM already installed, skipping"
else
  retry uv pip install "verl[vllm]==0.9.1"
fi
# Freeze what verl chose so the extras below cannot move torch/vLLM/transformers.
PINS=$WORK/verl_pins.txt
uv pip freeze | grep -iE '^(verl|torch|torchvision|torchaudio|vllm|transformers|numpy|ray)==' > "$PINS"
cat "$PINS"
# This repo without its own pins, so verl decides torch/numpy/transformers versions.
uv pip install --no-deps -e "$ROOT"
retry uv pip install -c "$PINS" jsonschema httpx pydantic-settings python-dotenv pyarrow pandas sentence-transformers modelscope tensorboard matplotlib pytest
if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  MAX_JOBS=${MAX_JOBS:-8} pip install flash-attn==2.8.3 --no-build-isolation \
    || echo "flash-attn build failed: train with ATTN=sdpa"
fi

echo "== 3/5 Models"
mkdir -p "$MODELS"
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir "$MODELS/Qwen2.5-1.5B-Instruct"
modelscope download --model BAAI/bge-small-en-v1.5 --local_dir "$MODELS/bge-small-en-v1.5"

echo "== 4/5 Data"
cd "$ROOT"
if [[ -s data/rl/train.jsonl && -s data/rl/corpus.jsonl && -s data/rl/metadata.json ]]; then
  # Uploaded with scripts/sync_to_autodl.sh: reuse the exact splits built locally.
  echo "data/rl already present (seed $(python -c "import json; print(json.load(open('data/rl/metadata.json'))['seed'])")), skipping download"
else
  HF_BASE=${HF_BASE:-https://hf-mirror.com} bash scripts/rl_download_raw.sh
  python scripts/rl_prepare_data.py
fi
python scripts/rl_make_verl_data.py
python scripts/rl_build_dense.py --model "$MODELS/bge-small-en-v1.5" --device cuda

echo "== 5/5 Check"
python - <<'PY'
import torch, verl, vllm, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("verl", verl.__version__, "vllm", vllm.__version__, "transformers", transformers.__version__)
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__, "-> ATTN=flash_attention_2")
except ImportError:
    print("flash_attn not installed -> train with ATTN=sdpa")
PY
python -m pytest -q tests/test_rl_env.py
echo "Setup done. Activate with: source $VENV/bin/activate"
