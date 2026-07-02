#!/usr/bin/env bash
# Runs ON the pod (in the rsynced repo root). Installs deps, downloads MolmoAct2 + SigLIP, runs the R4
# smoke gate. Idempotent-ish; safe to re-run. Billed while running — keep it moving.
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export DEBIAN_FRONTEND=noninteractive
export MUJOCO_GL=egl                  # xet handles fast HF downloads now; hf_transfer not needed

echo "=== [1/5] apt deps for headless EGL ==="
(apt-get update -y && apt-get install -y libegl1 libgl1 libglib2.0-0 git curl >/dev/null 2>&1) \
  || (sudo apt-get update -y && sudo apt-get install -y libegl1 libgl1 libglib2.0-0 git curl) || true

echo "=== [2/5] uv + venv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11 .venv-pod
PY=.venv-pod/bin/python

echo "=== [3/5] python deps (Ai2's authoritative pins: allenai/molmoact2/pyproject.toml) ==="
uv pip install --python "$PY" "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
# transformers 4.57.x (Ai2's pin). einops/protobuf/sentencepiece are MolmoAct2 remote-code deps; the slow
# tokenizer (use_fast=False in policy_molmoact2) needs protobuf+sentencepiece.
uv pip install --python "$PY" "transformers>=4.57,<4.58" accelerate safetensors huggingface_hub einops protobuf sentencepiece \
  mujoco numpy pillow sqlalchemy fastapi "uvicorn[standard]" httpx
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-CUDA')"

echo "=== [4/5] download checkpoints (~22GB MolmoAct2 + SigLIP) ==="
"$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/MolmoAct2-SO100_101'); snapshot_download('google/siglip-base-patch16-224'); print('checkpoints cached')"

echo "=== [5/5] R4 smoke gate (does the gripper approach the cube? + save frames) ==="
# float32 (~26GiB) fits the L40's 46GB and skips the fragile bf16 modeling patch. Use bfloat16 on a 24GB 4090.
PYTHONPATH=. MUJOCO_GL=egl DRISHTI_VLA_DTYPE="${DRISHTI_VLA_DTYPE:-float32}" \
  "$PY" -m drishti.drivers.rollout_record_molmoact2 --smoke
echo "=== smoke done ==="
