#!/usr/bin/env bash
# Stage 1+ — bring up the drishti rollout server (MuJoCo + MolmoAct2 + UQ) on a rented Prime Intellect
# 4090 pod. Run this ON the pod (after `prime pods ssh`), or adapt the prime CLI block to provision first.
#
# Budget note: a single RTX 4090 is ~$0.32/hr on Prime Intellect. MolmoAct2-SO100_101 inference is ~12 GiB
# bf16, so a 24GB 4090 is plenty. Tear the pod down (`prime pods terminate`) the moment you're done.
set -euo pipefail

# --- 0. (from your laptop) provision a 4090 and ssh in -----------------------------------------------
#   prime availability list --gpu-type RTX4090
#   prime pods create --name drishti-vla --gpu-type RTX4090 --gpu-count 1 --disk 100 --image cuda-12.4
#   prime pods ssh drishti-vla
# Then run the rest of this script on the pod.

REPO_URL="${REPO_URL:-https://github.com/Advaitic/drishti.git}"
BRANCH="${BRANCH:-rl-corrector}"
PORT="${PORT:-8100}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"          # headless GL on the pod (NOT the macOS default)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# --- 1. system deps for headless EGL rendering ------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y && sudo apt-get install -y libegl1 libgl1 libglib2.0-0 git
fi

# --- 2. python env (uv) + deps ----------------------------------------------------------------------
curl -LsSf https://astral.sh/uv/install.sh | sh || true
export PATH="$HOME/.local/bin:$PATH"

[ -d drishti ] || git clone "$REPO_URL" drishti
cd drishti && git fetch --all && git checkout "$BRANCH"

uv venv --python 3.11 .venv-pod
# torch with CUDA wheels comes from the pod image or the cu124 index; transformers pulls MolmoAct2's
# custom code via trust_remote_code at load time.
uv pip install --python .venv-pod/bin/python \
  "torch" "transformers>=4.44" accelerate safetensors \
  mujoco numpy pillow sqlalchemy "fastapi" "uvicorn[standard]" httpx

# --- 3. pre-pull the checkpoints (fail fast on a bad token, not at first request) --------------------
.venv-pod/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("allenai/MolmoAct2-SO100_101")      # ~12-22 GiB
snapshot_download("google/siglip-base-patch16-224")   # RND embedding source
print("checkpoints cached")
PY

# --- 4. sanity: model loads + one action, arm moves the RIGHT way (R4 joint-convention gate) --------
PYTHONPATH=. MUJOCO_GL=egl .venv-pod/bin/python -m drishti.drivers.rollout_record_molmoact2 --smoke

# --- 5. launch the rollout server (Stage-1 baseline uses ace-only; switch to calib after calibration)-
echo "launching rollout server on :$PORT (uncertainty=${UNCERTAINTY:-ace})"
PYTHONPATH=. MUJOCO_GL=egl UNCERTAINTY="${UNCERTAINTY:-ace}" \
  .venv-pod/bin/uvicorn "drishti.sim.rollout_server:build_default_app" --factory --host 0.0.0.0 --port "$PORT"

# From your laptop, tunnel it for the verifiers env's rollout_url:
#   prime pods forward drishti-vla $PORT:$PORT      # or use the pod's public URL
