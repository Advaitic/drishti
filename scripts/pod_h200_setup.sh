#!/usr/bin/env bash
# One-shot H200 setup for the rl-corrector GRPO run (colocated single-GPU).
# Installs EVERYTHING up front: system deps, uv, prime-rl (+ our verifiers env), the lerobot rollout-server
# venv, and pre-pulls the models. Runs each section, logs a STATUS line, and CONTINUES (no set -e) so ONE
# run's log shows every success/failure at once — no iterative install-guessing.
#
#   bash ~/drishti/scripts/pod_h200_setup.sh 2>&1 | tee ~/logs/setup.log
#
# Layout produced:
#   ~/prime-rl/.venv   -> prime-rl (vLLM Qwen3-VL corrector + LoRA trainer + orchestrator/verifiers)
#   ~/lerobot/.venv    -> rollout server (finetuned MolmoAct2 + MuJoCo + SMD/RND/ACE); drishti via PYTHONPATH
#   ~/ft_ckpt          -> finetuned MolmoAct2 checkpoint (uploaded separately)

log(){ echo -e "\n========== $* =========="; }
status(){ echo ">>> STATUS $1 = rc=$2"; }
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:$PATH"
cd ~

log "A. system deps (EGL for headless MuJoCo, build tools)"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential ninja-build git curl pkg-config \
    libegl1 libgl1 libglib2.0-0 libglvnd0 libosmesa6 libgl1-mesa-glx 2>&1 | tail -3
status apt $?

log "B. install uv"
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
uv --version; status uv $?

log "C. prime-rl (manual setup — clone + submodules + uv sync, avoids nested-clone bug)"
if [ ! -d ~/prime-rl/.git ]; then
  git clone --depth 1 https://github.com/PrimeIntellect-ai/prime-rl.git ~/prime-rl
fi
cd ~/prime-rl
git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config 2>&1 | tail -3
uv sync --all-extras 2>&1 | tail -6
status prime_rl_sync $?
echo "-- prime-rl entrypoints --"; ls .venv/bin/ | grep -iE "^rl$|trainer|inference|orchestrator" | tr '\n' ' '; echo

log "D. install our verifiers env into prime-rl's venv"
cd ~/prime-rl
uv pip install -e ~/drishti/environments/vla-corrector 2>&1 | tail -4
status vla_corrector_env $?
uv pip show verifiers 2>/dev/null | grep -iE "^Version"

log "E. lerobot rollout-server venv (Python 3.12, molmoact2 + dataset extras)"
if [ ! -d ~/lerobot/.git ]; then
  git clone --depth 1 https://github.com/huggingface/lerobot.git ~/lerobot
fi
cd ~/lerobot
uv sync --python 3.12 --extra molmoact2 --extra dataset 2>&1 | tail -6
status lerobot_sync $?

log "F. extra deps into the lerobot venv (rollout server + UQ + SigLIP tokenizer)"
cd ~/lerobot
uv pip install "transformers==5.5.4" sentencepiece protobuf einops \
    fastapi "uvicorn[standard]" httpx sqlalchemy mujoco pillow 2>&1 | tail -4
status lerobot_extras $?
echo "-- versions --"; .venv/bin/python -c "import torch,transformers,mujoco; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__,'mj',mujoco.__version__)" 2>&1 | tail -2

log "G. sanity: does the finetuned VLA + rollout server import in the lerobot venv?"
cd ~/drishti
PYTHONPATH=~/drishti MUJOCO_GL=egl ~/lerobot/.venv/bin/python -c "
import drishti.sim.rollout_server as rs
from drishti.policy_molmoact2_lerobot import load_molmoact2_lerobot
print('rollout_server + VLA adapter import OK')" 2>&1 | tail -6
status vla_import $?

log "H. pre-pull models (SigLIP for RND; Qwen3-VL-4B for the corrector)"
~/lerobot/.venv/bin/python -c "
from transformers import AutoModel, AutoProcessor
AutoModel.from_pretrained('google/siglip-base-patch16-224')
AutoProcessor.from_pretrained('google/siglip-base-patch16-224')
print('siglip cached')" 2>&1 | tail -3
status siglip_pull $?
~/prime-rl/.venv/bin/python -c "
from huggingface_hub import snapshot_download
p=snapshot_download('Qwen/Qwen3-VL-4B-Instruct')
print('qwen3vl cached at', p)" 2>&1 | tail -3
status qwen_pull $?

log "DONE — review the STATUS lines above. Checkpoint at ~/ft_ckpt:"
du -sh ~/ft_ckpt 2>/dev/null
echo "GPU:"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
