#!/usr/bin/env bash
# 2-GPU vision run — the PROVEN combined path. Two things only:
#   1. our rollout server (MolmoAct2 + MuJoCo + UQ) on GPU1, :8100
#   2. `uv run rl @ config` — ONE command; the launcher wires inference(GPU0) + trainer(GPU1) + orchestrator
#      + the trainer<->orchestrator run-registration handshake that the 1-GPU manual split couldn't do.
#   bash ~/drishti/scripts/pod_run_2gpu.sh
set -uo pipefail
UV=~/.local/bin/uv; LR=~/lerobot/.venv/bin; DRISHTI=~/drishti; PRL=~/prime-rl
LOGS=~/logs; mkdir -p "$LOGS"
export WANDB_MODE=disabled HF_HUB_ENABLE_HF_TRANSFER=0

echo "===== clean stale procs ====="; pkill -f rollout_server 2>/dev/null; pkill -f "uv run rl" 2>/dev/null; sleep 3

echo "===== 1. rollout server (GPU1, :8100) — MolmoAct2 + sim + calibrated UQ ====="
cd "$DRISHTI"
DRISHTI_LEROBOT_CKPT=~/ft_ckpt UNCERTAINTY=calib MUJOCO_GL=egl PYTHONPATH="$DRISHTI" CUDA_VISIBLE_DEVICES=1 \
  nohup "$LR/uvicorn" drishti.sim.rollout_server:build_default_app --factory --host 0.0.0.0 --port 8100 \
  > "$LOGS/rollout.log" 2>&1 & echo "  rollout pid=$!"
for i in $(seq 1 72); do
  curl -sf localhost:8100/healthz >/dev/null 2>&1 && { echo "  rollout READY"; break; }
  sleep 5; [ $i -eq 72 ] && { echo "  !! rollout not ready"; tail -20 "$LOGS/rollout.log"; exit 1; }
done

echo "===== 2. install env into prime-rl venv ====="
cd "$PRL"; $UV pip install -e "$DRISHTI/environments/vla-corrector" 2>&1 | tail -1

echo "===== 3. uv run rl (combined: inference GPU0 + trainer GPU1 + orchestrator) ====="
cd "$PRL"
nohup $UV run rl @ "$DRISHTI/configs/rl/grpo_qwen3vl_4b.toml" > "$LOGS/rl.log" 2>&1 & echo "  rl pid=$!"
echo "RUN_LAUNCHED — combined-rl logs land in $PRL/outputs/logs/{orchestrator,trainer,inference}.log"
