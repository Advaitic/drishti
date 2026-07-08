#!/usr/bin/env bash
# Colocated single-GPU (H200) RL run: 4 processes on GPU 0, coordinating via a shared output dir.
#   rollout server (MolmoAct2 VLA + MuJoCo + UQ)  :8100   [lerobot venv]
#   inference     (vLLM Qwen3-VL-4B corrector)    :8000   [prime-rl venv, CUDA_VISIBLE_DEVICES=0]
#   trainer       (FSDP + LoRA)                            [prime-rl venv, CUDA_VISIBLE_DEVICES=0]
#   orchestrator  (drives rollouts, weight sync)           [prime-rl venv]
# Weight broadcast + rollout link are filesystem-based, so separate processes coordinate via --output-dir.
#
#   bash ~/drishti/scripts/pod_run.sh smoke   # max_steps=2, tiny batch — validate the pipeline
#   bash ~/drishti/scripts/pod_run.sh run     # the real run (config defaults)
set -uo pipefail
MODE="${1:-smoke}"
UV=~/.local/bin/uv
LR=~/lerobot/.venv/bin
DRISHTI=~/drishti
PRL=~/prime-rl
OUT=$PRL/outputs/drishti-$MODE
LOGS=~/logs; mkdir -p "$LOGS"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ "$MODE" = "smoke" ]; then OVR="--max-steps 2"; OVR_ORCH="--max-steps 2 --batch-size 4 --group-size 4"; else OVR=""; OVR_ORCH=""; fi

echo "===== 0. clean stale procs ====="
pkill -f "rollout_server" 2>/dev/null; pkill -f "uv run inference" 2>/dev/null
pkill -f "uv run trainer" 2>/dev/null; pkill -f "uv run orchestrator" 2>/dev/null; sleep 3

echo "===== 1. rollout server (GPU 0, :8100) ====="
cd "$DRISHTI"
DRISHTI_LEROBOT_CKPT=~/ft_ckpt UNCERTAINTY=calib MUJOCO_GL=egl PYTHONPATH="$DRISHTI" CUDA_VISIBLE_DEVICES=0 \
  nohup "$LR/uvicorn" drishti.sim.rollout_server:build_default_app --factory --host 0.0.0.0 --port 8100 \
  > "$LOGS/rollout.log" 2>&1 & echo "  rollout pid=$!"
echo "  waiting for /healthz (VLA load ~2-3 min)..."
for i in $(seq 1 60); do
  if curl -sf localhost:8100/healthz >/dev/null 2>&1; then echo "  rollout READY"; break; fi
  sleep 5; [ $i -eq 60 ] && { echo "  !! rollout server not ready — see $LOGS/rollout.log"; tail -15 "$LOGS/rollout.log"; exit 1; }
done

echo "===== 2. inference server (GPU 0, :8000) ====="
cd "$PRL"
CUDA_VISIBLE_DEVICES=0 nohup $UV run inference @ "$DRISHTI/configs/rl/h200_infer.toml" --output-dir "$OUT" \
  > "$LOGS/inference.log" 2>&1 & echo "  inference pid=$!"
echo "  waiting for vLLM /health (model load ~3-5 min)..."
for i in $(seq 1 90); do
  if curl -sf localhost:8000/health >/dev/null 2>&1; then echo "  inference READY"; break; fi
  sleep 5; [ $i -eq 90 ] && { echo "  !! inference not ready — see $LOGS/inference.log"; tail -20 "$LOGS/inference.log"; exit 1; }
done

echo "===== 3. trainer (GPU 0) — MUST launch under torchrun (sets RANK/WORLD_SIZE) ====="
cd "$PRL"
CUDA_VISIBLE_DEVICES=0 nohup $UV run torchrun --role=trainer --rdzv-endpoint=localhost:29501 \
  --rdzv-id="drishti-$MODE" --nproc-per-node=1 \
  -m prime_rl.trainer.rl.train @ "$DRISHTI/configs/rl/h200_train.toml" --output-dir "$OUT" $OVR \
  > "$LOGS/trainer.log" 2>&1 & echo "  trainer pid=$!"
sleep 8

echo "===== 4. orchestrator (drives the run) ====="
cd "$PRL"
$UV run orchestrator @ "$DRISHTI/configs/rl/h200_orch.toml" --output-dir "$OUT" $OVR_ORCH \
  2>&1 | tee "$LOGS/orchestrator.log"
echo "===== run finished (mode=$MODE). weights at $OUT/weights ====="
ls "$OUT/weights" 2>/dev/null
