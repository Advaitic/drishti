#!/usr/bin/env bash
# Finetune SmolVLA on the scripted sim pick-place dataset.
#
# PROVEN locally via a 2-step run on MPS (finite loss, cameras wired, no NaN) before any
# paid GPU. The two non-obvious bits this encodes:
#   1. --policy.push_to_hub=false  (else lerobot demands a hub repo_id and aborts)
#   2. --rename_map  maps our up/side/wrist -> the camera1/2/3 keys smolvla_base expects
#
# Usage:
#   ./scripts/finetune.sh [device] [steps] [batch_size]
#   ./scripts/finetune.sh cuda 8000 32      # GPU run (Vast)
#   ./scripts/finetune.sh mps  200  4       # local smoke
#
# Dataset must be present at --dataset.root (scp the 21MB data/lerobot/so101_sim_pickplace
# dir to the box), or pushed to the HF hub and referenced by --dataset.repo_id alone.
set -euo pipefail

DEVICE="${1:-cuda}"
STEPS="${2:-8000}"
BS="${3:-32}"
OUT="${OUT:-outputs/smolvla_pickplace}"

RENAME='{"observation.images.up":"observation.images.camera1","observation.images.side":"observation.images.camera2","observation.images.wrist":"observation.images.camera3"}'

lerobot-train \
  --dataset.repo_id=drishti/so101_sim_pickplace \
  --dataset.root=data/lerobot/so101_sim_pickplace \
  --dataset.video_backend=pyav \
  --policy.path=lerobot/smolvla_base \
  --policy.device="${DEVICE}" --policy.push_to_hub=false \
  --rename_map="${RENAME}" \
  --batch_size="${BS}" --steps="${STEPS}" \
  --save_freq=1500 --log_freq=50 --num_workers=8 \
  --output_dir="${OUT}" \
  --wandb.enable=false
