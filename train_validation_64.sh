#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs_validation"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE="$LOG_DIR/train_validation_64_${TIMESTAMP}.log"

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
MUJOCO_BACKEND="${MUJOCO_GL:-egl}"

nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  MUJOCO_GL="$MUJOCO_BACKEND" \
  python unitree_rl_mjlab/scripts/train.py Unitree-G1-GetUp \
  --agent.max-iterations=2500 \
  --env.scene.num-envs=64 \
  "$@" \
  > "$LOG_FILE" 2>&1 &

PID=$!

echo "Started validation training"
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
