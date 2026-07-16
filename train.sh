#!/usr/bin/env bash
# ============================================================
# Training wrapper — reads hardware_config.yaml and sets the
# correct environment variables before launching training.
#
# Usage:
#   bash train.sh <task> [extra args]
#
# Examples:
#   bash train.sh Unitree-G1-Flat
#   bash train.sh Unitree-G1-Flat --env.scene.num-envs=128
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/hardware_config.yaml"
MJLAB_DIR="$SCRIPT_DIR/unitree_rl_mjlab"
EXTRA_ARGS=()

if [ $# -lt 1 ]; then
  echo "Usage: bash train.sh <task> [--gpu N] [extra args]"
  echo ""
  echo "  --gpu N    Run on a specific GPU index (e.g. --gpu 1)"
  echo ""
  echo "Available tasks (run from mjlab dir):"
  python "$MJLAB_DIR/scripts/list_envs.py" 2>/dev/null || true
  exit 1
fi

# --- Pre-parse our own --gpu N flag (consumed here, not forwarded to Python) ---
GPU_OVERRIDE=""
FILTERED_ARGS=()
i=1
while [ $i -le $# ]; do
  arg="${!i}"
  if [[ "$arg" == "--gpu" ]]; then
    i=$((i+1))
    GPU_OVERRIDE="${!i}"
  elif [[ "$arg" == --gpu=* ]]; then
    GPU_OVERRIDE="${arg#--gpu=}"
  else
    FILTERED_ARGS+=("$arg")
  fi
  i=$((i+1))
done
set -- "${FILTERED_ARGS[@]}"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "[ERROR] hardware_config.yaml not found at: $CONFIG_FILE"
  exit 1
fi

# --- Parse hardware_config.yaml ---
_yaml_top_level_get() {
  local key="$1"
  grep -E "^[[:space:]]*$key[[:space:]]*:" "$CONFIG_FILE" | head -1 \
    | sed 's/.*:\s*["\x27]\?\([^"#\x27]*\)["\x27]\?.*/\1/' | xargs
}

_yaml_get() {
  # Usage: _yaml_get <section> <key>
  grep -A 20 "^$1:" "$CONFIG_FILE" | grep -E "^\s+$2\s*:" | head -1 \
    | sed 's/.*:\s*["\x27]\?\([^"#\x27]*\)["\x27]\?.*/\1/' | xargs
}

BACKEND=$(_yaml_top_level_get "backend")
echo "[INFO] Backend: $BACKEND"

case "$BACKEND" in
  cpu)
    # CPU / AMD: disable CUDA, use osmesa for offscreen rendering
    export CUDA_VISIBLE_DEVICES=""
    export MUJOCO_GL=$(_yaml_get "cpu" "mujoco_gl")
    export OMP_NUM_THREADS=$(_yaml_get "cpu" "num_threads")
    NUM_ENVS=$(_yaml_get "cpu" "num_envs")

    echo "[INFO] MUJOCO_GL=$MUJOCO_GL  |  OMP_NUM_THREADS=$OMP_NUM_THREADS  |  num_envs=$NUM_ENVS"

    # Inject num_envs if not already provided by the caller
    if [[ "$*" != *"--env.scene.num-envs"* ]]; then
      EXTRA_ARGS+=("--env.scene.num-envs=$NUM_ENVS")
    fi
    EXTRA_ARGS+=("--gpu-ids" "None")
    ;;

  cuda)
    # NVIDIA GPU: set GPU IDs, EGL rendering
    GPU_IDS=$(_yaml_get "cuda" "gpu_ids")
    # Convert YAML list [0, 1] → "0,1"
    GPU_IDS_CLEAN=$(echo "$GPU_IDS" | tr -d '[] ' | tr ',' ',')
    # --gpu N override takes precedence over hardware_config.yaml
    if [ -n "$GPU_OVERRIDE" ]; then
      GPU_IDS_CLEAN="$GPU_OVERRIDE"
    fi
    export CUDA_VISIBLE_DEVICES="$GPU_IDS_CLEAN"
    export MUJOCO_GL=$(_yaml_get "cuda" "mujoco_gl")
    NUM_ENVS=$(_yaml_get "cuda" "num_envs")

    echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  |  MUJOCO_GL=$MUJOCO_GL  |  num_envs=$NUM_ENVS"

    if [[ "$*" != *"--env.scene.num-envs"* ]]; then
      EXTRA_ARGS+=("--env.scene.num-envs=$NUM_ENVS")
    fi
    if [[ "$*" != *"--gpu-ids"* ]]; then
      # --gpu-ids indexes into the list of devices CUDA_VISIBLE_DEVICES already
      # exposed to this process, NOT physical GPU IDs — with CUDA_VISIBLE_DEVICES=1,
      # that GPU becomes the process's device 0. So this must always be a
      # sequential range [0], [0,1], ... sized to the number of selected GPUs,
      # never the physical IDs in GPU_IDS_CLEAN themselves (e.g. "[1]" would index
      # past a single-GPU masked view and crash with IndexError).
      NUM_SELECTED=$(echo "$GPU_IDS_CLEAN" | awk -F',' '{print NF}')
      SEQ_IDS=$(seq 0 $((NUM_SELECTED - 1)) | paste -sd, -)
      EXTRA_ARGS+=("--gpu-ids" "[$SEQ_IDS]")
    fi
    ;;

  *)
    echo "[ERROR] Unknown backend '$BACKEND' in hardware_config.yaml"
    exit 1
    ;;
esac

# --- Launch training ---
cd "$MJLAB_DIR"
echo "[INFO] Launching: python scripts/train.py $* ${EXTRA_ARGS[*]}"
echo ""
# -u: unbuffered stdout/stderr. launch.sh redirects this to a log FILE (not a
# tty), so Python defaults to full block buffering — early prints (e.g. the
# "[INFO] Logging experiment in directory: ..." line launch.sh's auto-video
# watcher greps for) can sit unflushed for a long time even though training
# is running fine, making the watcher time out waiting for a line that IS
# being printed, just not yet written to disk. -u also makes `tail -f` on the
# training log real-time instead of arriving in delayed bursts.
python -u scripts/train.py "$@" "${EXTRA_ARGS[@]}"
