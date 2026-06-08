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
  echo "Usage: bash train.sh <task> [extra args]"
  echo ""
  echo "Available tasks (run from mjlab dir):"
  python "$MJLAB_DIR/scripts/list_envs.py" 2>/dev/null || true
  exit 1
fi

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
    export CUDA_VISIBLE_DEVICES="$GPU_IDS_CLEAN"
    export MUJOCO_GL=$(_yaml_get "cuda" "mujoco_gl")
    NUM_ENVS=$(_yaml_get "cuda" "num_envs")

    echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  |  MUJOCO_GL=$MUJOCO_GL  |  num_envs=$NUM_ENVS"

    if [[ "$*" != *"--env.scene.num-envs"* ]]; then
      EXTRA_ARGS+=("--env.scene.num-envs=$NUM_ENVS")
    fi
    if [[ "$*" != *"--gpu-ids"* ]]; then
      EXTRA_ARGS+=("--gpu-ids" "[$GPU_IDS_CLEAN]")
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
python scripts/train.py "$@" "${EXTRA_ARGS[@]}"
