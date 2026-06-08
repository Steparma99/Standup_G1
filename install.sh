#!/usr/bin/env bash
# ============================================================
# Install script — reads hardware_config.yaml and sets up
# the correct conda environment for cpu or cuda backend.
# Usage: bash install.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/hardware_config.yaml"
MJLAB_DIR="$SCRIPT_DIR/unitree_rl_mjlab"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "[ERROR] hardware_config.yaml not found at: $CONFIG_FILE"
  exit 1
fi

# Parse backend from YAML (no external parser needed)
BACKEND=$(grep -E '^\s*backend\s*:' "$CONFIG_FILE" | sed 's/.*:\s*["\x27]\?\([a-z]*\)["\x27]\?.*/\1/')

echo "[INFO] Detected backend: $BACKEND"

case "$BACKEND" in
  cpu)
    ENV_YAML="$SCRIPT_DIR/unitree_rl_mjlab/conda_envs/env_cpu.yaml"
    ENV_NAME="unitree_rl_cpu"
    APT_DEPS="libyaml-cpp-dev libboost-all-dev libeigen3-dev libosmesa6-dev"
    ;;
  cuda)
    ENV_YAML="$SCRIPT_DIR/unitree_rl_mjlab/conda_envs/env_cuda.yaml"
    ENV_NAME="unitree_rl_cuda"
    APT_DEPS="libyaml-cpp-dev libboost-all-dev libeigen3-dev"
    ;;
  *)
    echo "[ERROR] Unknown backend '$BACKEND'. Set 'backend' to 'cpu' or 'cuda' in hardware_config.yaml"
    exit 1
    ;;
esac

# --- System packages ---
echo "[INFO] Installing system packages: $APT_DEPS"
sudo apt-get update -qq
sudo apt-get install -y $APT_DEPS

# --- Conda environment ---
if conda env list | grep -q "^$ENV_NAME "; then
  echo "[INFO] Conda env '$ENV_NAME' already exists."
  read -r -p "         Recreate from scratch? [y/N] " REPLY
  if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    echo "[INFO] Removing existing env '$ENV_NAME' ..."
    conda env remove -n "$ENV_NAME" -y
    echo "[INFO] Creating conda env '$ENV_NAME' from $ENV_YAML ..."
    conda env create -f "$ENV_YAML"
  else
    echo "[INFO] Skipping env creation — using existing env."
  fi
else
  echo "[INFO] Creating conda env '$ENV_NAME' from $ENV_YAML ..."
  conda env create -f "$ENV_YAML"
fi

# --- Install the mjlab package in editable mode (path-agnostic) ---
echo "[INFO] Installing unitree_rl_mjlab[$BACKEND] in editable mode ..."
conda run -n "$ENV_NAME" pip install -e "${MJLAB_DIR}[${BACKEND}]"

echo ""
echo "============================================================"
echo "  Installation complete!"
echo "  Activate with:  conda activate $ENV_NAME"
echo "  Then train with: bash train.sh <task> [extra args]"
echo "============================================================"
