#!/usr/bin/env bash
# ============================================================
# launch.sh — Lancia un training in background con log su file
#
# Uso:
#   bash launch.sh <task> [run_name] [extra args...]
#
# Esempi:
#   bash launch.sh Unitree-G1-GetUp
#   bash launch.sh Unitree-G1-GetUp getup_v2
#   bash launch.sh Unitree-G1-GetUp getup_v2 --agent.max-iterations=5000
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/unitree_rl_mjlab/logs"
PID_FILE="$SCRIPT_DIR/.training.pid"

if [ $# -lt 1 ]; then
    echo "Uso: bash launch.sh <task> [run_name] [extra args...]"
    echo "Esempio: bash launch.sh Unitree-G1-GetUp getup_v2"
    exit 1
fi

TASK="$1"
shift

RUN_NAME="${1:-run_$(date +%Y%m%d_%H%M%S)}"
# Controlla se il secondo argomento è un extra arg (inizia con --) o un nome
if [[ "$RUN_NAME" == --* ]]; then
    RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
else
    shift || true  # consuma run_name solo se non inizia con --
fi

EXTRA_ARGS=("$@")
LOG_FILE="$LOG_DIR/train_${RUN_NAME}.log"

mkdir -p "$LOG_DIR"

# Se c'è già un training, avvisa
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[WARN] Training già in corso (PID $OLD_PID). Killalo prima con: bash kill.sh"
        exit 1
    fi
fi

echo "[INFO] Task:     $TASK"
echo "[INFO] Run name: $RUN_NAME"
echo "[INFO] Log:      $LOG_FILE"
echo "[INFO] Extra:    ${EXTRA_ARGS[*]:-nessuno}"
echo ""

nohup bash "$SCRIPT_DIR/train.sh" "$TASK" \
    --agent.run-name="$RUN_NAME" \
    --agent.logger=wandb \
    "${EXTRA_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "[OK] Training lanciato — PID: $PID"
echo ""
echo "Monitora con:  bash status.sh"
echo "Log live con:  tail -f $LOG_FILE"
echo "Killa con:     bash kill.sh"
echo ""

# Aspetta 6 secondi e mostra le prime righe del log per confermare
sleep 6
if kill -0 "$PID" 2>/dev/null; then
    echo "[OK] Processo vivo dopo 6s"
    echo "--- Ultime righe log ---"
    tail -20 "$LOG_FILE"
else
    echo "[ERRORE] Processo morto. Log completo:"
    cat "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
