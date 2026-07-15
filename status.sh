#!/usr/bin/env bash
# ============================================================
# status.sh — Mostra stato del training in corso
# Uso: bash status.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/unitree_rl_mjlab/logs"

echo "========================================"
echo "  TRAINING STATUS"
echo "========================================"

# --- PID: un file per run (.training_<run>.pid; il glob copre anche il
# legacy .training.pid). Più run possono essere vivi in parallelo. ---
FOUND=0
for PID_FILE in "$SCRIPT_DIR"/.training*.pid; do
    [ -f "$PID_FILE" ] || continue
    FOUND=1
    RUN=$(basename "$PID_FILE" .pid); RUN=${RUN#.training}; RUN=${RUN#_}
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        ELAPSED=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ' || echo "?")
        echo "[OK] Training IN CORSO — run: ${RUN:-<default>}  PID: $PID  durata: $ELAPSED"
    else
        echo "[STOP] Training TERMINATO — run: ${RUN:-<default>} (PID $PID non più attivo)"
        rm -f "$PID_FILE"
    fi
done
if [ "$FOUND" -eq 0 ]; then
    # Cerca comunque un processo train.py attivo
    PROCS=$(pgrep -a -f "unitree_rl_mjlab/scripts/train.py" 2>/dev/null || true)
    if [ -n "$PROCS" ]; then
        echo "[WARN] Training attivo ma senza PID file:"
        echo "$PROCS"
    else
        echo "[--] Nessun training in corso"
    fi
fi

echo ""

# --- GPU ---
if command -v nvidia-smi &>/dev/null; then
    echo "--- GPU ---"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
               --format=csv,noheader,nounits | \
        awk -F',' '{printf "  GPU%s (%s): %s%% util | %s/%s MiB\n",$1,$2,$3,$4,$5}'
    echo ""
fi

# --- Ultimo log ---
LATEST_LOG=$(ls -t "$LOG_DIR"/train_*.log 2>/dev/null | head -1 || true)
if [ -n "$LATEST_LOG" ]; then
    echo "--- Ultimo log: $(basename "$LATEST_LOG") ---"
    tail -25 "$LATEST_LOG"
    echo ""
    echo "Log live:  tail -f $LATEST_LOG"
else
    echo "Nessun log trovato in $LOG_DIR"
fi

echo ""
echo "Killa con:   bash kill.sh [run_name]   (senza argomenti killa TUTTI i run)"
echo "========================================"
