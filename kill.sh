#!/usr/bin/env bash
# ============================================================
# kill.sh — Killa il training in corso e verifica
# Uso: bash kill.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.training.pid"

echo "========================================"
echo "  KILL TRAINING"
echo "========================================"

KILLED=0

# Kill dal PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[INFO] Termino gruppo processi PID $PID..."
        # Killa l'intero gruppo (train.sh + python train.py figlio)
        PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ' || echo "")
        if [ -n "$PGID" ] && [ "$PGID" != "0" ]; then
            kill -- -"$PGID" 2>/dev/null || true
        fi
        kill "$PID" 2>/dev/null || true
        KILLED=1
    else
        echo "[INFO] PID $PID già terminato"
    fi
    rm -f "$PID_FILE"
fi

# Kill qualsiasi train.py residuo
RESIDUI=$(pgrep -f "unitree_rl_mjlab/scripts/train.py" 2>/dev/null || true)
if [ -n "$RESIDUI" ]; then
    echo "[INFO] Processo residuo trovato, termino..."
    pkill -f "unitree_rl_mjlab/scripts/train.py" 2>/dev/null || true
    KILLED=1
fi

if [ $KILLED -eq 0 ]; then
    echo "[--] Nessun training da killare"
fi

# Verifica finale
sleep 2
STILL_ALIVE=$(pgrep -f "unitree_rl_mjlab/scripts/train.py" 2>/dev/null || true)
if [ -n "$STILL_ALIVE" ]; then
    echo "[WARN] Processo ancora vivo, uso SIGKILL..."
    pkill -9 -f "unitree_rl_mjlab/scripts/train.py" 2>/dev/null || true
    sleep 1
fi

FINAL=$(pgrep -f "unitree_rl_mjlab/scripts/train.py" 2>/dev/null || true)
if [ -z "$FINAL" ]; then
    echo "[OK] Training terminato con successo"
else
    echo "[ERRORE] Non riesco a killare: $FINAL"
fi

echo "========================================"
