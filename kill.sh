#!/usr/bin/env bash
# ============================================================
# kill.sh — Killa il training in corso e verifica
# Uso: bash kill.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.training.pid"
VIDEO_PID_FILE="$SCRIPT_DIR/.autovideo.pid"

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

# ------------------------------------------------------------
# Killa anche il watcher dei video automatici (auto_video.sh)
# ------------------------------------------------------------
if [ -f "$VIDEO_PID_FILE" ]; then
    VPID=$(cat "$VIDEO_PID_FILE")
    if kill -0 "$VPID" 2>/dev/null; then
        echo "[INFO] Termino auto_video (PID $VPID)..."
        VPGID=$(ps -o pgid= -p "$VPID" 2>/dev/null | tr -d ' ' || echo "")
        if [ -n "$VPGID" ] && [ "$VPGID" != "0" ]; then
            kill -- -"$VPGID" 2>/dev/null || true
        fi
        kill "$VPID" 2>/dev/null || true
    fi
    rm -f "$VIDEO_PID_FILE"
fi
pkill -f "scripts/auto_video.sh" 2>/dev/null || true
V_ALIVE=$(pgrep -f "scripts/auto_video.sh" 2>/dev/null || true)
if [ -z "$V_ALIVE" ]; then
    echo "[OK] Auto-video terminato"
else
    echo "[WARN] auto_video ancora vivo: $V_ALIVE (SIGKILL...)"
    pkill -9 -f "scripts/auto_video.sh" 2>/dev/null || true
fi

echo "========================================"
