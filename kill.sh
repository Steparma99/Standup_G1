#!/usr/bin/env bash
# ============================================================
# kill.sh — Killa training + auto_video
#
# Uso:
#   bash kill.sh              # killa TUTTI i run (training + watcher video)
#   bash kill.sh <run_name>   # killa SOLO quel run (PID file .training_<run>.pid)
#
# I PID file sono per-run (creati da launch.sh): .training_<run>.pid e
# .autovideo_<run>.pid. Il glob .training*.pid copre anche i file legacy
# (.training.pid / .autovideo.pid) di run lanciati prima di questo schema.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_QUERY="${1:-}"

echo "========================================"
echo "  KILL TRAINING${RUN_QUERY:+ (run: $RUN_QUERY)}"
echo "========================================"

# Killa il gruppo di processi puntato da un PID file, poi rimuove il file.
_kill_pidfile() {
    local f="$1"
    [ -f "$f" ] || return 0
    local PID
    PID=$(cat "$f")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[INFO] Termino gruppo processi PID $PID ($(basename "$f"))..."
        local PGID
        PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ' || echo "")
        if [ -n "$PGID" ] && [ "$PGID" != "0" ]; then
            kill -- -"$PGID" 2>/dev/null || true
        fi
        kill "$PID" 2>/dev/null || true
    else
        echo "[INFO] PID $PID ($(basename "$f")) già terminato"
    fi
    rm -f "$f"
}

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
if [ -n "$RUN_QUERY" ]; then
    _kill_pidfile "$SCRIPT_DIR/.training_${RUN_QUERY}.pid"
    # Residui: il comando python contiene --agent.run-name=<run>
    pkill -f "agent.run-name=${RUN_QUERY}" 2>/dev/null || true
else
    for f in "$SCRIPT_DIR"/.training*.pid; do _kill_pidfile "$f"; done
    pkill -f "unitree_rl_mjlab/scripts/train.py" 2>/dev/null || true
fi

sleep 2
if [ -n "$RUN_QUERY" ]; then
    KILL_PATTERN="agent.run-name=${RUN_QUERY}"
else
    KILL_PATTERN="unitree_rl_mjlab/scripts/train.py"
fi
STILL_ALIVE=$(pgrep -f "$KILL_PATTERN" 2>/dev/null || true)
if [ -n "$STILL_ALIVE" ]; then
    echo "[WARN] Processo ancora vivo, uso SIGKILL..."
    pkill -9 -f "$KILL_PATTERN" 2>/dev/null || true
    sleep 1
fi
FINAL=$(pgrep -f "$KILL_PATTERN" 2>/dev/null || true)
if [ -z "$FINAL" ]; then
    echo "[OK] Training terminato"
else
    echo "[ERRORE] Non riesco a killare: $FINAL"
fi

# ------------------------------------------------------------
# Watcher video (auto_video.sh)
# ------------------------------------------------------------
if [ -n "$RUN_QUERY" ]; then
    _kill_pidfile "$SCRIPT_DIR/.autovideo_${RUN_QUERY}.pid"
    # Residui: auto_video.sh riceve il run name come primo argomento
    pkill -f "auto_video.sh.*${RUN_QUERY}" 2>/dev/null || true
    V_ALIVE=$(pgrep -f "auto_video.sh.*${RUN_QUERY}" 2>/dev/null || true)
else
    for f in "$SCRIPT_DIR"/.autovideo*.pid; do _kill_pidfile "$f"; done
    pkill -f "scripts/auto_video.sh" 2>/dev/null || true
    V_ALIVE=$(pgrep -f "scripts/auto_video.sh" 2>/dev/null || true)
fi
if [ -z "$V_ALIVE" ]; then
    echo "[OK] Auto-video terminato"
else
    echo "[WARN] auto_video ancora vivo: $V_ALIVE (SIGKILL...)"
    pkill -9 -f "scripts/auto_video.sh" 2>/dev/null || true
fi

echo "========================================"
