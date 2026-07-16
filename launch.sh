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
#
# Video automatici (auto_video.sh viene lanciato in parallelo, ON di default):
#   ogni --video-interval iterazioni (default 500) renderizza il checkpoint con
#   la STESSA forza di assistenza del training (force-matched) e beta=1.0, e
#   salva in logs/rsl_rl/g1_getup/<run>/videos/play/model_<N>_force<F>N_beta<B>.mp4.
#   Flag (consumati da launch.sh, NON passati a train.py):
#     --no-video            disabilita i video automatici
#     --video-interval N    intervallo iterazioni (default 500)
#     --video-device D      device per il render, es. cpu / cuda:1
#
# Run CONCORRENTI: i PID file sono per-run (.training_<run>.pid /
# .autovideo_<run>.pid), quindi si possono lanciare più run con NOMI DIVERSI
# in parallelo, ognuno sulla propria GPU (--gpu N è consumato da train.sh):
#   bash launch.sh Unitree-G1-GetUp run_a --agent.max-iterations=2000
#   bash launch.sh Unitree-G1-GetUp run_b --gpu 1 --video-device cuda:1 ...
# kill.sh <run_name> killa un run specifico; kill.sh senza argomenti killa tutto.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/unitree_rl_mjlab/logs"
TASK_LOGROOT="$SCRIPT_DIR/unitree_rl_mjlab/logs/rsl_rl/g1_getup"
# PID file PER-RUN (.training_<run>.pid): consente run CONCORRENTI su GPU diverse
# (es. scratch su GPU0 + resume su GPU1). Definiti dopo che RUN_NAME è noto.

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

# Separa i flag video (consumati qui) dagli extra args per train.py.
AUTO_VIDEO=1
VIDEO_INTERVAL=500
VIDEO_DEVICE=""
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-video) AUTO_VIDEO=0; shift;;
        --video-interval) VIDEO_INTERVAL="$2"; shift 2;;
        --video-device) VIDEO_DEVICE="$2"; shift 2;;
        *) EXTRA_ARGS+=("$1"); shift;;
    esac
done
LOG_FILE="$LOG_DIR/train_${RUN_NAME}.log"
PID_FILE="$SCRIPT_DIR/.training_${RUN_NAME}.pid"
VIDEO_PID_FILE="$SCRIPT_DIR/.autovideo_${RUN_NAME}.pid"

mkdir -p "$LOG_DIR"

# Se c'è già un training CON QUESTO RUN NAME, avvisa (run con nomi diversi
# possono coesistere — usare --gpu per metterli su GPU diverse).
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[WARN] Training '$RUN_NAME' già in corso (PID $OLD_PID). Killalo prima con: bash kill.sh $RUN_NAME"
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
echo "Killa con:     bash kill.sh $RUN_NAME   (bash kill.sh senza argomenti killa TUTTI i run)"
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

# ------------------------------------------------------------
# Video automatici: lancia auto_video.sh in parallelo al training.
# Aspetta che train.py crei la run dir (2026-..._<RUN_NAME>), poi avvia il
# watcher che renderizza ogni VIDEO_INTERVAL iterazioni con la stessa forza
# di assistenza del training (force-matched) e beta=1.0.
# ------------------------------------------------------------
if [ "$AUTO_VIDEO" -eq 1 ]; then
    # Se c'è un watcher residuo di una run precedente, terminalo.
    if [ -f "$VIDEO_PID_FILE" ]; then
        OLD_VPID=$(cat "$VIDEO_PID_FILE")
        if kill -0 "$OLD_VPID" 2>/dev/null; then
            echo "[INFO] Termino auto_video precedente (PID $OLD_VPID)"
            kill "$OLD_VPID" 2>/dev/null || true
        fi
        rm -f "$VIDEO_PID_FILE"
    fi
    VIDEO_LOG="$LOG_DIR/videos_${RUN_NAME}.log"
    DEV_ARGS=()
    [ -n "$VIDEO_DEVICE" ] && DEV_ARGS=(--device "$VIDEO_DEVICE")
    nohup bash -c "
        # Legge la run dir ESATTA dalla riga che train.py stampa nel SUO log
        # ('[INFO] Logging experiment in directory: <path>'), invece di indovinarla
        # con un glob su \$RUN_NAME. Il glob poteva agganciarsi a una dir STALE di
        # un tentativo precedente/fallito con lo stesso nome (es. crash su GPU
        # piena prima di creare il process reale) e restarci bloccato per sempre,
        # perché il loop si fermava al PRIMO match e non ricontrollava mai più.
        d=''
        for i in \$(seq 1 180); do
            line=\$(grep -m1 '\[INFO\] Logging experiment in directory:' '$LOG_FILE' 2>/dev/null)
            if [ -n \"\$line\" ]; then
                cand=\"\${line#*Logging experiment in directory: }\"
                cand=\"\${cand%/}\"
                if [ -d \"\$cand\" ]; then
                    d=\"\$cand\"
                    break
                fi
            fi
            sleep 5
        done
        if [ -z \"\$d\" ]; then
            echo '[auto_video-launcher] run dir mai apparsa nel log di training, esco.'
            exit 1
        fi
        echo \"[auto_video-launcher] run dir (dal log di training): \$d\"
        exec bash '$SCRIPT_DIR/unitree_rl_mjlab/scripts/auto_video.sh' \"\$(basename \"\$d\")\" \
            --interval '$VIDEO_INTERVAL' ${DEV_ARGS[*]:-}
    " > "$VIDEO_LOG" 2>&1 &
    VIDEO_PID=$!
    echo "$VIDEO_PID" > "$VIDEO_PID_FILE"
    echo "[OK] Video automatici attivi (PID $VIDEO_PID) — ogni $VIDEO_INTERVAL iter, force-matched"
    echo "     Log video:  tail -f $VIDEO_LOG"
    echo "     Output:     <run_dir>/videos/play/model_<N>_force<F>N_beta<B>.mp4"
else
    echo "[--] Video automatici disabilitati (--no-video)"
fi
