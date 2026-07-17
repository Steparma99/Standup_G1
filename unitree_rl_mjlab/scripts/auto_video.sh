#!/usr/bin/env bash
#
# auto_video.sh — automatically render an eval video (scripts/play.py) for each new
# checkpoint at a fixed iteration interval, WHILE a training run is in progress.
#
# It watches a run's checkpoint dir and, whenever a `model_<N>.pt` with N a multiple of
# --interval appears (checkpoints are saved every 100 iters, so 500/1000/... all exist),
# runs play.py with your standard eval characteristics and saves the clip to
#   logs/rsl_rl/g1_getup/<run>/videos/play/model_<N>_force<F>N_beta<B>.mp4
# (the filename records the exact assist force and beta the clip was rendered with)
#
# Decoupled from training: if a render fails (e.g. transient GPU OOM) it just logs and
# retries on the next poll; it never touches the training process.
#
# USAGE (run alongside training, no tmux -> nohup):
#   cd ~/Standup/Standup_G1/unitree_rl_mjlab
#   nohup bash scripts/auto_video.sh phase2_posture_fix > videos_auto.log 2>&1 &
#
#   # first positional arg = run dir name, OR any substring of it, OR "latest".
#   # e.g. "phase2_posture_fix" resolves to the newest 2026-..._phase2_posture_fix dir.
#
# OPTIONS (all optional):
#   --interval N     render every N iterations              (default 500)
#   --beta B         --eval-beta passed to play.py          (default 0.97)
#   --device D       torch device, e.g. cpu / cuda:0 / cuda:1
#                    (default: play.py picks cuda:0; use cpu to avoid GPU contention
#                     with training at the cost of a slower render)
#   --num-envs N     (default 16)      --extra N   max-extra-envs   (default 8)
#   --vlen N         video-length frames (default 500)
#   --width N --height N   video resolution (default 1920x1080)
#   --cam-distance F (default 6)   --cam-elevation F (default -15)
#   --poll SECONDS   how often to check for new checkpoints (default 60)
#   --once           render the single latest eligible checkpoint and exit
#   --env-name NAME  conda env to run play.py in (default unitree_rl_cuda)
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

TASK="Unitree-G1-GetUp"
LOGROOT="logs/rsl_rl/g1_getup"

# ---- defaults ----
INTERVAL=500
BETA=1.0           # fallback when the run has no logged beta (curriculum off)
MATCH_BETA=1       # 1 = pin each checkpoint's own training beta at eval
DEVICE=""
MATCH_FORCE=1      # 1 = pin each checkpoint's own training assist force at eval
FORCE_OVERRIDE=""  # if set, pin this constant force (N) for every checkpoint
NUM_ENVS=16
EXTRA=8
VLEN=500
VW=1920
VH=1080
CAMD=6
CAME=-15
POLL=60
ONCE=0
DRYRUN=0
ENV_NAME="unitree_rl_cuda"

if [ $# -lt 1 ]; then
  echo "usage: bash scripts/auto_video.sh <run|substring|latest> [options]  (see header)"
  exit 1
fi
RUN_QUERY="$1"; shift || true

while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2;;
    --beta) BETA="$2"; MATCH_BETA=0; shift 2;;
    --no-match-beta) MATCH_BETA=0; shift;;
    --assist-force) FORCE_OVERRIDE="$2"; MATCH_FORCE=0; shift 2;;
    --no-match-force) MATCH_FORCE=0; shift;;
    --device) DEVICE="$2"; shift 2;;
    --num-envs) NUM_ENVS="$2"; shift 2;;
    --extra) EXTRA="$2"; shift 2;;
    --vlen) VLEN="$2"; shift 2;;
    --width) VW="$2"; shift 2;;
    --height) VH="$2"; shift 2;;
    --cam-distance) CAMD="$2"; shift 2;;
    --cam-elevation) CAME="$2"; shift 2;;
    --poll) POLL="$2"; shift 2;;
    --once) ONCE=1; shift;;
    --dry-run) DRYRUN=1; shift;;
    --env-name) ENV_NAME="$2"; shift 2;;
    *) echo "[auto_video] unknown option: $1"; exit 1;;
  esac
done

# ---- resolve the run dir ----
resolve_run() {
  if [ -d "$LOGROOT/$RUN_QUERY" ]; then
    echo "$LOGROOT/$RUN_QUERY"; return
  fi
  if [ "$RUN_QUERY" = "latest" ]; then
    ls -1dt "$LOGROOT"/*/ 2>/dev/null | head -1 | sed 's:/*$::'; return
  fi
  # newest dir whose name contains the query substring
  ls -1dt "$LOGROOT"/*"$RUN_QUERY"*/ 2>/dev/null | head -1 | sed 's:/*$::'
}
RUNDIR="$(resolve_run)"
if [ -z "$RUNDIR" ] || [ ! -d "$RUNDIR" ]; then
  echo "[auto_video] could not resolve a run dir for query '$RUN_QUERY' under $LOGROOT"
  exit 1
fi
# Final videos land in videos/play/ (same dir play.py renders into), named with the
# eval parameters so each clip is self-describing:
#   model_<N>_force<F>N_beta<B>.mp4
PLAYDIR="$RUNDIR/videos/play"
OUTDIR="$PLAYDIR"
mkdir -p "$PLAYDIR"

echo "[auto_video] repo     : $REPO"
echo "[auto_video] run dir  : $RUNDIR"
echo "[auto_video] interval : every $INTERVAL iters  | beta=$BETA  device=${DEVICE:-<play default cuda:0>}"
echo "[auto_video] output   : $OUTDIR/model_<N>_force<F>N_beta<B>.mp4"
echo "[auto_video] poll     : ${POLL}s   once=$ONCE"

render_ckpt() {
  local ckpt="$1" n="$2"
  echo "[auto_video] $(date '+%H:%M:%S') rendering model_${n} ..."
  local dev_arg=()
  [ -n "$DEVICE" ] && dev_arg=(--device "$DEVICE")
  # Resolve the assist force to pin at eval so the video matches this checkpoint's
  # training conditions. --assist-force overrides; else --match-force looks up the
  # run's logged assistance_force at iteration n; else 0 (unaided).
  local force=""
  if [ -n "$FORCE_OVERRIDE" ]; then
    force="$FORCE_OVERRIDE"
  elif [ "$MATCH_FORCE" -eq 1 ]; then
    force="$(conda run -n "$ENV_NAME" python scripts/assist_force_at.py "$RUNDIR" "$n" 2>/dev/null | tail -1)"
  fi
  case "$force" in ''|*[!0-9.]*) force="";; esac  # keep only a clean number
  local force_note="unaided (0 N)"
  [ -n "$force" ] && force_note="${force} N"
  # Resolve the beta to pin at eval. With the beta curriculum enabled, each env's
  # action authority anneals during training, so the video must use the run's
  # logged beta at iteration n (same lookup mechanism as the assist force);
  # falls back to $BETA when the metric is absent (curriculum disabled) or the
  # lookup fails.
  local beta="$BETA"
  if [ "$MATCH_BETA" -eq 1 ]; then
    local matched_beta
    matched_beta="$(conda run -n "$ENV_NAME" python scripts/assist_force_at.py "$RUNDIR" "$n" \
      --tag Episode_Metrics/curriculum/beta_rescaler 2>/dev/null | tail -1)"
    case "$matched_beta" in ''|0|*[!0-9.]*) :;; *) beta="$matched_beta";; esac
  fi
  echo "[auto_video]   eval params: assist_force=$force_note  beta=$beta"
  # Filename records the exact eval parameters this clip was rendered with.
  local out="$OUTDIR/model_${n}_force${force:-0}N_beta${beta}.mp4"
  if [ "$DRYRUN" -eq 1 ]; then
    echo "[auto_video] DRY-RUN would execute:"
    echo "  GETUP_EVAL_ASSIST_FORCE=${force:-<unset>} MUJOCO_GL=egl conda run -n $ENV_NAME --no-capture-output python scripts/play.py $TASK \\"
    echo "    --checkpoint-file $ckpt --eval-beta $beta --num-envs $NUM_ENVS --max-extra-envs $EXTRA \\"
    echo "    --cam-distance $CAMD --cam-elevation $CAME --video=True --video-length $VLEN \\"
    echo "    --video-width $VW --video-height $VH ${DEVICE:+--device $DEVICE}"
    echo "  -> then move newest $PLAYDIR/*.mp4 to $out"
    return 0
  fi
  GETUP_EVAL_ASSIST_FORCE="${force}" MUJOCO_GL=egl conda run -n "$ENV_NAME" --no-capture-output \
    python scripts/play.py "$TASK" \
      --checkpoint-file "$ckpt" \
      --eval-beta "$beta" \
      --num-envs "$NUM_ENVS" --max-extra-envs "$EXTRA" \
      --cam-distance "$CAMD" --cam-elevation "$CAME" \
      --video=True --video-length "$VLEN" --video-width "$VW" --video-height "$VH" \
      "${dev_arg[@]}"
  local rc=$?
  if [ $rc -ne 0 ] && [ -z "$DEVICE" ]; then
    # Default device is the training GPU — a render there can fail every poll
    # (OOM/contention with the 4096-env training), silently producing NO videos
    # for the whole run. Fall back to a CPU render: slower but reliable.
    echo "[auto_video] model_${n} failed on GPU (rc=$rc) — retrying on cpu"
    GETUP_EVAL_ASSIST_FORCE="${force}" MUJOCO_GL=egl conda run -n "$ENV_NAME" --no-capture-output \
      python scripts/play.py "$TASK" \
        --checkpoint-file "$ckpt" \
        --eval-beta "$beta" \
        --num-envs "$NUM_ENVS" --max-extra-envs "$EXTRA" \
        --cam-distance "$CAMD" --cam-elevation "$CAME" \
        --video=True --video-length "$VLEN" --video-width "$VW" --video-height "$VH" \
        --device cpu
    rc=$?
  fi
  if [ $rc -ne 0 ]; then
    echo "[auto_video] FAILED model_${n} (rc=$rc) — will retry next poll"
    return 1
  fi
  # play.py writes videos/play/rl-video-step-0.mp4 (overwritten each run); grab the
  # newest rl-video-* there (NOT the model_* clips we already saved into the same
  # dir) and rename it to the parameter-stamped per-iteration name.
  local latest
  latest="$(ls -1t "$PLAYDIR"/rl-video-*.mp4 2>/dev/null | head -1)"
  if [ -n "$latest" ]; then
    mv -f "$latest" "$out"
    echo "[auto_video] saved $out"
  else
    echo "[auto_video] WARN: no mp4 found in $PLAYDIR after render of model_${n}"
    return 1
  fi
}

while true; do
  # collect checkpoints ascending by iteration number (sort by the numeric model_<N>,
  # not the full path — paths contain underscores so a naive -t_ sort is wrong).
  mapfile -t ckpts < <(
    for f in "$RUNDIR"/model_*.pt; do
      [ -e "$f" ] || continue
      b="$(basename "$f" .pt)"; echo "${b#model_} $f"
    done | sort -n | awk '{print $2}'
  )
  latest_n=-1; latest_ckpt=""
  for ckpt in "${ckpts[@]}"; do
    [ -e "$ckpt" ] || continue
    base="$(basename "$ckpt" .pt)"; n="${base#model_}"
    case "$n" in ''|*[!0-9]*) continue;; esac
    if [ "$n" -gt 0 ] && [ $((n % INTERVAL)) -eq 0 ]; then
      if [ "$ONCE" -eq 1 ]; then
        [ "$n" -gt "$latest_n" ] && { latest_n="$n"; latest_ckpt="$ckpt"; }
      else
        # already rendered? (match both the new parameter-stamped name and the
        # legacy plain model_<N>.mp4 / old videos/auto location)
        if ls "$OUTDIR/model_${n}_"*.mp4 >/dev/null 2>&1 \
           || [ -e "$OUTDIR/model_${n}.mp4" ] \
           || [ -e "$RUNDIR/videos/auto/model_${n}.mp4" ]; then
          : # skip, clip already exists
        else
          render_ckpt "$ckpt" "$n"
        fi
      fi
    fi
  done
  if [ "$ONCE" -eq 1 ]; then
    [ -n "$latest_ckpt" ] && render_ckpt "$latest_ckpt" "$latest_n"
    echo "[auto_video] --once done."; exit 0
  fi
  sleep "$POLL"
done
