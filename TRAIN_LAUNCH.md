****  Making a fast training to visualize initial position and first steps ****

  cd ~/HumanUP/unitree_rl_mjlab && \
python scripts/train.py Unitree-G1-GetUp \
  --agent.max-iterations=500 \
  --agent.experiment-name=g1_getup \
  --agent.run-name=debug_local \
  --agent.logger=tensorboard \
  --env.scene.num-envs=1 \
  --gpu-ids None \
  --viewer True

Base training (Use defaults: 4096 env, 12000 iter, wandb)

bash launch.sh Unitree-G1-GetUp "insert run name"

Customize with specific GPU, envs, iterations:

bash launch.sh Unitree-G1-GetUp "insert run name" \
  --gpu 1 \
  --agent.max-iterations=2000 \
  --env.scene.num-envs=4096

  # --gpu N  → run on GPU N only (0, 1, 2 ...). Default from hardware_config.yaml.
  # --env.scene.num-envs overrides the yaml default
  # --agent.max-iterations overrides the default (12000)

  # VIDEO AUTOMATICI (ON di default): launch.sh lancia anche auto_video.sh in
  # parallelo — ogni 500 iter renderizza il checkpoint con la STESSA forza di
  # assistenza del training (force-matched, via GETUP_EVAL_ASSIST_FORCE) e
  # beta=1.0, salvando in <run_dir>/videos/auto/model_<N>.mp4.
  # --no-video            → disabilita
  # --video-interval N    → cambia intervallo (default 500)
  # --video-device cpu    → render su CPU (zero contesa GPU, più lento)
  # Log video: tail -f unitree_rl_mjlab/logs/videos_<run_name>.log
  # bash kill.sh termina training E video watcher.

Other options:

bash launch.sh Unitree-G1-GetUp "insert run name" \
  --gpu 0 \
  --agent.max-iterations=5000 \
  --env.scene.num-envs=1024 \
  --agent.logger=tensorboard

Or:

MUJOCO_GL=egl conda run -n unitree_rl_cuda --no-capture-output \
python scripts/train.py Unitree-G1-GetUp \
  --agent.max-iterations=300 \
  --agent.run-name=test_bootstrap_fix \
  --env.scene.num-envs=512

**Resume a training**

cd ~/Standup/Standup_G1/unitree_rl_mjlab && \
nohup python scripts/train.py Unitree-G1-GetUp \
  --agent.resume=True \
  --agent.load-run= ""\
  --agent.load-checkpoint="model_.pt" \/
  --agent.experiment-name=g1_getup \
  --agent.run-name=resume_after_fix \
  --agent.max-iterations=4000 \
  --env.scene.num-envs=8192 \
  > train_resume.log 2>&1 &
disown


For visualizing log
bash status.sh

Log in in real time
tail -f ~/Standup/Standup_G1/unitree_rl_mjlab/logs/train_<run_name>.log

Log from the beginning
less ~/Standup/Standup_G1/unitree_rl_mjlab/logs/train_<run_name>.log

Killing
bash kill.sh

Verifying manually
ps -ef | grep "train.py" | grep -v grep


**Sending on localhost a visualization test**

cd ~/Standup/Standup_G1/unitree_rl_mjlab && \
MUJOCO_GL=egl conda run -n unitree_rl_cuda --no-capture-output \
python scripts/play.py Unitree-G1-GetUp \
  --checkpoint-file logs/rsl_rl/g1_getup/<run-name>/model_<N>.pt \
  --eval-beta 0.97 \
  --viewer viser \
  --num-envs 1

cd ~/Standup/Standup_G1/unitree_rl_mjlab && \
MUJOCO_GL=egl conda run -n unitree_rl_cuda --no-capture-output \
python scripts/play.py Unitree-G1-GetUp \
  --checkpoint-file logs/rsl_rl/g1_getup/2026-06-24_11-16-19_Phase_3_no_jump/model_5497.pt \
  --eval-beta 0.97 \
  --num-envs 16 --max-extra-envs 8 \
  --cam-distance 6 --cam-elevation -15 \
  --video=True --video-length 500 --video-width 1920 --video-height 1080

MUJOCO_GL=egl conda run -n unitree_rl_cuda python scripts/play.py Unitree-G1-GetUp   --checkpoint-file logs/rsl_rl/g1_getup/2026-06-30_12-19-34/model_1999.pt   --num-envs 1   --video=True   --video-length 300



**Visualize Rewards on the server**:

conda run -n unitree_rl_cuda python scripts/print_rewards.py 2026-06-30_12-19-34



cd ~/Standup/Standup_G1/unitree_rl_mjlab/logs/rsl_rl/g1_getup
RUN="2026-06-24_09-34-38_Phase_3_no_jump/"

conda run --no-capture-output -n unitree_rl_cuda python3 - "$RUN" <<'EOF'
import sys, glob
from tensorboard.backend.event_processing import event_accumulator
run = sys.argv[1]
ev = glob.glob(run + "events.out.tfevents.*")[0]
ea = event_accumulator.EventAccumulator(ev, size_guidance={'scalars': 0})
ea.Reload()

def sample(tag, n=12):
    if tag not in ea.Tags().get('scalars', []):
        print(f"{tag}: NOT FOUND")
        return
    vals = ea.Scalars(tag)
    idxs = range(len(vals)) if len(vals) <= n else [round(i*(len(vals)-1)/(n-1)) for i in range(n)]
    pts = [(vals[i].step, vals[i].value) for i in idxs]
    print(tag + ":")
    print("  " + "  ".join(f"it{s}={v:.4f}" for s, v in pts))

tags = [
  "Train/mean_episode_length", "Train/mean_reward",
  "Episode_Termination/no_progress_timeout", "Episode_Termination/time_out",
  "Episode_Termination/ground_penetration", "Episode_Termination/base_vel_explosion",
  "Episode_Termination/joint_vel_explosion",
  "Episode_Metrics/stage/stage0", "Episode_Metrics/stage/stage1",
  "Episode_Metrics/stage/stage2", "Episode_Metrics/stage/stage3",
  "Episode_Metrics/success/candidate", "Episode_Metrics/success/ever_stood",
  "Episode_Metrics/success/stable_hold", "Episode_Metrics/success/fall_after_success",
  "Episode_Metrics/curriculum/assistance_force", "Episode_Metrics/curriculum/beta_rescaler",
  "Episode_Metrics/reward_group/task", "Episode_Metrics/reward_group/style",
  "Episode_Metrics/reward_group/regularization", "Episode_Metrics/reward_group/post_task",
  "Episode_Metrics/contact/feet", "Episode_Metrics/contact/torso",
]
for t in tags:
    sample(t)
print()
print("last iteration step seen:", ea.Scalars("Train/mean_episode_length")[-1].step)
EOF
