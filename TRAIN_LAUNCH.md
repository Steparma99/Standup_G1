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

conda run -n unitree_rl_cuda python scripts/print_rewards.py 2026-07-30_15-46-24_v30_support_polygon


cd ~/Standup/Standup_G1/unitree_rl_mjlab/logs/rsl_rl/g1_getup
RUN="2026-06-24_09-34-38_Phase_3_no_jump/"


RUN=2026-07-16_16-17-51_v9_beta_prone        # e.g. 2026-07-16_..._v9_beta_prone_pass3
ITER=model_4500.pt    # e.g. 1000 (must match an existing model_<N>.pt)

FORCE=$(conda run -n unitree_rl_cuda python scripts/assist_force_at.py "$RUN" "$ITER")
BETA=$(conda run -n unitree_rl_cuda python scripts/assist_force_at.py "$RUN" "$ITER" \
  --tag Episode_Metrics/curriculum/beta_rescaler)
echo "force=$FORCE  beta=$BETA"

GETUP_EVAL_ASSIST_FORCE="$FORCE" MUJOCO_GL=egl conda run -n unitree_rl_cuda --no-capture-output \
  python scripts/play.py Unitree-G1-GetUp \
    --checkpoint-file "logs/rsl_rl/g1_getup/$RUN/model_${ITER}.pt" \
    --eval-beta "$BETA" \
    --num-envs 16 --max-extra-envs 8 \
    --cam-distance 6 --cam-elevation -15 \
    --video=True --video-length 500 --video-width 1920 --video-height 1080

mv logs/rsl_rl/g1_getup/$RUN/videos/play/rl-video-step-0.mp4 \
   logs/rsl_rl/g1_getup/$RUN/videos/play/model_${ITER}_force${FORCE}N_beta${BETA}.mp4



cd ~/Standup/Standup_G1/unitree_rl_mjlab
nohup bash scripts/auto_video.sh 2026-07-17_10-21-22_v11_beta_cooldown --interval 500 --once > /dev/null 2>&1
