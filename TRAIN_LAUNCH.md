  Training
  CPU / AMD:

  conda run -n unitree_rl_cpu python TRAIN_RENDER.py --engine mjlab --task Unitree-G1-GetUp --mode train --max-iterations
  100 --num-envs 64

  GPU NVIDIA:

  conda run -n unitree_rl_cuda python TRAIN_RENDER.py --engine mjlab --task Unitree-G1-GetUp --mode train --max-iterations
  100 --num-envs 4096

  Training breve di test:

  conda run -n unitree_rl_cpu python TRAIN_RENDER.py --engine mjlab --task Unitree-G1-GetUp --mode train --max-iterations 1
  --num-envs 2

  Se vuoi lanciare senza wrapper, diretto:
  CPU / AMD:

  cd /home/aidapt/TEST_STAND/Standup_G1/unitree_rl_mjlab
  HOME=/tmp MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp WARP_CACHE_DIR=/tmp/warp conda run -n unitree_rl_cpu python scripts/
  train.py Unitree-G1-GetUp --gpu-ids None --agent.max-iterations=100 --env.scene.num-envs=64

  GPU NVIDIA:

  cd /home/aidapt/TEST_STAND/Standup_G1/unitree_rl_mjlab
  conda run -n unitree_rl_cuda python scripts/train.py Unitree-G1-GetUp --gpu-ids 0 --agent.max-iterations=100
  --env.scene.num-envs=4096

  Render / Play
  Dopo il training, render dell’ultimo checkpoint:
  CPU / AMD:

  conda run -n unitree_rl_cpu python TRAIN_RENDER.py --engine mjlab --task Unitree-G1-GetUp --mode play

  GPU NVIDIA:

  conda run -n unitree_rl_cuda python TRAIN_RENDER.py --engine mjlab --task Unitree-G1-GetUp --mode play

  Training + play automatico:

  conda run -n unitree_rl_cpu python TRAIN_RENDER.py --engine mjlab --task Unitree-G1-GetUp --mode both --max-iterations
  100 --num-envs 64

Log e pesi
  I log e i checkpoint finiscono qui:

  - cartella esperimento: unitree_rl_mjlab/logs/rsl_rl/g1_getup
  - ogni run crea una sottocartella timestamp, ad esempio:
    unitree_rl_mjlab/logs/rsl_rl/g1_getup/2026-06-04_13-22-56_train_render_quick

  Dentro trovi:

  - pesi: model_0.pt, model_100.pt, ... e ultimo model_<iter>.pt
  - config dump: params/env.yaml, params/agent.yaml
  - diff git: git/unitree_rl_mjlab.diff
  - eventuali video play/train se registrati

  TensorBoard
  Sì, ora il task usa tensorboard.

  Lancio:

  cd /home/aidapt/TEST_STAND/Standup_G1/unitree_rl_mjlab
  conda run -n unitree_rl_cpu tensorboard --logdir logs/rsl_rl/g1_getup --port 6006

  Poi apri nel browser:

  http://localhost:6006
