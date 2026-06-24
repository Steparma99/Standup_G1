# Standup G1 — RL Get-Up Policy for Unitree G1 29DOF

Training a humanoid robot to stand up from the ground using Reinforcement Learning in MuJoCo simulation, with the goal of Sim2Real deployment on the real Unitree G1.

---

## Overview

This project trains a **get-up policy** for the **Unitree G1 (29 DOF)** humanoid robot.
The robot starts lying on the ground — either **supine** (face up) or **prone** (face down) — and must learn to stand on its feet autonomously, from randomized initial poses.

The simulation backend is **MuJoCo** via the [`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab) framework, using **PPO** (Proximal Policy Optimization) from [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl).

The training pipeline is designed to run on both **CPU (AMD/any hardware)** and **NVIDIA GPU**, controlled by a single config file.

---

## Key Features

- **Multi-pose randomized reset** — each episode starts from a randomly perturbed SUPINE or PRONE keyframe, with joint noise, root orientation noise, and full yaw randomization
- **HoST-style multi-critic reward** — 33 reward terms split into 4 groups (task / style / regularization / post-task), each with its own critic head, combined in PPO with fixed group weights (`2.5 / 1.9 / 0.1 / 1.0`)
- **Rich contact logging** — 8 body-part contact sensors (head, torso, knees, forearms, hands, feet) feeding into metrics and rewards
- **Low-pass filtered actions** — EMA filter on joint position targets for smooth, deployable motor commands
- **Asymmetric actor-critic** — actor sees only deployable observations (no privileged info); critic sees full privileged state
- **Headless video recording** — `play.py --video` writes an mp4 of a policy (with multi-env, camera, and quality flags) without needing an interactive viewer
- **Assistance-force curriculum** — a decaying upward support force on the torso bootstraps early standing, annealing to zero per-env as each robot succeeds
- **Domain randomization ready** — motor strength, PD gains, action delay (all off by default, activate progressively)
- **Dual hardware support** — `backend: cpu` or `backend: cuda` in `hardware_config.yaml` switches everything automatically

---

## Robot

| Property | Value |
|---|---|
| Robot | Unitree G1 EDU U4 |
| DOF | 29 |
| Control | Joint position targets (PD actuators) |
| Control frequency | 50 Hz policy (physics 200 Hz, decimation 4) — deployment uses 500 Hz PD / decimation 10 |
| Episode length | 10 s (500 control steps) |

---

## Training Setup

### Actor Observations (deployable — no privileged state)

| Signal | Dim | Notes |
|---|---|---|
| Angular velocity (IMU) | 3 | with noise |
| Projected gravity | 3 | orientation estimate |
| Joint positions (relative) | 29 | |
| Joint velocities (relative) | 29 | |
| Last action | 29 | |
| PD tracking error | 29 | commanded target − measured q |
| **History length** | **×4** | **total actor dim: 488** |

### Critic Observations (privileged — training only)

Actor obs + pelvis height, torso height, CoM, lin vel, foot positions, foot contact forces, contact states for all body parts, stage gates, foot friction.

### Reward System (HoST-style multi-critic)

33 terms across 4 groups; each group has its own critic, combined in PPO with fixed group
weights (`task 2.5 / style 1.9 / regularization 0.1 / post-task 1.0`). Group membership lives in
`src/tasks/getup/rl/reward_groups.py`, the terms in `getup_env_cfg.py`, and the functions in
`mdp/rewards.py`. Every term is logged per-episode as `Episode_Reward/<term>` (TensorBoard + W&B);
the per-group sums appear as `Episode_Metrics/reward_group/*`.

- **TASK** — high-level objectives that drive the rise:
  `height_progress`, `prone_recovery`, `supine_rising_prep`, `task_head_height`, `task_base_orientation`
- **STYLE** — shape the motion:
  `style_shank_orientation`, `style_ankle_parallel`, `style_foot_displacement`, `style_foot_distance`,
  `style_base_ang_vel`, `style_hip_deviation`, `style_knee_deviation`, `style_waist_yaw_deviation`,
  `style_shoulder_roll_deviation`, **`style_waist_upright`** *(new — keeps the trunk straight above 0.5 m)*, `style_feet_stumble`
- **REGULARIZATION** — smoothness / effort (group weight 0.1):
  `action_rate_l2`, `action_acc_l2`, `joint_vel_l2`, **`reg_arm_vel`** *(new — calms shoulder/elbow/wrist)*,
  `joint_tracking_error`, `joint_pos_limits`, `joint_vel_limits` (`joint_acc_l2` / `joint_torques_l2` / `joint_power_l2` wired but off by default)
- **POST-TASK** — shape the held stand, gated to pelvis height > 0.65 m:
  `post_base_height`, `post_base_orientation`, `post_base_ang_vel`, `post_base_lin_vel`,
  `post_upper_body_posture`, `post_feet_parallel`, `stable_success_hold`

> **Latest tuning (`smoother_v1`)** — stronger smoothness penalties (`action_rate_l2`/`action_acc_l2`
> `-1e-3 → -2.5e-3`), the new `reg_arm_vel` and `style_waist_upright` terms, sharper standing
> uprightness (`post_base_orientation` scale `5 → 8`), and earlier arm-posture shaping
> (`post_upper_body_posture` active from 0.5 m) — for a less curved torso and calmer arms.

### Termination Conditions

`time_out` · `nan` · `joint_vel_explosion` · `base_vel_explosion` · `ground_penetration` · `no_progress_timeout` (stall guard, exempt once standing)

---

## PPO Configuration

| Parameter | Value |
|---|---|
| Algorithm | Multi-critic PPO — 4 group critics (`λ_actor=1.0`, `λ_critic=0.1`) |
| Actor network | 512 → 256 → 128, ELU (clamped Gaussian policy) |
| Critic network | 512 → 256, ELU (one per reward group) |
| Learning rate | 5e-4 (adaptive KL) |
| Entropy coefficient | 0.005 |
| Steps per env | 50 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Max iterations | 12 000 |
| Save interval | 100 iterations |

---

## Project Structure

```
Standup_G1/
├── hardware_config.yaml          # Backend switch: cpu / cuda
├── train.sh                      # Main training launcher
├── TRAIN_RENDER.py               # Unified train + play wrapper
├── install.sh                    # Conda env setup
│
├── unitree_rl_mjlab/
│   ├── src/tasks/getup/          # Custom get-up task
│   │   ├── config/g1/
│   │   │   ├── env_cfgs.py       # G1-specific environment config
│   │   │   └── rl_cfg.py         # PPO hyperparameters
│   │   ├── getup_env_cfg.py      # Base environment (rewards, obs, events)
│   │   └── mdp/
│   │       ├── rewards.py        # All reward functions
│   │       ├── observations.py   # Actor + privileged critic obs
│   │       ├── events.py         # Reset: multi-pose randomized
│   │       ├── actions.py        # LowPassJointPositionAction
│   │       ├── terminations.py   # Safety terminations
│   │       └── metrics.py        # TensorBoard metrics (stages, success, contact)
│   ├── src/assets/robots/unitree_g1/  # G1 29DOF MuJoCo XML + meshes
│   └── scripts/
│       ├── train.py              # Training entry point
│       └── play.py               # Policy visualization
│
└── simulation/                   # Original HumanUP Isaac Gym code (reference)
    └── legged_gym/
        └── resources/robots/g1_modified/   # G1 URDFs + custom collision meshes
```

---

## Installation

**Requirements:** Linux, Python 3.11, Conda

```bash
# Clone the repo
git clone https://github.com/Steparma99/Standup_G1.git
cd Standup_G1

# Set backend in hardware_config.yaml (cpu or cuda), then:
bash install.sh
```

---

## Usage

### Verify the environment builds (sanity check)

```bash
conda activate unitree_rl_cpu   # or unitree_rl_cuda
cd unitree_rl_mjlab
MUJOCO_GL=osmesa python - <<'EOF'
import mjlab.tasks, src.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
cfg = load_env_cfg("Unitree-G1-GetUp")
cfg.scene.num_envs = 4
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
obs, _ = env.reset()
print("OK — actor:", tuple(obs['actor'].shape), "| action dim:", env.action_manager.total_action_dim)
EOF
```

### Training

```bash
# CPU (development / smoke test)
bash train.sh Unitree-G1-GetUp

# NVIDIA GPU (set backend: cuda in hardware_config.yaml)
bash train.sh Unitree-G1-GetUp

# Background (server, no tmux)
nohup bash train.sh Unitree-G1-GetUp > train.log 2>&1 &
disown
```

`train.sh` reads `hardware_config.yaml` and auto-injects `--env.scene.num-envs` and `--gpu-ids`,
so you only pass the task and any agent overrides. Use `--agent.logger=wandb` for W&B (else `tensorboard`).

### Resume training from a checkpoint

```bash
bash train.sh Unitree-G1-GetUp \
  --agent.resume=True \
  --agent.load-run=<run-dir-name> \
  --agent.load-checkpoint=model_<N>.pt \
  --agent.max-iterations=1500 \
  --agent.run-name=smoother_v1 \
  --agent.logger=wandb
```

- `--agent.load-run` is a **regex matched against run-dir names** under `logs/rsl_rl/g1_getup/`
  (not an absolute path; `.*` picks the most recent run).
- `--agent.max-iterations` is **added on top** of the loaded checkpoint's iteration count.
- Resuming with a new `--agent.run-name` writes to a fresh timestamped dir while loading the old weights.

### Monitor with TensorBoard

```bash
tensorboard --logdir unitree_rl_mjlab/logs/rsl_rl/g1_getup
```

Key metrics to watch: `Episode_Metrics/success/stable_hold`, `Episode_Metrics/stage/stage3`,
`Episode_Metrics/reward_group/*`, and any `Episode_Reward/<term>` (e.g. `reg_arm_vel`, `style_waist_upright`).

### Visualize a checkpoint

```bash
# Interactive, local display:
cd unitree_rl_mjlab
MUJOCO_GL=glfw python scripts/play.py Unitree-G1-GetUp \
  --checkpoint-file logs/rsl_rl/g1_getup/<run>/model_<N>.pt \
  --num-envs 1 --device cpu --viewer native

# Headless web viewer (server): add --viewer viser and MUJOCO_GL=egl
```

### Record a video (headless)

`--video` records an mp4 and exits — no interactive viewer required.

```bash
cd unitree_rl_mjlab
MUJOCO_GL=egl python scripts/play.py Unitree-G1-GetUp \
  --checkpoint-file logs/rsl_rl/g1_getup/<run>/model_<N>.pt \
  --num-envs 1 \
  --video True --video-length 500 --video-width 1280 --video-height 720
# -> logs/rsl_rl/g1_getup/<run>/videos/play/rl-video-step-0.mp4
```

The video is written at the policy rate (50 fps), so playback is **real-time**
(`--video-length 500` → exactly 10 s). For a slow-motion copy to analyze the motion, re-encode:
`ffmpeg -i rl-video-step-0.mp4 -vf "setpts=2*PTS" slowmo_0.5x.mp4` (0.5× speed).

`play.py` recording / camera flags:

| Flag | Purpose |
|---|---|
| `--video True` | Record an mp4 headlessly, then exit (boolean needs explicit `True`) |
| `--video-length` | Frames to record (50 fps → 500 ≈ 10 s, real-time) |
| `--video-width` / `--video-height` | Output resolution |
| `--max-extra-envs` | Also render N neighbor envs (multi-env view; needs `--num-envs > 1`) |
| `--env-idx` | Which environment the camera tracks |
| `--cam-distance` / `--cam-elevation` / `--cam-azimuth` / `--cam-fovy` | Camera framing |
| `--shadows False` / `--reflections False` | Disable shadows / reflections (default `True`) |
| `--eval-beta` | Override the action-rescaler beta for an in-progress checkpoint |

Multi-environment showcase:

```bash
MUJOCO_GL=egl python scripts/play.py Unitree-G1-GetUp \
  --checkpoint-file logs/rsl_rl/g1_getup/<run>/model_<N>.pt \
  --num-envs 16 --max-extra-envs 8 --cam-distance 6 --cam-elevation -15 \
  --video True --video-width 1920 --video-height 1080
```

---

## Hardware Config

Edit `hardware_config.yaml` to match your machine:

```yaml
backend: "cpu"    # or "cuda"

cpu:
  num_envs: 64
  mujoco_gl: "osmesa"   # offscreen; use "glfw" for interactive viewer

cuda:
  gpu_ids: [0]
  num_envs: 4096
  mujoco_gl: "egl"
```

---

## Roadmap

- [x] MuJoCo migration from Isaac Gym
- [x] G1 29DOF SUPINE + PRONE keyframes
- [x] Multi-pose randomized reset
- [x] HoST-style multi-critic reward (4 groups, per-group critics)
- [x] Assistance-force curriculum (decaying torso support)
- [x] Contact sensors + metrics (head, torso, knees, forearms, hands)
- [x] Low-pass filtered joint actions
- [x] Asymmetric actor-critic (deployable actor obs)
- [x] Domain randomization framework (motor strength, PD gains, action delay)
- [x] First stable training run on NVIDIA GPU (~4.8k iters, stable_hold ≈ 0.78)
- [x] Headless video recording (multi-env / camera / quality flags)
- [x] Reward-smoothing pass for calmer arms + straighter torso (`smoother_v1`)
- [ ] Domain randomization activation (progressive)
- [ ] Sim2Real transfer on real Unitree G1
- [ ] Multi-position recovery (integrate HoST)

---

## Acknowledgements

- [`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab) — MuJoCo RL framework by Unitree Robotics
- [`HumanUP`](https://github.com/RunpeiDong/HumanUP) — original Isaac Gym get-up research (He et al., RSS 2025) — the inspiration and starting point for this project
- [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl) — PPO implementation
- [`legged_gym`](https://github.com/leggedrobotics/legged_gym) — RL environment framework
