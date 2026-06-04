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
- **Staged reward system** — 16 reward terms covering height progress, uprighting, foot contact, CoM stability, and safety penalties
- **Rich contact logging** — 8 body-part contact sensors (head, torso, knees, forearms, hands, feet) feeding into metrics and rewards
- **Low-pass filtered actions** — EMA filter on joint position targets for smooth, deployable motor commands
- **Asymmetric actor-critic** — actor sees only deployable observations (no privileged info); critic sees full privileged state
- **Domain randomization ready** — motor strength, PD gains, action delay (all off by default, activate progressively)
- **Dual hardware support** — `backend: cpu` or `backend: cuda` in `hardware_config.yaml` switches everything automatically

---

## Robot

| Property | Value |
|---|---|
| Robot | Unitree G1 EDU U4 |
| DOF | 29 |
| Control | Joint position targets (PD actuators) |
| Control frequency | 100 Hz (physics at 500 Hz, decimation 5) |
| Episode length | 10 s |

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

### Reward Terms

| Term | Role |
|---|---|
| `base_height_exp` | Main standing height reward |
| `torso_height_exp` | Torso elevation |
| `body_up_exp` | Upright orientation |
| `stand_on_feet` | Feet on ground |
| `height_progress` | Dense progress signal (each step) |
| `stable_success_hold` | Stay standing for 50 consecutive steps |
| `prone_supine_righting` | Early-stage: leave the flat ground pose |
| `com_over_support` | CoM projection over support polygon |
| `dof_error_when_standing` | Near-default joint pose when upright |
| `feet_slip` | Penalize foot sliding |
| `feet_distance` | Keep feet in natural width |
| `head_contact_penalty` | Discourage head ground contact |
| `head_impact_penalty` | Penalize violent head impacts |
| `action_saturation` | Penalize commands outside joint limits |
| `joint_pos_limits` | Hard joint limit violation |
| `is_terminated` | Episode failure penalty |

### Termination Conditions

`time_out` · `nan` · `joint_vel_explosion` (>50 rad/s) · `ground_penetration` · `standing_fall_timeout` · `head_impact` (>1200 N)

---

## PPO Configuration

| Parameter | Value |
|---|---|
| Network | 512 → 256 → 128, ELU |
| Learning rate | 1e-3 (adaptive KL) |
| Entropy coefficient | 0.01 |
| Steps per env | 24 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Max iterations | 10 000 |
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

# With video recording every 2000 steps
bash train.sh Unitree-G1-GetUp --video

# Background (server)
nohup bash train.sh Unitree-G1-GetUp > train.log 2>&1 &
```

### Monitor with TensorBoard

```bash
tensorboard --logdir unitree_rl_mjlab/logs/rsl_rl/g1_getup
```

Key metrics to watch: `success/ever_stood`, `stage/stage3`, `termination/head_impact`, `contact/head_force_max`

### Visualize a checkpoint

```bash
cd unitree_rl_mjlab
MUJOCO_GL=glfw python scripts/play.py Unitree-G1-GetUp \
  --checkpoint-file logs/rsl_rl/g1_getup/<run>/model_<N>.pt \
  --num-envs 1 --device cpu --viewer native
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
- [x] Staged reward system (height progress, CoM, stable hold)
- [x] Contact sensors + metrics (head, torso, knees, forearms, hands)
- [x] Low-pass filtered joint actions
- [x] Asymmetric actor-critic (deployable actor obs)
- [x] Domain randomization framework (motor strength, PD gains, action delay)
- [ ] First stable training run on NVIDIA GPU
- [ ] Domain randomization activation (progressive)
- [ ] Sim2Real transfer on real Unitree G1
- [ ] Multi-position recovery (integrate HoST)

---

## Acknowledgements

- [`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab) — MuJoCo RL framework by Unitree Robotics
- [`HumanUP`](https://github.com/RunpeiDong/HumanUP) — original Isaac Gym get-up research (He et al., RSS 2025) — the inspiration and starting point for this project
- [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl) — PPO implementation
- [`legged_gym`](https://github.com/leggedrobotics/legged_gym) — RL environment framework
