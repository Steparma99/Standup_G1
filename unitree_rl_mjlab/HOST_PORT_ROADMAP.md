# Porting HoST to a personal 29-DoF G1 on MuJoCo — roadmap

Goal: a personal HoST stand-up pipeline with **your 29-DoF G1** and **MuJoCo (mjlab)**, not
Isaac Gym. The *objective* is identical to HoST; only the robot DoF count and the simulator
differ. This file is the macro plan: what's done, what to check, what to implement, and what
to "steal" (copy the *logic*, not the code) from the HoST repo under `HoST/`.

## The two invariants (why this is a port, not a copy)

1. **29 DoF vs HoST's 23 DoF** (`HoST/.../g1/g1_config_ground.py: num_actions=23`). Every joint
   index, keyframe, PD gain, obs dimension and action dimension differs. → *adapt* per-joint data.
2. **MuJoCo/mjlab vs Isaac Gym.** Every line of HoST's `envs/` is Isaac Gym (`self.gym.*`,
   `gymtorch`). ~100% of the *sim code* is non-portable; ~100% of the *design* is. → *reimplement*
   sim, *steal* logic.

**HoST (inheritance + Isaac Gym) → your pipeline (composition + mjlab managers):**

| HoST (a method on the env class) | Your mjlab home | File |
|---|---|---|
| `compute_reward` / `_reward_*` | RewardManager terms | `mdp/rewards.py` + reward dict |
| `check_termination` | TerminationManager terms | `mdp/terminations.py` |
| `compute_observations` | ObservationManager terms | `mdp/observations.py` |
| `reset_idx` / `_reset_dofs` / `_reset_root_states` | EventManager `mode="reset"` | `mdp/events.py` |
| `_compute_torques` (PD) | actuator cfg + action term | `g1_constants.py`, `mdp/actions.py` |
| `pull_force` (in `step`) | `AssistanceCurriculum` event | `mdp/events.py` |
| terrain build | MuJoCo geoms / heightfield | scene cfg (to build) |
| `task_registry.register` | `register_mjlab_task` | `config/g1/__init__.py` |
| `G1Cfg*` config classes | env_cfg builder functions | `config/g1/env_cfgs.py` |

---

## Phase 0 — DONE (committed)
- Multi-critic PPO (4 group critics, per-group GAE → per-group normalize → weighted advantage) + L2C2.
- Full HoST reward: task(2) / style(8) / regularization(9) / post-task(6) = 25 terms.
- HoST ground/prone terminations (timeout + dof-vel 300 + base-vel 20, grace 30) + safety nets.
- HoST-faithful reset (`reset_to_random_keyframe` + settle + assist + beta curricula).
- Per-group reward logging (`Episode_Metrics/reward_group/*`).
- Verified end-to-end: stability over resets, checkpoint/resume/ONNX.

---

## Phase 1 — Make GROUND actually learn (← current focus)
Empirical tuning, **not new code**. Get the robot to reliably stand from supine+prone on flat ground.
- **Watch:** `success/ever_stood` ↑, `task_head_height` ↑, `stage2/3` ↑, `curriculum/assistance_force` ↓.
- **Levers if stuck:** assist-force magnitude/decay; a style-penalty warm-up (ramp the binary
  −10/−20 penalties in over the first N iters so they don't suppress early flailing); reward-weight
  rebalance; the head-height target/proxy.
- **Steal/verify from:** `g1_config_ground.py` (curriculum + reward weights), `host_ground.py`.

---

## Phase 2 — Complete ground-task fidelity (the deferred reward bits)
- **`ankle_parallel`** (style, HoST weight 20) — needs **foot keypoints** (several sites per foot to
  take the height-variance). Implement: add keypoint sites to the foot in the G1 MJCF (or compute
  from the existing `*_foot[1..7]_collision` geom positions), then `var(z_keypoints) < 0.05 → +20`.
  Steal: HoST `feet_ori` / ankle-keypoint code.
- **`feet_stumble`** — 0 on flat ground; wire when terrains arrive (Phase 4).
- **Head height** — currently the `torso_link` proxy (no head body in the MJCF). If you add a head
  body/site, switch `task_head_height` to the true head.
- **Verify** assist (`pull_force`) and beta (`action_rescale`) magnitude/decay against
  `g1_config_ground.py` exactly (currently calibrated by feel).

---

## Phase 3 — Domain randomization (sim2real robustness)
HoST randomizes on reset (`g1_config_ground.py` `domain_rand`, and the `reset_idx` block you pasted):
Kp/Kd factors, motor strength, actuation offset, **action delay**. You currently have
`base_com` / `foot_friction` / `encoder_bias`.
- Implement as mjlab events + actuator config:
  - **Kp/Kd per-env** → per-env actuator gain randomization.
  - **motor strength** → per-env torque scale on the applied actuation.
  - **action delay** → a delay buffer (event or action-term wrapper).
- Steal: the `domain_rand` config block + the `reset_idx` randomization code.
- Check: ground still converges with DR on; add a DR curriculum (ramp ranges) if it hurts early.

---

## Phase 4 — Multi-terrain: platform / wall / slope (the big infrastructure)
Structure = **one shared mdp library + one config function per terrain + one registered task each**
(`Unitree-G1-GetUp-Ground/-Platform/-Wall/-Slope`). This mirrors HoST's `g1_config_*` files in
mjlab's composition style, with **zero duplicated physics logic**.

- **4a. Terrain in MuJoCo (hardest part).** Build platform/slope/wall as MJCF geoms (boxes/planes/
  ramps) or a heightfield in the scene cfg. HoST's `utils/terrain.py` (Isaac Gym heightfield/trimesh)
  is **not portable** — reuse only the *dimensions/layout logic*. Steal: `utils/terrain.py`,
  `host_platform.py` / `host_wall.py` / `host_slope.py` (spawn + terrain interaction).
- **4b. Per-terrain spawn poses** (against a wall, on a platform edge, on a slope) → new keyframes +
  `reset_to_random_keyframe` params per config.
- **4c. Per-terrain terminations** (functions mostly exist or are small additions):
  - platform: low base height `root_z < 0.2` (have `root_height_below_minimum`) + foot height
    `> 0.35` (have `feet_too_high`).
  - wall: low **head** height `< 0.2` (add a head-height termination) + foot height.
  - slope: relative height vs reference `< −0.1` (add a relative-height termination).
- **4d. Per-terrain reward weights** — make the terrain-dependent values config params:
  `style_knee_deviation` Ground −0.25 / PSW −10; `feet_stumble` Ground 0 / PSW −25.
- **4e. Register each terrain** via `register_mjlab_task` (the `envs/__init__.py` pattern).
- Steal: `envs/__init__.py` (registration), `g1_config_{platform,wall,slope}.py` (all thresholds).

---

## Phase 5 — Sim2real deployment
- ONNX export already wired; deploy policy runs at fixed `beta = 0.25`.
- Hardware interface (Unitree SDK) + `hardware_config.yaml`. Out of scope for sim, but the end goal.

---

## Recommended order & the "steal vs reimplement" rule
**Order:** 1 → 2 → 3 → 4 → 5. Get ground rock-solid (1–2) before terrains (4). DR (3) before real
deployment; terrains (4) are independent of DR. **Don't start 4 until ground stands** — terrains
add variables that make debugging the base behavior much harder.

**Cheat sheet for every HoST file you open:**
- **STEAL (copy the math/thresholds):** reward formulas, termination thresholds, curriculum decay
  rules, obs structure, config numbers. Sources: `g1_config_*.py`, the `_reward_*`/`check_termination`
  methods in `host_*.py`.
- **REIMPLEMENT (mjlab-native, cannot copy):** every sim call, terrain geometry, PD/actuator, the env
  loop — because Isaac Gym ≠ MuJoCo.
- **ADAPT (23 → 29 DoF):** joint indices, keyframes, PD gains, obs/action dims.
