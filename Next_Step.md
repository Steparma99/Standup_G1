# G1 Get-Up — Complete Implementation & Research Roadmap

This file has two parts:

- **Part 1 — XML Migration Pipeline.** Engineering steps to move the get-up training onto the real official G1 model (`g1_body29_hand14.xml`, 43 DOF) under one unified policy.
- **Part 2 — Research-Grounded Policy Enhancements.** The literature-backed ideas to layer on top of the working HoST baseline to move from "a standup" to a *robust, smooth standup usable in many scenarios*. Each idea is documented with its reference paper, the idea taken, the reason, what it adds, and implementation notes.

**Recommended global ordering:** complete Part 1 (migrate XML → smoke test → fresh Phase-1 baseline) **first**, then layer Part 2 enhancements one at a time. Do **not** introduce new reward terms or critic inputs in the same run as the `29→43` action-space change — otherwise regressions cannot be attributed.

---
---

# PART 1 — XML Migration Pipeline (`g1_body29_hand14.xml`, 43-DOF Unified Policy)

## Context & Goal

Migrate get-up training from the old model `unitree_g1/xmls/g1.xml` (29 DOF, simplified rubber hands) to the official model `g1/g1_body29_hand14.xml` (29 body + 14 hand DOF = 43 total) with correctly articulated fingers.

**Decision (confirmed):** all joints — including the 14 finger joints — are controlled by a **single unified RL policy**. The policy learns to keep the fragile hands away from the ground via a contact-force penalty. This is the cleanest path for sim2real: one policy, one controller, no parallel hand controller.

**Cost:** action space grows `29 → 43`; existing 29-DOF checkpoints become incompatible. Training restarts from scratch at Phase 1.

## ⚠️ Three Corrections vs. the Earlier Draft (read first)

### 1. The mjlab training XML must have **NO `<actuator>` block** at all
The current working `g1.xml` contains zero `<actuator>` elements (`grep -c "<actuator>"` → `0`). `mjlab` builds position actuators from `BuiltinPositionActuatorCfg` groups in `g1_constants.py`. The new XML ships a 43-entry `<motor>` torque block that would collide/duplicate with mjlab's actuators. **Delete the whole `<actuator>` block.** Hands become a position-control group in `g1_constants.py`. This supersedes the earlier back-and-forth about adding/removing hand motors — that block is going away entirely.

### 2. The mesh / asset path needs a one-line fix
`get_assets()` in `g1_constants.py` uses `"assets"` but ignores its `meshdir` argument. The new model lives at `g1/` with `meshdir="meshes"` and STLs in `g1/meshes/`. Without a fix, asset loading looks in the wrong place.

### 3. The hand-contact penalty is almost free
The body-part contact sensors `contact_hand_left` / `contact_hand_right` already exist in `env_cfgs.py` (lines ~270–285). We only need a small reward function reading them, mirroring the existing `head_contact_penalty` in `rewards.py`. No new sensor wiring.

## Collision-geom architecture

`g1_constants.py` `FULL_COLLISION` matches geoms by name regex `.*_collision` and overrides their `condim`, `friction`, `priority`, `contype`. The body-part `ContactSensorCfg`s in `env_cfgs.py` also reference these geoms by exact name. So the new XML must contain named collision geoms identical to:

```
pelvis_collision, torso_collision, head_collision,
(left|right)_hip_collision, (left|right)_thigh_collision, (left|right)_shin_collision,
(left|right)_linkage_brace_collision, (left|right)_elbow_yaw_collision,
(left|right)_wrist_collision, (left|right)_foot[1-7]_collision
```

The XML only needs the right name + geometry; `condim`/`friction` are set by `FULL_COLLISION` and may be omitted in the XML.

## Implementation Steps

All XML edits are in `src/assets/robots/unitree_g1/xmls/g1_body29_hand14.xml` unless noted.

### Step 1 — Edit the new XML

**1a. Delete the entire `<actuator>` block.** Remove every `<motor>` element; mjlab creates position actuators itself. *Motivation:* avoids duplicate/conflicting actuators; matches how `g1.xml` works today.

**1b. Replace the `<sensor>` block** with the names the code reads (`robot/imu_lin_acc`, `robot/imu_lin_vel`, `robot/imu_ang_vel`):
```xml
<sensor>
  <gyro          name="imu_ang_vel"  site="imu_in_pelvis"/>
  <velocimeter   name="imu_lin_vel"  site="imu_in_pelvis"/>
  <accelerometer name="imu_lin_acc"  site="imu_in_pelvis"/>
  <subtreeangmom name="root_angmom"  body="pelvis"/>
</sensor>
```
*Motivation:* the new XML is mjx-style with non-matching names; sensors are looked up by name at runtime.
> **Note for Part 2:** `subtreeangmom name="root_angmom"` is already the **angular part of the centroidal momentum** — Enhancement 2 reuses it. Consider adding `subtreecom` and `subtreelinvel` here at the same time (see Part 2, Enhancement 2).

**1c. Add named collision geoms** (copy verbatim from `g1.xml`):

| Body | Geom(s) |
|---|---|
| `pelvis` | `pelvis_collision`, capsule, `size="0.07"`, `pos="0 0 -0.08"` |
| `left/right_hip_roll_link` | `*_hip_collision`, capsule, `size="0.06"`, `fromto="0.02 0 0 0.02 0 -0.08"` |
| `left/right_hip_yaw_link` | `*_thigh_collision`, capsule, `size="0.055"`, `fromto="-0.0 0 -0.03 -0.06 0 -0.17"` |
| `left/right_knee_link` | `*_shin_collision` + `*_linkage_brace_collision`, `size="0.03"` |
| `torso_link` | `torso_collision`, capsule, `size="0.09"` + `head_collision`, sphere, `size="0.06"`, `pos="0 0 .43"` |
| `left/right_elbow_link` | `*_elbow_yaw_collision`, capsule, `size="0.035"` |
| `left/right_wrist_pitch_link` | `*_wrist_collision`, capsule, `size="0.035"` |
| `left/right_wrist_yaw_link` | `*_hand_collision`, capsule, `size="0.035"` |

`condim`/`friction` are applied by `FULL_COLLISION`, so geoms can be plain `type="capsule"`/`type="sphere"`.

**1d. Replace foot geoms + add foot sites.** In `left/right_ankle_roll_link`: delete the 4 anonymous `<geom size="0.005" .../>` spheres; add the 7 named foot capsules `*_foot[1-7]_collision`; add `<site name="left_foot"/>` / `<site name="right_foot"/>`. Copy from `g1.xml` lines 99–106 / 141–148. *Motivation:* `stand_on_feet`, `feet_ground_contact`, and `FEET_ONLY_COLLISION` key off these exact names/sites.

**1e. Add palm sites.** `<site name="left_palm" pos="0 0 0"/>` inside `left_wrist_yaw_link`; mirror for right. *Motivation:* parity with `g1.xml`; referenced by some scripts.

**1f. Add `<contact>` excludes:**
```xml
<contact>
  <exclude body1="left_elbow_link"  body2="left_wrist_pitch_link"/>
  <exclude body1="right_elbow_link" body2="right_wrist_pitch_link"/>
  <exclude body1="pelvis"           body2="right_hip_roll_link"/>
  <exclude body1="pelvis"           body2="left_hip_roll_link"/>
</contact>
```

**1g. Remove embedded scene setup:** delete the trailing `<statistic>`, `<visual>`, second `<asset>` (sky/ground textures), and second `<worldbody>` (light + floor geom). *Motivation:* mjlab/the scene file provide their own ground & lighting; a floor inside the robot XML would duplicate the environment floor.

### Step 2 — `g1_constants.py`

**2a. Point `G1_XML` at the new file** (lines 25–27):
```python
G1_XML: Path = SRC_PATH / "assets/robots/unitree_g1/xmls/g1_body29_hand14.xml"
```

**2b. Fix `get_assets()` to honor `meshdir`** (line 33):
```python
assets_dir = G1_XML.parent / meshdir   # was: / "assets"
```

**2c. Add a hand actuator group.** The 14 finger joints are small motors (XML `actuatorfrcrange` ±1.4 Nm; `thumb_0` ±2.45 Nm). Add a `BuiltinPositionActuatorCfg` with low stiffness so the policy can pose fingers but not fight obstacles:
```python
G1_ACTUATOR_HAND = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hand_.*_joint",),
    stiffness=5.0,          # pose fingers gently rather than fight obstacles
    damping=0.2,
    effort_limit=1.4,       # matches XML cap; thumb_0 is 2.45 but 1.4 is safe/uniform
    armature=...,           # BuiltinPositionActuator reflected_inertia
)
```
Add it to both articulations (the get-up task uses `HOST`): `G1_ARTICULATION.actuators`, `G1_ARTICULATION_HOST.actuators` (line ~503), and the delayed variant in `get_g1_supine_robot_cfg_with_delay()` if DR action-delay is used. *Motivation:* every joint must be actuated or it flops; `G1_ACTION_SCALE` is auto-derived from `G1_ARTICULATION.actuators`, so the action vector updates automatically to 43 outputs.

**2d. Optional hand keyframe defaults.** Finger joints not listed in keyframe dicts default to `0.0` (relaxed/open), a safe home target. Leave as-is unless a visual check shows a bad rest pose.
> **Note for Part 2:** this same keyframe machinery is what Enhancement 4 (Key State Initialization) reuses to seed diverse start postures.

### Step 3 — Hand-contact penalty reward

**3a. Add reward in `src/tasks/getup/mdp/rewards.py`** (mirror `head_contact_penalty`, ~line 193), reading `contact_hand_left` / `contact_hand_right` via `_contact_force_norm` (line 28), with a reset-ramp mask so settling contacts are not punished:
```python
def hand_contact_penalty(env, ramp_steps=...):
    l, _ = _contact_force_norm(env, "contact_hand_left")
    r, _ = _contact_force_norm(env, "contact_hand_right")
    return (l | r).float() * _reset_ramp(env, ramp_steps)
```

**3b. Register in `getup_env_cfg.py`:**
```python
"reg_hand_contact": RewardTermCfg(
    func=mdp.hand_contact_penalty,
    weight=-5.0,
    params={...},
)
```
Start at `-5.0`; tune up if the policy still uses hands.

**3c. Map to a group in `src/tasks/getup/rl/reward_groups.py`:**
```python
"reg_hand_contact": "regularization"
```
*Motivation:* `build_group_onehot` raises if any active term is unmapped.
> **Note for Part 2:** this is your **first concrete contact-control reward**. It belongs to the same theme that Enhancements 2 and 3 generalize (controlling *which* contacts the policy uses, not just how smooth it is).

### Step 4 — Visualization model `scene_g1.xml`

The standalone scripts (`pose_lab.py`, `visualize_keyframes.py`, `diag_contact_geoms.py`, …) and `simulate/config.yaml` use `unitree_g1/xmls/scene_g1.xml`, which embeds its own body + `<actuator>` block. Lowest-risk: create a parallel `scene_g1_hand14.xml` that includes the new body + ground/light, and point `simulate/config.yaml` at it when visualizing this model. Visualization only; does not affect training; deferrable until after training works.

### Step 5 — Verification

```bash
cd ~/Standup/Standup_G1/unitree_rl_mjlab
```

**5a. Asset + actuator sanity (via mjlab spec):**
```bash
python -c "
from src.assets.robots.unitree_g1.g1_constants import get_spec, G1_ARTICULATION
import mujoco
m = get_spec().compile()
print('nq', m.nq, 'nv', m.nv, 'nu', m.nu)   # expect nv=49, nu=43 after mjlab adds actuators
"
```

**5b. Visual check** (fingers, named collision geoms, foot sites):
```bash
python -m src.assets.robots.unitree_g1.g1_constants
```

**5c. Short training smoke test** (catches action-size + new reward term):
```bash
nohup env MUJOCO_GL=egl python scripts/train.py Unitree-G1-GetUp \
  --agent.max-iterations=10 \
  --env.scene.num-envs=16 \
  > logs/test_new_xml.log 2>&1
cat logs/test_new_xml.log
```
Expect action dim 43 and no missing-geom / unmapped-reward errors.

### Step 6 — Fresh Phase-1 training (43 DOF)

```bash
nohup env MUJOCO_GL=egl python scripts/train.py Unitree-G1-GetUp \
  --agent.experiment-name=g1_getup \
  --agent.run-name="Phase1_43dof" \
  --agent.max-iterations=4000 \
  --env.scene.num-envs=4096 \
  > logs/train_phase1_43dof.log 2>&1 &
disown
```

### Part 1 Progress Tracker

| # | Step | File | Status |
|---|---|---|---|
| 1a | Delete `<actuator>` block | `g1_body29_hand14.xml` | ☐ |
| 1b | Fix sensor names (+`velocimeter`, +`root_angmom`) | `g1_body29_hand14.xml` | ☐ |
| 1c | Add named collision geoms | `g1_body29_hand14.xml` | ☐ |
| 1d | Replace foot geoms + foot sites | `g1_body29_hand14.xml` | ☐ |
| 1e | Add palm sites | `g1_body29_hand14.xml` | ☐ |
| 1f | Add `<contact>` excludes | `g1_body29_hand14.xml` | ☐ |
| 1g | Remove embedded scene setup | `g1_body29_hand14.xml` | ☐ |
| 2a | Point `G1_XML` to new file | `g1_constants.py` | ☐ |
| 2b | Fix `get_assets()` meshdir | `g1_constants.py` | ☐ |
| 2c | Add hand actuator group (both articulations) | `g1_constants.py` | ☐ |
| 2d | Optional hand keyframe defaults | `g1_constants.py` | ☐ |
| 3a | `hand_contact_penalty` | `rewards.py` | ☐ |
| 3b | Register `reg_hand_contact` term | `getup_env_cfg.py` | ☐ |
| 3c | Map term to `regularization` group | `rl/reward_groups.py` | ☐ |
| 4 | Visualization scene file (`scene_g1_hand14.xml`) | new | ☐ (deferrable) |
| 5a | Spec parse check (`nv=49`, `nu=43`) | — | ☐ |
| 5b | Visual check | — | ☐ |
| 5c | Smoke training (10 iters) | — | ☐ |
| 6 | Fresh Phase-1 training | — | ☐ |

### Part 1 Risk Notes
- Action dim `29 → 43` ⇒ checkpoints incompatible (intended; fresh training).
- Hand subtree mass now distributed into finger bodies; total mass correct but dynamics differ slightly.
- New `left/right_wrist_yaw_link` inertia differs from `g1.xml` (which folded hand mass into the wrist); with real finger bodies this is now physically correct.

---
---

# PART 2 — Research-Grounded Policy Enhancements

## Where we are, and the goal

**Baseline (already implemented):** HoST multi-critic get-up. It produces *a* standup, but it is rough on (1) **motion quality** ("not the best in terms of movement"), (2) **contact behavior** across surfaces, and (3) **generalization** across terrains and starting postures.

**Goal:** a *smooth, contact-sensible, robust* standup that works in many scenarios — not just on open flat ground from a supine pose.

**Diagnostic principle.** These three gaps have different root causes and different fixes, so treat them separately rather than throwing one bigger reward at all of them:
- *Motion quality* → architecture/reward fix (cheap) → Enhancement 1.
- *Contact behavior* → balance/contact-aware critic (medium) → Enhancements 2 & 3.
- *Posture/terrain coverage* → curriculum + state initialization (sim-budget) → Enhancements 3 & 4.
- *Hardware safety* → safety filtering (gate before real robot) → Enhancement 5.

## What from HoST is already doing the heavy lifting (keep all of it)

These three HoST mechanisms are the foundation Part 2 builds on; do not remove them:
1. **Multi-critic architecture** — separate critics per reward group, so dense regularization and sparse task rewards don't cross-talk. This is the platform every enhancement below plugs into.
2. **Action rescaler β** — `pd_t = p_t + β·a_t`, with `a_t ∈ [−1,1]`, where β tightens over training to implicitly bound joint speed/torque and suppress violent motion.
   - **Verify in your code:** β must decrease **coupled to task progress** (HoST drops it only *after* the head reaches target height), **not** on a flat wall-clock schedule. Task-coupling is what prevents tightening from killing exploration before a working trajectory exists. A time-based decay loses this property.
3. **Smoothness regularization** — penalizes oscillation in the action sequence.

> Enhancement 1 (LCP) stacks *with* β and smoothness; they act at different levels (β bounds action magnitude, smoothness penalizes oscillation in reward, LCP constrains the policy function's sensitivity). They are complementary, not redundant.

---

## Enhancement 1 — Lipschitz-Constrained Policies (LCP)

**Reference.** Chen, He, et al., *Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies*, arXiv:2410.11825 (IROS 2025). Code: `github.com/zixuan417/smooth-humanoid-locomotion`.

**Idea to take.** Add a differentiable Lipschitz (gradient-penalty) constraint to the policy during training, bounding how fast the policy output can change with respect to input changes. This produces inherently smooth motion at the *function* level, instead of relying on a large set of hand-tuned smoothness reward weights.

**Why I want it.** My current standup is jerky — the classic "many actuators + high torque limits → violent motion" failure. HoST's β + smoothness reward help but are brittle and need per-platform tuning. The LCP paper's own motivation is exactly this: smoothness rewards/low-pass filters require tedious, platform-specific hyperparameter tuning. A Lipschitz constraint addresses smoothness directly and transfers better.

**What it adds.** Cleaner, more natural motion; better sim2real (smooth policies transfer more reliably); less reward-weight babysitting. It is the cheapest high-signal change, and a smoother base policy makes every downstream curriculum easier to debug (noisy base policies make terrain/posture failures ambiguous).

**Implementation notes.**
- Add the gradient-penalty term to the PPO policy loss; keep β and smoothness as-is. Tune a single coefficient `λ_Lipschitz`.
- Touchpoint: the agent/PPO loss in the RL training code (not the env/reward configs).
- This is a training-loop change, **not** new simulation infrastructure → do it **first**, right after the 43-DOF baseline is stable.
- Ablate: same seed/config with and without LCP; compare action-rate spectra and torque profiles, not just success rate.

---

## Enhancement 2 — Balance-Embedding Asymmetric Critic (Capture Point / CoM / Centroidal Momentum)

**Reference.** Poddar, McCrory, Penco, et al., *Embedding Classical Balance Control Principles in Reinforcement Learning for Humanoid Recovery*, arXiv:2603.08619 (2026).

**Idea to take.** Feed **privileged physics** the robot cannot measure on hardware — **Capture Point (CP)**, **CoM state**, and **Centroidal Momentum** — into the **critic only** (asymmetric actor–critic), and shape reward around whether whole-body momentum is trending toward a recoverable/stable state. The **actor stays proprioception-only** for zero-shot hardware transfer.

**Why I want it.** Sparse task reward tells the policy *whether* it stood up, not *how close to recoverable* its current momentum is. The balance metrics give an explicit, dense "distance-to-stability" learning signal that lets the critic value good whole-body coordination. The paper's ablation is the key evidence: **removing the balance structure makes standup learning fail entirely** — so this is structural, not cosmetic. As a bonus, these metrics can **replace HoST's artificial vertical pulling force F**, which is cleaner for sim2real (no artificial force to wean the policy off).

**What it adds.** (a) Better whole-body coordination and contact sensibility; (b) the value signal that makes Enhancement 3 (obstacle reconfiguration) *learnable* — the critic can reward a body reconfiguration that doesn't immediately change task reward but improves recoverability; (c) a path to drop the artificial force F.

**Implementation notes.**
- **Asymmetric critic:** add CP/CoM/centroidal-momentum to the **critic observation group**, never the policy observation group (mjlab keeps these as separate obs groups).
- **Sensors you can reuse / add (XML `<sensor>` block, Part 1 Step 1b):**
  - `subtreeangmom` (`root_angmom`) — **already being added** → angular part of centroidal momentum.
  - add `subtreecom` → CoM position; add `subtreelinvel` → CoM linear velocity (linear momentum ∝ `m · v_com`).
- **Capture Point (LIP model):** `ξ_xy = x_com,xy + v_com,xy / ω`, with `ω = sqrt(g / z_com)`. Feed `ξ`, and the signed distance from `ξ` to the support polygon, to the critic.
- **Centroidal momentum:** `h = [k ; l]` (angular `k` from `subtreeangmom` about the CoM; linear `l = m · v_com`).
- **Reward shaping (new critic group):** reward CP moving toward the support region and momentum trending toward the upright recoverable set; map the new term in `reward_groups.py` (every active term must be mapped or `build_group_onehot` raises).
- **Phasing out F:** bring CP/momentum signals online first, then taper F. **Do not run F and the balance reward at full strength simultaneously** — they fight.
- Priority: **second** (right after LCP). Highest-value architectural add; everything contact-related depends on it.

---

## Enhancement 3 — Obstacle-Robust "Reconfigure-and-Retry" *(my idea — intuition kept, implementation evolved)*

**Supporting references.** HoST terrain training (Huang et al., arXiv:2502.08378); balance-embedding critic (arXiv:2603.08619); robust blind-locomotion-via-randomized-obstacles literature (policies learn recovery by repeatedly colliding with randomized obstacles under proprioception).

**The idea (preserved).** A truly real-world-robust humanoid should behave like a human: **when a movement is blocked by an obstacle, it should move *other* body parts to reach a new feasible configuration, then re-attempt the standup.** This reconfigure-and-retry behavior is the core capability I want, because it is what separates a demo-only standup from one that works "in every scenario."

**Evolution of the implementation (and why).** My original plan was to *detect* obstacles by comparing **desired vs actual joint torque**, and after `n` blocked steps trigger a "move other parts" response. I'm dropping that detector, for two concrete reasons:
1. **The signal is ambiguous.** Under PD control, a blocked joint shows up as position-tracking error + torque saturation — but that is *exactly* the normal signature of pushing against the **ground** during a legitimate standup. The robot pushes on things for almost the entire get-up. A torque-saturation detector therefore fires constantly and cannot separate "good contact (floor)" from "bad contact (obstacle)."
2. **A hand-coded detect→trigger state machine is brittle** — it needs a threshold `n`, a torque threshold, and a hand-designed "what to move next" response, none of which generalize.

**Robust formulation (keep the behavior, make it emergent).** Don't detect obstacles explicitly. Make the policy *intrinsically* robust so reconfigure-and-retry **emerges** as a continuous behavior (no mode switching):
1. **Randomize obstacles in the training distribution** — place small blocking geoms near the body at episode reset, on top of HoST's existing terrains (ground / platform / wall / slope). Curriculum on obstacle size/density (start trivial, scale up).
2. **Keep the sparse "reached standing" goal reward dominant** — this *forces* the policy to find a way around whatever blocks it. Avoid over-shaping that would let it ignore obstacles.
3. **Give the critic balance information (Enhancement 2)** — so it can value a reconfiguration that improves recoverability even when task reward hasn't moved yet. This is the mechanism that makes the behavior learnable.

**Optional blockage signal — as an *observation*, never a reward trigger.** If I want the policy to have explicit evidence of being stuck, feed **accumulated joint position-tracking error** (`pd_t − p_t` over a sliding window) into the **policy observation**. That gives proprioceptive evidence "I'm not going where I commanded," and lets the policy *learn* a response rather than me specifying one. The G1's current-based torque estimate can be an observation too — but **never** as a reward-triggering obstacle detector.

**What it adds.** The headline generalization improvement: genuine multi-scenario robustness, turning a brittle hand-coded rule into a learned, continuous behavior. It is also the **contact-minimization** story — combined with the hand-contact penalty (Part 1, Step 3) and balance critic, the policy learns to free itself with *appropriate, minimal* contact.

**Implementation / exploration notes.**
- Scene/terrain: extend the env scene config with randomized blocking objects at reset; reuse HoST's terrain-curriculum hooks for difficulty scheduling.
- Keep sparse goal reward weight high relative to shaping terms.
- Add position-tracking-error to the policy obs group (ablate on/off).
- Pair tightly with Enhancement 2 — do not attempt this *before* the balance critic exists, or the policy lacks the value signal to learn reconfiguration.
- Evaluate on **held-out obstacle configurations** never seen in training (this is the real test of "every scenario").
- Priority: **third**; sim-budget heavy (random obstacles + random falls enlarges the state space — expect more H200 hours and use a difficulty curriculum to keep exploration tractable).

---

## Enhancement 4 — Key State Initialization (KSI) for Posture Coverage

**Reference.** Chen, Wang, Luo, et al., *HiFAR: Multi-Stage Curriculum Learning for High-Dynamics Humanoid Fall Recovery*, arXiv:2502.20061 (IROS 2025).

**Idea to take.** Instead of seeding episodes only from random joint perturbations, **initialize from a curated set of representative keyframes** (prone / supine / side fall poses) so training directly covers the hard, structured starting configurations.

**Why I want it.** The "different starting positions" weakness comes from uniform random init under-sampling the structured fall postures that matter. Explicit keyframe init guarantees coverage and (per HiFAR) stabilizes and speeds convergence.

**What it adds.** Direct coverage of the posture-generalization gap, with faster/more stable training — addresses exactly the "works from any pose" requirement.

**Implementation notes.**
- Reuse the mjlab **keyframe machinery you already touch** in Part 1 Step 2d. Define a keyframe pose set spanning supine/prone/side (and partially-seated if relevant).
- Sample the initial state from this set at reset, with a curriculum (start near-supine → progressively add prone/side/asymmetric).
- Touchpoint: the reset / initial-state distribution in the get-up env config.
- Priority: **fourth**; parallelizable with Enhancement 3 (both modify the reset distribution / scene, so co-design them).

---

## Enhancement 5 — CBF-RL Safety Filtering *(gate before hardware, not a quality tool)*

**Reference.** Yang, Werner, de Sa, Ames, *CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions*, arXiv:2510.14959 (2025). Validated on **Unitree G1**.

**Idea to take.** Enforce **Control Barrier Functions during training** (active safety filtering of rollouts + barrier-inspired rewards) so the deployed policy **internalizes** safety and needs **no runtime filter**.

**Why I want it.** This is a **safety** tool, not a motion-quality tool — it will not make the standup better or more general, it will make it *safe to run on the real robot*. Before real-G1 trials I don't want exploratory standups self-colliding, hitting joint limits, or slamming the fragile articulated hands into the ground.

**What it adds.** Hardware-safe deployment: internalized self-collision / joint-limit / dangerous-configuration avoidance, with no runtime overhead. Complements (does not improve) motion quality.

**Implementation notes.**
- Define CBFs for: self-collision, joint-limit margins, and (relevant to the new fragile hands) hand–floor proximity.
- Add the barrier reward + filtered rollouts to the training loop.
- Touchpoint: RL training loop + safety-constraint definitions.
- Priority: **last** — gate this immediately before sim2real / hardware, not during the sim-quality phase.

---

## Logged for the future (out of current scope)

- **VIGOR** — *Visual Goal-In-Context Inference for Unified Humanoid Fall Safety*, arXiv:2602.16511 (2026). Vision-guided contact selection + anticipation. Defer until there is a concrete cluttered-perception requirement; high implementation cost, low value for the current proprioceptive standup goal.
- **Unified Fall-Safety Policy from a Few Demonstrations** — arXiv:2511.07407 (2025). "Recover *while falling*" (pre-impact regime), distinct from standup-from-ground. Future work; pairs naturally with VIGOR for the full fall→recover→stand continuum.
- **ASAP** — *Aligning Simulation and Real-World Physics*, arXiv:2502.01143 (RSS 2025). Residual delta-action model to close the sim-to-real contact gap; validated on G1. **Study now**, implement at the hardware-gap stage (i.e., if the policy works in sim but degrades on the real robot).

---

## Consolidated Roadmap

| Order | Enhancement | Closes which gap | Main touchpoints | Cost | Depends on |
|---|---|---|---|---|---|
| 0 | 43-DOF XML migration (Part 1) | correct model + fragile-hand contact | XML, `g1_constants.py`, `rewards.py` | medium | — |
| 1 | Lipschitz-Constrained Policies | motion quality (smoothness) | PPO/agent loss | low | baseline |
| 2 | Balance-embedding asymmetric critic | contact + coordination; drop force F | critic obs group, XML sensors, `reward_groups.py` | medium | LCP done |
| 3 | Obstacle-robust reconfigure-and-retry | generalization across obstacles/contacts | scene/terrain cfg, policy obs (tracking err) | high (sim) | Enh. 2 |
| 4 | Key State Initialization (HiFAR) | posture coverage | reset/init-state cfg, keyframes | medium (sim) | baseline (co-design w/ Enh. 3) |
| 5 | CBF-RL safety filtering | hardware safety | training loop, CBF defs | medium | before hardware only |

**Engineering hygiene reminder:** change one thing per training run where possible. Never combine the action-space migration (Part 1) with a new reward term or new critic input in the same run, or regressions become unattributable. Establish the 43-DOF Phase-1 baseline first, then introduce Enhancements 1→5 incrementally, ablating each against the previous checkpoint.

## Part 2 Progress Tracker

| # | Enhancement | Status |
|---|---|---|
| E0 | 43-DOF baseline trained & sane (Part 1 complete) | ☐ |
| E1a | Add Lipschitz gradient-penalty to PPO loss | ☐ |
| E1b | Ablate LCP vs β+smoothness (action-rate / torque profiles) | ☐ |
| E2a | Add `subtreecom` / `subtreelinvel` sensors | ☐ |
| E2b | Compute CP + centroidal momentum; add to **critic** obs only | ☐ |
| E2c | Add balance reward term + map to new critic group | ☐ |
| E2d | Taper / remove HoST vertical force F | ☐ |
| E3a | Randomized blocking obstacles at reset (+ difficulty curriculum) | ☐ |
| E3b | (Optional) position-tracking-error → policy obs | ☐ |
| E3c | Held-out obstacle-config evaluation | ☐ |
| E4a | Define supine/prone/side keyframe init set | ☐ |
| E4b | Curriculum'd Key State Initialization at reset | ☐ |
| E5 | CBF-RL safety filtering (pre-hardware gate) | ☐ |

---

## Reference List

- HoST — Huang et al., *Learning Humanoid Standing-up Control across Diverse Postures*, arXiv:2502.08378 (RSS 2025). Code: `github.com/OpenRobotLab/HoST`.
- LCP — Chen et al., *Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies*, arXiv:2410.11825 (IROS 2025). Code: `github.com/zixuan417/smooth-humanoid-locomotion`.
- Balance Principles — Poddar et al., *Embedding Classical Balance Control Principles in Reinforcement Learning for Humanoid Recovery*, arXiv:2603.08619 (2026).
- HiFAR — Chen et al., *HiFAR: Multi-Stage Curriculum Learning for High-Dynamics Humanoid Fall Recovery*, arXiv:2502.20061 (IROS 2025). Code: `github.com/Hi-FAR/HiFAR`.
- CBF-RL — Yang et al., *CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions*, arXiv:2510.14959 (2025).
- HumanUP — He et al., *Learning Getting-Up Policies for Real-World Humanoid Robots*, arXiv:2502.12152 (RSS 2025). Code: `github.com/RunpeiDong/humanup`.
- VIGOR — *Visual Goal-In-Context Inference for Unified Humanoid Fall Safety*, arXiv:2602.16511 (2026).
- Unified Fall-Safety — *Unified Humanoid Fall-Safety Policy from a Few Demonstrations*, arXiv:2511.07407 (2025).
- ASAP — He et al., *Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills*, arXiv:2502.01143 (RSS 2025). Code: `github.com/LeCAR-Lab/ASAP`.