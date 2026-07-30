"""Unitree G1 get-up environment configuration."""

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from src.tasks.getup.mdp.actions import (
    HandHoldActionCfg,
    LowPassJointPositionActionCfg,
)
from src.tasks.getup.mdp.events import AssistanceCurriculum, BetaRescalerCurriculum
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
import src.tasks.getup.mdp as mdp

from src.assets.robots import get_g1_supine_robot_cfg_host
from src.assets.robots.unitree_g1.g1_constants import (
    HAND_HOLD_JOINT_POS,
    HOME_KEYFRAME,
    PRONE_KEYFRAME,
    # SEATED_KEYFRAME,  # temporarily disabled — not a static equilibrium without
    #                     back support (topples during settling); re-enable in the
    #                     reset keyframes below when the scene adds seated support.
    SIDE_LEFT_KEYFRAME,
    SIDE_RIGHT_KEYFRAME,
    SUPINE_KEYFRAME,
)
from src.tasks.getup.getup_env_cfg import make_getup_env_cfg

# ---------------------------------------------------------------------------
# P1.4/P1.5/P1.6 domain-randomization flags.
# Set these to True once the policy is stable on the fixed-baseline.
# ---------------------------------------------------------------------------
_DR_MOTOR_STRENGTH_ENABLE = False    # P1.4: scale effort limits per env
_DR_PD_GAINS_ENABLE       = False    # P1.5: scale Kp/Kd per env
_DR_ACTION_DELAY_ENABLE   = False    # P1.6: action lag buffer per env

# ---------------------------------------------------------------------------
# Pose perturbation. When False the robot spawns at the EXACT canonical
# keyframe pose (no joint noise, no root roll/pitch noise). The 4 ground-lying
# poses (SUPINE, PRONE, SIDE_LEFT, SIDE_RIGHT) are used for episode diversity
# (SEATED temporarily removed — see the reset keyframes below); this flag only
# controls whether random noise is added around each pose. Set to True once the
# policy reliably stands from all poses.
# v16: enabled — the supine getup is solid (stable_hold ~0.72 at v15/it11000), so
# the fixed-pose baseline has served its purpose. The default noise ranges from
# getup_env_cfg.py now apply (joint ±0.5 rad limbs / ±0.3 waist, root roll/pitch
# ±0.1, full random yaw).
# ---------------------------------------------------------------------------
_ADD_POSE_PERTURBATION = True

# ---------------------------------------------------------------------------
# Assistance curriculum (HoST-style decaying upward support force on the torso).
# Bootstraps get-up learning: a per-env force helps the robot rise and decays
# toward zero as each env succeeds, so the help anneals away on its own.
# ---------------------------------------------------------------------------
_ASSIST_CURRICULUM_ENABLE      = True
_ASSIST_INITIAL_FORCE_N        = 120.0   # ~30% of G1 weight (~400 N); gentle lift
# Slowed 5.0 -> 2.0: the previous runs collapsed because the force decayed toward 0
# faster than the shared policy internalised the UNAIDED stand (crutch-dependence:
# stable_hold peaked ~110 N then fell monotonically to it). Removing less force per
# held stand gives the policy many more episodes at each force level to adapt.
_ASSIST_FORCE_DECAY_PER_SUCCESS = 2.0    # N removed per HELD stand (per env)
_ASSIST_FORCE_MIN              = 0.0     # fully unassisted floor
_ASSIST_SUCCESS_HEIGHT         = 0.65    # pelvis height counting as "stood" (task threshold)
# Fix 1 — decay gate: the assist only retires after a GENUINE held stand (pelvis
# above success_height WHILE both feet planted, held this many consecutive control
# steps). Matches the stable_hold success metric (n_hold=50). Replaces the old
# partial-credit-on-peak-height gate that decayed the lift on one-frame touches.
_ASSIST_HOLD_STEPS             = 50
# Rate limiter: after an env retires 2 N of assist, it must wait this many
# full episodes before it is allowed to decay again. This caps the aggregate
# force-removal rate even when many envs are simultaneously succeeding.
_ASSIST_DECAY_COOLDOWN_EPISODES = 3
# Fix 2 — reversible crutch: after this many consecutive FAILED episodes (no held
# stand) an env's support is bumped back up by _ASSIST_RAMP_UP_STEP_N (clamped to
# the initial force), so a regressed env recovers instead of sitting at 0 forever.
_ASSIST_RAMP_UP_AFTER_FAILURES = 8       # was 20 — recover support faster so the
                                         # servo tracks true competence, not a ratchet
_ASSIST_RAMP_UP_STEP_N         = 10.0
_ASSIST_UNACTUATED_STEPS       = 60      # no assist force during the unactuated/settle
                                         # window (HoST uses unactuated_time=30 = 0.6 s
                                         # @ 50 Hz; v18 extended 30->40->60 = 1.2 s
                                         # alongside _SETTLE_STEPS). Matches _SETTLE_STEPS.

# ---------------------------------------------------------------------------
# Action-rescaler (beta) curriculum (HoST). beta is the action scale in
# p^d = q_default + beta * a. It starts at 1.0 (large authority while the policy
# explores on the floor) and is decremented by 0.02 every time an env's episode
# ends with head height >= _BETA_SUCCESS_HEAD_HEIGHT, down to a 0.25 floor (the
# exact value HoST's no-curriculum ablation uses). Per-env, success-driven — the
# SAME mechanism as the vertical assist force — so it is invariant to env count /
# episode length. The policy observes its beta (beta_rescaler obs term).
# ---------------------------------------------------------------------------
_BETA_CURRICULUM_ENABLE   = True
# v25 MANUAL CURRICULUM: freeze beta at its initial value (set per-run by
# GETUP_TRAIN_INITIAL_BETA) and step it down DELIBERATELY between runs, once the
# policy is demonstrably good at the current level. True = frozen (no automatic
# decay/ramp/breaker); flip to False to restore the automatic success-gated
# annealing. Introduced after the auto-curriculum decayed beta faster than the
# policy could adapt and collapsed the run (v25 Pass A).
_BETA_FREEZE              = True
_BETA_INITIAL             = 1.0
_BETA_DECREMENT           = 0.01   # was 0.02 — pace authority withdrawal with the
                                   # slower assist-force decay (avoid crutch-dependence)
_BETA_MIN                 = 0.25   # floor (HoST fixed-beta ablation value)
_BETA_SUCCESS_HEAD_HEIGHT = 0.75   # was 0.90 — lowered to H_STAGE2+0.10 so curriculum activates once robot learns to stand
# Fix 1 — beta only decays after a GENUINE held stand (head height above threshold
# WHILE both feet planted, held this many consecutive steps), not a one-frame spike.
# v25: 50 -> 15 to MATCH the stable_success_hold reward's n_hold. 15 steps (0.15 s
# @100 Hz) still rejects one-frame spikes (Fix 1's purpose) but stops beta's own
# "genuine stand" bar from being ~3x stricter than the reward that defines success —
# part of why beta_success_ema sat far below the reward's stable_hold and pinned the
# circuit breaker (below) in the paused state.
_BETA_HOLD_STEPS          = 15
# Fix 2 — restore authority (bump beta up) after this many consecutive failed
# episodes, so a regressed env is not locked at the low-authority floor forever.
_BETA_RAMP_UP_AFTER_FAILURES = 10   # was 20 — recover authority faster (matches assist)
_BETA_RAMP_UP_STEP        = 0.02
# Fix 3 — decay-cooldown rate limiter: after an env decrements beta, it must wait
# this many full episodes before it may decrement again. Caps the aggregate
# beta-withdrawal rate even when thousands of envs succeed simultaneously — the
# runaway feedback loop that collapsed v10_beta_home_anchor once beta fell below
# ~0.87. Widened 3 -> 6 after v11_beta_cooldown still bled (stable_hold
# 0.80 -> 0.49 as beta 0.90 -> 0.63): halve the maximum population decay rate so
# PPO gets roughly twice the gradient steps per unit of withdrawn authority.
_BETA_DECAY_COOLDOWN_EPISODES = 6
# Fix 4 — population circuit breaker (see BetaRescalerCurriculum). Fixes 1-3 are
# all per-env and could not stop v11's population-wide slow bleed: no single env
# failed long enough to trigger its own re-ramp, yet the shared policy was losing
# the skill as the mean beta fell. The breaker watches an EMA of the held-stand
# fraction across resetting envs and, when it drops below _BETA_PAUSE_BELOW,
# halts ALL beta decay and nudges beta back up each episode until the EMA
# recovers above _BETA_RESUME_ABOVE (hysteresis gap prevents flapping).
# v25 RECALIBRATION — the 0.65/0.75 pair assumed a healthy operating point where
# stable_hold plateaus at ~0.80. This run never gets there: stable_hold sits at
# ~0.28 and the genuine-stand EMA is lower still, so the breaker tripped to PAUSED
# on the very first resumed episode and NEVER recovered — with beta frozen at 1.0,
# the policy keeps full action noise, can't hold a quiet stand, so success can't
# rise above resume_above, so the breaker stays paused: a hard deadlock (confirmed
# by curriculum/beta_rescaler == 1.0000 for the entire v24 run). Lowered so the
# breaker permits decay at the CURRENTLY achievable success rate while still
# catching a true collapse toward zero. The 0.20 hysteresis gap prevents flapping.
# RAISE these back toward 0.55/0.70 once stable_hold climbs above ~0.6 (the
# collapse-protection regime the original values were meant for). Logged as
# curriculum/beta_success_ema and curriculum/beta_paused — watch both.
_BETA_PAUSE_BELOW    = 0.15
_BETA_RESUME_ABOVE   = 0.35
_BETA_RECOVERY_STEP  = 0.0025  # per-episode upward nudge while paused (~decrement/cooldown pace)
_BETA_EMA_ALPHA      = 0.5     # EMA time constant ~2 population-generations of episodes

# ---------------------------------------------------------------------------
# v22: HOME-pose tolerance bands (dead-zones).
# Per-joint HALF-WIDTH (radians) of a no-penalty window centred on the HOME
# target. Inside |q - HOME| <= band a joint pays no posture penalty / earns full
# posture reward; outside, the existing quadratic/exp shaping resumes unchanged.
# Rationale: the exact HOME keyframe may not be a holdable equilibrium, so the
# posture terms were fighting a policy trying to settle into a nearby stable
# stance (post_home_pose_l2 / post_upper_body_posture drifted WORSE across the
# whole v21 run). These bands are applied CONSISTENTLY to every joint-space
# HOME-pose term (post_home_pose_l2, post_standing_posture,
# post_upper_body_posture, post_bilateral_symmetry) so a genuine tolerance
# window actually exists — banding only some terms would leave the others still
# pulling to the exact point inside the window. Banding only ever SHRINKS each
# metric, so no term can grow in magnitude (penalties -> less negative, bounded
# rewards -> toward their existing max). Widen if the probe shows the policy
# still cannot settle; tighten if HOME conformance visibly degrades.
# _HOME_POSE_BANDS: regex keys (for the three regex-keyed terms).
_HOME_POSE_BANDS = {
    ".*_shoulder_pitch_joint": 0.15,
    ".*_shoulder_roll_joint": 0.15,
    ".*_shoulder_yaw_joint": 0.20,
    ".*_elbow_joint": 0.15,
    ".*_wrist_.*": 0.30,
    "waist_.*": 0.10,
    ".*_hip_pitch_joint": 0.12,
    ".*_hip_roll_joint": 0.12,
    ".*_hip_yaw_joint": 0.12,
    ".*_knee_joint": 0.12,
    ".*_ankle_pitch_joint": 0.10,
    ".*_ankle_roll_joint": 0.10,
}
# _HOME_POSE_BANDS_FAMILY: joint-FAMILY keys (post_bilateral_symmetry), same values.
_HOME_POSE_BANDS_FAMILY = {
    "shoulder_pitch": 0.15, "shoulder_roll": 0.15, "shoulder_yaw": 0.20,
    "elbow": 0.15, "wrist_roll": 0.30, "wrist_pitch": 0.30, "wrist_yaw": 0.30,
    "hip_pitch": 0.12, "hip_roll": 0.12, "hip_yaw": 0.12, "knee": 0.12,
    "ankle_pitch": 0.10, "ankle_roll": 0.10,
}

# ---------------------------------------------------------------------------
# Reset drop + settling phase.
#
# The keyframes (g1_constants.py) now spawn ~0.5 cm above their true resting height,
# so each pose starts essentially ON the floor. _RESET_FALL_HEIGHT adds a tiny extra
# drop: 0.03 m is chosen to exactly absorb the reset pose_range z of -0.03 m, so the
# WORST-CASE spawn never goes below the keyframe height (no nominal penetration),
# while the typical drop is only a few cm (gentle). (HoST drops ~0.4 m → torso
# impacts up to ~3.5 kN; we start near the ground instead.) Raise it if joint
# perturbation causes spawn penetration; lower it for an even gentler landing.
#
# During the SETTLING window (_SETTLE_STEPS env-steps) the policy does NOT control:
# the PD holds the current pose so MuJoCo resolves the spawn contacts and the robot
# accommodates onto the floor; residual velocity is zeroed when the window ends
# (events.settle_zero_velocity). The destructive impact penalties are faded in over
# _MASK_STEPS (rewards._reset_ramp) so the unavoidable landing contact is not
# penalized. Set _SETTLE_STEPS / _MASK_STEPS to 0 to disable each mechanism.
# step_dt = 0.01 s, so 10 steps = 0.1 s.
# ---------------------------------------------------------------------------
_RESET_FALL_HEIGHT = 0.03
# Unactuated/settle window = HoST's unactuated_time (30 steps = 0.6 s @ 50 Hz). The
# policy does NOT control for these 30 steps: the PD holds the CURRENT pose (a passive
# limp-settle) so MuJoCo fully resolves spawn contacts / interpenetration before the
# policy takes over, and the assist force is suppressed over the same window. (We keep
# the gentle 3 cm spawn drop rather than HoST's ~0.4 m drop — a big drop would worsen
# the very spawn-instability this window is meant to absorb. We also keep hold-current
# instead of HoST's zero-action, because here zero action targets default_joint_pos,
# which is far from the supine spawn and would yank the robot up instead of settling.)
# v18: 30 -> 40 (0.6 s -> 0.8 s) — with _ADD_POSE_PERTURBATION + PRONE the perturbed
# spawn poses were sometimes still visibly moving toward rest when the policy took
# over. Keep _ASSIST_UNACTUATED_STEPS (above) and getup_env_cfg._UNACTUATED_STEPS
# (termination grace windows) matched to this value.
# v18b: 40 -> 60 (0.8 s -> 1.2 s) — 40 still showed spawns visibly settling when the
# policy took over. Note velocity is force-zeroed exactly at window end
# (events.settle_zero_velocity), so if 60 still looks unsettled the culprit is
# contact/pose convergence or the velocity-snap discontinuity itself, not window
# length — investigate there instead of raising this further.
_SETTLE_STEPS = 60
_MASK_STEPS = 10

# ---------------------------------------------------------------------------
# Liftoff phase (v18b): for the first _LIFTOFF_STEPS right after the settle window
# the action filter runs with a REDUCED alpha and slew cap — the initial rise (leg
# drive / torso lift) is the fastest, jerkiest part of the motion and a flat
# whole-episode alpha/max_rate low enough to tame it would make the whole stand
# sluggish. Values then blend linearly back to the nominal alpha/max_rate over
# _LIFTOFF_RAMP_STEPS (a hard step back would itself be a jerk seam). Permanent
# structural behavior (part of the trained dynamics), not a training curriculum.
# alpha time constant at 50 Hz: nominal 0.12 -> ~156 ms; liftoff 0.06 -> ~330 ms.
# ---------------------------------------------------------------------------
_LIFTOFF_STEPS      = 30    # 0.6 s of slow, deliberate initial rise
_LIFTOFF_ALPHA      = 0.06  # half the nominal 0.12
_LIFTOFF_MAX_RATE   = 1.25  # rad/s, half the nominal 2.5 — still enough authority
                            # to drive the rise without stalling the liftoff
_LIFTOFF_RAMP_STEPS = 20    # 0.4 s linear blend back to nominal values


def unitree_g1_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 29-DOF get-up task configuration."""
    cfg = make_getup_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 400  # increased for body-part contact sensors
    # A G1 sprawled on the floor (head/torso/pelvis/2 knees/2 forearms/2 hands/2
    # feet, each a multi-point geom) can exceed 100 simultaneous contacts; a dropped
    # contact = a missing reward/force signal. Raised with matching njmax headroom
    # (~4 constraint rows per contact, pyramidal cone). Watch the run for overflow
    # warnings: lower if memory-bound, raise further if they appear.
    # v16: raised from 80/380 — with pose perturbation + the PRONE keyframe, an
    # 8-env smoke test undersold the real tail: a full 4096-env training run
    # logged sustained overflow up to njmax=720 (not a one-off spike — dozens of
    # occurrences in the 480-720 range), meaning contacts were being dropped on
    # essentially every iteration. Sized with real headroom over the observed
    # 720 peak this time. These buffers are allocated PER ENV, so this increase
    # (480->900, ~1.9x) will very likely OOM again at num_envs=4096 (130/650 —
    # a smaller 1.35x increase — already OOM'd once); LOWER --env.scene.num-envs
    # (e.g. 2048) on the next launch to compensate. Raise further only if fresh
    # overflow warnings appear at the new size; check GPU memory (nvidia-smi) /
    # lower num-envs further first if it OOMs again.
    cfg.sim.nconmax = 190
    cfg.sim.njmax = 900

    # Robot spawns from SUPINE by default; the reset event below overrides the
    # pose every episode by sampling from the reference set. Uses the HoST PD
    # variant (hip Kp=200, knee Kp=275; all other gains/damping unchanged).
    cfg.scene.entities = {"robot": get_g1_supine_robot_cfg_host()}

    # Ground-lying poses for episode diversity (SUPINE, PRONE, SIDE_LEFT,
    # SIDE_RIGHT). PRONE is important for real-world forward falls.
    # SEATED is TEMPORARILY REMOVED: without back support it is not a static
    # equilibrium — a settle probe showed it topples backward through the 30-step
    # settling window (projected-gravity z drifts from -0.97 toward ~0, i.e. it
    # ends up ~supine and still moving), so the policy inherited real angular
    # momentum on handover and "huge initial movement". It will be re-added once
    # the scene supports back support / props / varied terrain to make a seated
    # start statically stable.  Perturbation noise stays off until
    # _ADD_POSE_PERTURBATION=True.
    # SUPINE-ONLY baseline: collapse the reset to a single fixed start pose so the
    # policy learns a near-deterministic supine standup first (the robot spawns at
    # the exact same pose + yaw every episode). PRONE / SIDE_LEFT / SIDE_RIGHT are
    # temporarily removed — with stable_hold stuck at 0 across 1.3k iters, asking one
    # policy to solve four different getups (prone and side are much harder than
    # supine) was spreading the gradient too thin. Re-add poses ONE AT A TIME only
    # after the robot reliably stands and holds from supine.
    # v10: PRONE temporarily removed again — the v9 Pass 3 collapse traced to the
    # beta-anchor bug (see the action-anchor block below), but to validate that
    # fix on a single changed variable we go back to the known-stable SUPINE-only
    # baseline first. Re-add PRONE as its own isolated pass once the fixed beta
    # pipeline is confirmed stable.
    # v16: PRONE re-added — the beta-anchor bug that sank the previous PRONE
    # attempts was fixed in v10 (HOME anchor) and the supine getup is now stable
    # with beta frozen at 0.72, so this is the planned isolated re-introduction.
    # SIDE_LEFT / SIDE_RIGHT stay out until SUPINE+PRONE is validated.
    cfg.events["reset_pose"].params["keyframes"] = (SUPINE_KEYFRAME, PRONE_KEYFRAME)
    if not _ADD_POSE_PERTURBATION:
        cfg.events["reset_pose"].params["joint_pos_range"] = {}
        cfg.events["reset_pose"].params["pose_range"] = {}

    # Spawn each keyframe lifted by _RESET_FALL_HEIGHT so the robot falls a small
    # gentle distance onto the floor every reset (see constant above).
    cfg.events["reset_pose"].params["fall_height"] = _RESET_FALL_HEIGHT

    # ------------------------------------------------------------------
    # Settling phase (see the constants block above):
    #   1. policy held for _SETTLE_STEPS steps (action term, set below),
    #   2. residual velocity zeroed at the end of the window (this step-event),
    #   3. impact penalties faded in over _MASK_STEPS (reward params, set below),
    #   4. ALL rewards zeroed for the whole window (gate_settling_rewards step-event):
    #      the policy is passive during settling, so no reward it cannot influence
    #      should enter the training signal or the logged episode returns.
    # ------------------------------------------------------------------
    if _SETTLE_STEPS > 0:
        cfg.events["settle_zero_velocity"] = EventTermCfg(
            mode="step",
            func=mdp.settle_zero_velocity,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "settle_steps": _SETTLE_STEPS,
            },
        )
        cfg.events["gate_settling_rewards"] = EventTermCfg(
            mode="step",
            func=mdp.gate_settling_rewards,
            params={"settle_steps": _SETTLE_STEPS},
        )

    # Foot contact sensor — same as velocity task, used for stand_on_feet reward.
    site_names = ("left_foot", "right_foot")
    geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
    )

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # ------------------------------------------------------------------
    # Body-part contact sensors for detailed contact logging (P1.1 spec).
    #
    # Each sensor uses mode="geom" to target specific collision geoms,
    # and secondary=terrain to detect floor contact only.
    # Head uses reduce="netforce" because we care about total impact magnitude;
    # the others keep reduce="maxforce" for lighter-weight contact logging.
    #
    # Sensor shapes after build:
    #   found : [B, N_primaries]       (0=no contact, >0=count)
    #   force : [B, N_primaries, 3]    (force vector of max contact)
    # ------------------------------------------------------------------
    _terrain = ContactMatch(mode="body", pattern="terrain")

    contact_head_cfg = ContactSensorCfg(
        name="contact_head",
        primary=ContactMatch(mode="geom", pattern="head_collision", entity="robot"),
        secondary=_terrain,
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    # torso_collision (capsule along spine) + pelvis_collision (sphere at hips)
    contact_torso_cfg = ContactSensorCfg(
        name="contact_torso",
        primary=ContactMatch(
            mode="geom",
            pattern=("torso_collision", "pelvis_collision"),
            entity="robot",
        ),
        secondary=_terrain,
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
    )

    # Knee = shin capsule + linkage-brace capsule (both on left_knee_link)
    contact_knee_left_cfg = ContactSensorCfg(
        name="contact_knee_left",
        primary=ContactMatch(
            mode="geom",
            pattern=("left_shin_collision", "left_linkage_brace_collision"),
            entity="robot",
        ),
        secondary=_terrain,
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
    )
    contact_knee_right_cfg = ContactSensorCfg(
        name="contact_knee_right",
        primary=ContactMatch(
            mode="geom",
            pattern=("right_shin_collision", "right_linkage_brace_collision"),
            entity="robot",
        ),
        secondary=_terrain,
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
    )

    # Forearm = elbow capsule (left_elbow_link) + wrist capsule (left_wrist_pitch_link)
    contact_forearm_left_cfg = ContactSensorCfg(
        name="contact_forearm_left",
        primary=ContactMatch(
            mode="geom",
            pattern=("left_elbow_yaw_collision", "left_wrist_collision"),
            entity="robot",
        ),
        secondary=_terrain,
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
    )
    contact_forearm_right_cfg = ContactSensorCfg(
        name="contact_forearm_right",
        primary=ContactMatch(
            mode="geom",
            pattern=("right_elbow_yaw_collision", "right_wrist_collision"),
            entity="robot",
        ),
        secondary=_terrain,
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
    )

    # Hand = hand_collision capsule on wrist_yaw_link
    contact_hand_left_cfg = ContactSensorCfg(
        name="contact_hand_left",
        primary=ContactMatch(mode="geom", pattern="left_hand_collision", entity="robot"),
        secondary=_terrain,
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
    )
    contact_hand_right_cfg = ContactSensorCfg(
        name="contact_hand_right",
        primary=ContactMatch(mode="geom", pattern="right_hand_collision", entity="robot"),
        secondary=_terrain,
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        contact_head_cfg,
        contact_torso_cfg,
        contact_knee_left_cfg,
        contact_knee_right_cfg,
        contact_forearm_left_cfg,
        contact_forearm_right_cfg,
        contact_hand_left_cfg,
        contact_hand_right_cfg,
    )

    # HoST action scaling: p^d = q_default + beta * a, with a single scalar action
    # scale of 1.0 (the per-env beta curriculum supplies the actual rescaling,
    # 1.0 -> 0.25). This REPLACES the previous per-joint G1_ACTION_SCALE
    # (0.25*effort/stiffness); with beta=1.0 the residual is up to ±1 rad/joint, so
    # actions are more aggressive early and anneal down as each env succeeds.
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, LowPassJointPositionActionCfg)
    joint_pos_action.scale = 1.0
    # Motion-smoothness governor (distinct from beta). alpha is the EMA weight on
    # the freshly commanded PD target: q_cmd_t = alpha*desired + (1-alpha)*q_cmd_{t-1}.
    # It controls how FAST the target moves, whereas beta controls how FAR the
    # target can move from HOME. Lowering it 0.5 -> 0.25 roughly doubles the command
    # time constant (~29ms -> ~70ms at the 50 Hz control rate) so the get-up motion
    # is slower/less jerky. It does NOT shrink the reachable envelope (the EMA still
    # converges to the same desired target), only the transient rate. Watch the next
    # video: drop further toward 0.15 if still too fast, raise toward 0.35 if the
    # robot reacts too sluggishly to catch falls.
    # v16: 0.25 -> 0.15 — the v15 video was still too fast; command time constant
    # grows ~70ms -> ~123ms at 50 Hz. Raise back toward 0.25 if the robot can no
    # longer catch itself during the rise.
    # v17: 0.15 -> 0.12 — passB video still not smooth (note: that run also ran
    # with the slew cap loosened to 3.0 via env var, so re-judge smoothness with
    # BOTH this alpha and the default 2.5 rad/s cap before lowering further).
    # Time constant ~123ms -> ~156ms at 50 Hz.
    # v19: 0.12 -> 0.10 — v18b ran with the correct 2.5 baked cap (the re-judge
    # asked for above) and the rise is STILL faster than wanted; liftoff-phase
    # values (_LIFTOFF_ALPHA/_LIFTOFF_MAX_RATE) untouched — the early rise was
    # judged good in v18b, this slows the mid/late rise only.
    joint_pos_action.alpha = 0.10
    _override_alpha = os.environ.get("GETUP_ACTION_ALPHA")
    if _override_alpha not in (None, ""):
        joint_pos_action.alpha = float(_override_alpha)
        print(f"[getup] action low-pass alpha overridden to {joint_pos_action.alpha:.3f} "
              "(GETUP_ACTION_ALPHA)")
    # Slew-rate limit (rad/s) on the commanded target — bounds ABSOLUTE motion speed,
    # which alpha (a fraction-of-distance EMA) structurally cannot: the biggest joint
    # travel early in the rise stays fast even at low alpha, and anti_jump_velocity
    # confirmed base LINEAR speed is never the active constraint (stayed ~0 with its
    # gate wide open), so per-joint TARGET slew is the remaining lever for the
    # "too fast during the first part of the rise" symptom.
    # v16: promoted from opt-in test knob to a BAKED-IN default (2.5 rad/s) after
    # the v15 slew-rate test confirmed training stays healthy with the limiter on
    # — the limiter is part of the trained dynamics, never a deploy-only filter.
    # GETUP_ACTION_MAX_RATE still overrides for tuning (set 0 to get "no limit"
    # back via a huge value, e.g. 1e6).
    # v19: 2.5 -> 2.0 rad/s — paired with the alpha 0.12->0.10 cut above; the
    # slew cap is what bounds the big early-travel joints that alpha (fraction-
    # of-distance EMA) structurally cannot slow. Revert to 2.5 first if the
    # robot can no longer catch itself during the rise.
    joint_pos_action.max_rate = 2.0
    _override_max_rate = os.environ.get("GETUP_ACTION_MAX_RATE")
    if _override_max_rate not in (None, ""):
        joint_pos_action.max_rate = float(_override_max_rate)
        print(f"[getup] action slew-rate limit overridden to {joint_pos_action.max_rate:.3f} rad/s "
              "(GETUP_ACTION_MAX_RATE)")
    # Liftoff phase (v18b): slower alpha/slew for the first _LIFTOFF_STEPS after the
    # settle window, blending back to nominal over _LIFTOFF_RAMP_STEPS. See the
    # _LIFTOFF_* constants above for rationale/values.
    joint_pos_action.liftoff_steps = _LIFTOFF_STEPS
    joint_pos_action.liftoff_alpha = _LIFTOFF_ALPHA
    joint_pos_action.liftoff_max_rate = _LIFTOFF_MAX_RATE
    joint_pos_action.liftoff_ramp_steps = _LIFTOFF_RAMP_STEPS
    _override_liftoff_steps = os.environ.get("GETUP_ACTION_LIFTOFF_STEPS")
    if _override_liftoff_steps not in (None, ""):
        joint_pos_action.liftoff_steps = int(_override_liftoff_steps)
        print(f"[getup] liftoff steps overridden to {joint_pos_action.liftoff_steps} "
              "(GETUP_ACTION_LIFTOFF_STEPS; 0 disables the liftoff phase)")
    _override_liftoff_alpha = os.environ.get("GETUP_ACTION_LIFTOFF_ALPHA")
    if _override_liftoff_alpha not in (None, ""):
        joint_pos_action.liftoff_alpha = float(_override_liftoff_alpha)
        print(f"[getup] liftoff alpha overridden to {joint_pos_action.liftoff_alpha:.3f} "
              "(GETUP_ACTION_LIFTOFF_ALPHA)")
    _override_liftoff_max_rate = os.environ.get("GETUP_ACTION_LIFTOFF_MAX_RATE")
    if _override_liftoff_max_rate not in (None, ""):
        joint_pos_action.liftoff_max_rate = float(_override_liftoff_max_rate)
        print(f"[getup] liftoff slew-rate overridden to {joint_pos_action.liftoff_max_rate:.3f} rad/s "
              "(GETUP_ACTION_LIFTOFF_MAX_RATE)")
    # v10 beta-anchor fix: anchor the residual/beta scheme on the standing HOME
    # pose, NOT the entity's spawn default (= SUPINE, lying on the back). The
    # entity must keep spawning supine, but mjlab derives default_joint_pos from
    # the spawn keyframe, so without this override the beta curriculum's
    # p^d = q_default + beta*a was squeezing the reachable PD targets toward
    # LYING ON THE FLOOR as beta decayed 1.0 -> 0.25 — the exact opposite of the
    # HoST intent (shrink action authority around a good standing reference).
    # This was the mechanism behind the v9 Pass 3 collapse (beta and stable_hold
    # declined in lockstep from it~3682). HOME_KEYFRAME is the same pose the
    # posture rewards (post_standing_posture / post_upper_body_posture) already
    # pull toward, so action authority and reward shaping now agree.
    joint_pos_action.default_pos_override = dict(HOME_KEYFRAME.joint_pos)
    # Settling phase: hold the policy for the first _SETTLE_STEPS steps so the robot
    # accommodates onto the floor before it takes control.
    joint_pos_action.settle_steps = _SETTLE_STEPS
    # Fingers: PD-held at the G1 hold pose (policy action space excludes them).
    hand_hold_action = cfg.actions["hand_hold"]
    assert isinstance(hand_hold_action, HandHoldActionCfg)
    hand_hold_action.target_joint_pos = dict(HAND_HOLD_JOINT_POS)

    # Viewer follows torso.
    cfg.viewer.body_name = "torso_link"

    # Wire up reward params that need robot-specific body/site names (HoST reward set).
    # Head-height task uses torso_link as the head-height proxy (no separate head body).
    cfg.rewards["task_head_height"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["style_foot_displacement"].params["asset_cfg"].site_names = site_names
    cfg.rewards["style_foot_distance"].params["asset_cfg"].site_names = site_names
    cfg.rewards["supine_rising_prep"].params["asset_cfg"].site_names = site_names
    # Bridge reward needs the foot sites to check feet-planted (anti-jump gate).
    cfg.rewards["pelvis_height_bridge"].params["asset_cfg"].site_names = site_names
    # CoM-over-support uses the two foot sites for the support-polygon midpoint.
    cfg.rewards["com_over_support"].params["asset_cfg"].site_names = site_names

    # Post-task upper-body posture: target = HOME standing pose, masked to the
    # upper-body joints (arms + waist); legs/ankles get weight 0 (default).
    cfg.rewards["post_upper_body_posture"].params["target_joint_pos"] = dict(
        HOME_KEYFRAME.joint_pos
    )
    # Weight the joints that visually define a natural stance (shoulder pitch/roll,
    # elbow) over the ones that barely read on camera (wrists) or are already handled
    # by post_standing_posture's waist weight (2.0). The uniform 1.0 mask diluted the
    # normalized mean-error signal across 17 DOF, most of it on low-impact wrists.
    # v9 LEFT/RIGHT SPLIT (experiment): the right arm stayed behind the back through
    # v1-v8 while the left converged — HOME_KEYFRAME is correctly mirrored (checked
    # against the MJCF joint-range convention), so this is a policy/physical
    # asymmetry, not a target bug. Because the metric is a normalized MEAN, the
    # converged left arm masks the bad right arm; up-weighting the right joints puts
    # the residual gradient where the residual error is.
    # v9-symmetry: the v1-v8 asymmetric bump (right>left) may itself have taught the
    # policy an asymmetric objective, reinforcing the right-arm-behind-back rather than
    # curing it. HOME_KEYFRAME is confirmed correctly mirrored, so weight was the only
    # asymmetric input. Equalize both sides at the *higher* value (2.5) so the overall
    # pull toward HOME is not weakened vs the last 2000 iters — only the L/R imbalance
    # is removed. If the right arm still trails once assist force hits 0, that points to
    # a real physical/policy asymmetry (see plan Pass 2), not a weight artifact.
    cfg.rewards["post_upper_body_posture"].params["joint_weights"] = {
        ".*_shoulder_pitch_joint": 2.5,
        ".*_shoulder_roll_joint": 2.5,
        # Arm-behind-the-back is primarily a shoulder-orientation defect, not an elbow
        # one, so raise the yaw pull to make the residual arm error expensive in the
        # one term dedicated to upper-body HOME conformance.
        ".*_shoulder_yaw_joint": 1.5,
        ".*_elbow_joint": 2.5,
        ".*_wrist_.*": 0.3,
        # v16: 0.5 -> 1.5 — the final pose shows a torso-vs-legs twist (waist_yaw);
        # at 0.5 the waist barely registered in the normalized mean vs the 2.5 arms.
        "waist_.*": 1.5,
    }
    # v22: HOME dead-zone (see _HOME_POSE_BANDS) — tolerate a window around HOME
    # instead of pulling to the exact keyframe.
    cfg.rewards["post_upper_body_posture"].params["joint_bands"] = dict(_HOME_POSE_BANDS)
    # v9 Pass 2 (arm-weight bump): after Pass 1 (symmetric weights + assist force
    # decayed to ~0) the arm still trailed and this term kept WORSENING through the
    # whole 1200-it resume (-1.05 -> -1.23), while stable_hold/fall_after_success
    # stayed solid. Symmetric weights alone didn't fix it. Bump this term's weight
    # -2.5 -> -6.0 (base default at getup_env_cfg.py:980) to give the arm-to-HOME
    # gradient more authority against the other post_task terms. NOT a "huge" jump:
    # the exp-clamped form's worst-case transient scales linearly with weight
    # (already ~-48/step at -2.5), and this reward's own history shows big weight
    # jumps (-10->-20) gave diminishing returns. post_standing_posture (weight 10)
    # is deliberately left untouched — change one variable at a time. If it works,
    # dial back to a moderate ~-4.0 in a follow-up (a full revert risks drift-back).
    cfg.rewards["post_upper_body_posture"].weight = -6.0

    # Left/right arm posture diagnostic metrics (observation only — see metrics.py).
    # Same target pose + per-joint weights as post_upper_body_posture above, just
    # filtered to one side each, so the combined metric's L/R asymmetry (recurring
    # across runs on alternating sides — see comments above) can be read directly
    # instead of guessed at from video.
    if "posture/left_arm_error" in cfg.metrics:
        cfg.metrics["posture/left_arm_error"].params["target_joint_pos"] = dict(
            HOME_KEYFRAME.joint_pos
        )
        cfg.metrics["posture/left_arm_error"].params["joint_weights"] = {
            "left_shoulder_pitch_joint": 2.5,
            "left_shoulder_roll_joint": 2.5,
            "left_shoulder_yaw_joint": 1.5,
            "left_elbow_joint": 2.5,
            "left_wrist_.*": 0.3,
        }
    if "posture/right_arm_error" in cfg.metrics:
        cfg.metrics["posture/right_arm_error"].params["target_joint_pos"] = dict(
            HOME_KEYFRAME.joint_pos
        )
        cfg.metrics["posture/right_arm_error"].params["joint_weights"] = {
            "right_shoulder_pitch_joint": 2.5,
            "right_shoulder_roll_joint": 2.5,
            "right_shoulder_yaw_joint": 1.5,
            "right_elbow_joint": 2.5,
            "right_wrist_.*": 0.3,
        }

    # Full-body HOME pose L2 penalty (replaces exp-form standing_posture).
    # Direct quadratic penalty: constant gradient at any distance from HOME.
    # Leg joints highest weight (fix asymmetric stance / legs-one-over-other).
    # Waist high weight (fix crooked torso). Arms moderate (stop random flailing).
    cfg.rewards["post_standing_posture"].params["target_joint_pos"] = dict(
        HOME_KEYFRAME.joint_pos
    )
    cfg.rewards["post_standing_posture"].params["joint_weights"] = {
        # Legs — main fix for asymmetric / one-legged stance
        ".*_hip_pitch_joint": 3.0,
        ".*_hip_roll_joint": 3.0,   # "legs one over other" is a hip-roll deviation
        ".*_hip_yaw_joint": 2.0,
        ".*_knee_joint": 3.0,
        ".*_ankle_pitch_joint": 2.0,
        ".*_ankle_roll_joint": 1.0,
        # Waist — fix crooked torso. v16: 2.0 -> 4.0, now the TOP weight — the
        # residual final-pose defect is the torso not rotated in line with the
        # legs (waist_yaw twist), so the waist gradient must not be diluted by
        # the (already converged) high-weight legs in this normalized mean.
        "waist_.*": 4.0,
        # Arms — raised 1.5->2.5 / wrists 0.5->1.5 so the arm gradient is not drowned
        # out by the high-weight legs in this normalised metric (fixes arms trailing
        # behind the trunk). HOME sets shoulder_pitch/roll + elbow targets.
        # v9-symmetry: equalized L/R at the higher value (3.5) — see the matching note
        # on post_upper_body_posture above. Both terms must agree on how hard to pull
        # each arm, else they fight over the left-arm target.
        ".*_shoulder_.*": 3.5,
        ".*_elbow_joint": 3.5,
        ".*_wrist_.*": 1.5,
    }
    # v22: HOME dead-zone (see _HOME_POSE_BANDS).
    cfg.rewards["post_standing_posture"].params["joint_bands"] = dict(_HOME_POSE_BANDS)

    # v20: constant-gradient HOME pull (post_home_pose_l2) — same HOME target as
    # the two exp-form terms above, but RAW L2 (see getup_env_cfg.py comment).
    # Weights are ABSOLUTE (the metric is an unnormalized SUM, not a mean), so
    # they are deliberately smaller than the exp terms' relative weights: each
    # unit of weight here costs `weight * err^2` reward per step directly.
    # Arms/legs/waist at 1.0 = the joints where the stuck crooked poses live;
    # shoulder_yaw/ankles lower; wrists minimal (barely read on camera and the
    # exp terms already cover them).
    # v21: term weight -2 -> -4 (getup_env_cfg.py) AND arm/waist joint weights
    # 1.0 -> 2.0 — the v20 run logged this term FLAT at ~-1.4 for 1800 its while
    # the visible residual stayed in the arms/waist; this is the only
    # non-saturating HOME pull so it gets both the budget and the focus. Legs
    # stay at 1.0 (their error is already shaped by the high-weight exp terms).
    cfg.rewards["post_home_pose_l2"].params["target_joint_pos"] = dict(
        HOME_KEYFRAME.joint_pos
    )
    cfg.rewards["post_home_pose_l2"].params["joint_weights"] = {
        ".*_shoulder_pitch_joint": 2.0,
        ".*_shoulder_roll_joint": 2.0,
        # Concentrate the non-saturating HOME pull on the joints that define the two
        # visible residual defects: the trailing arm (shoulder yaw) and the crooked
        # lower-body line (hip/ankle frontal-transverse alignment).
        ".*_shoulder_yaw_joint": 1.5,
        ".*_elbow_joint": 2.0,
        ".*_wrist_.*": 0.2,
        "waist_.*": 2.0,
        ".*_hip_pitch_joint": 1.0,
        ".*_hip_roll_joint": 2.0,
        ".*_hip_yaw_joint": 1.5,
        ".*_knee_joint": 1.0,
        ".*_ankle_pitch_joint": 0.4,
        ".*_ankle_roll_joint": 1.0,
    }
    # v22: HOME dead-zone (see _HOME_POSE_BANDS).
    cfg.rewards["post_home_pose_l2"].params["joint_bands"] = dict(_HOME_POSE_BANDS)

    # v21: post_bilateral_symmetry REWRITTEN to HOME-anchored WORST-SIDE error
    # (see rewards.py docstring). It now needs the HOME target too — same dict
    # as the other posture terms so all HOME pulls agree. Keys are still joint
    # FAMILY names (no left_/right_ prefix — the function resolves both sides).
    # metric = max(weighted mean HOME error of left side, same for right side):
    # the good side can never mask the bad side, and the gradient always points
    # at HOME (the old L-vs-R form pulled the good limb toward the bad one).
    # Weights kept from v19 (arms 3.0, legs 2.5): with the max() form they set
    # WHICH joints define "worst side", not an L/R budget split.
    cfg.rewards["post_bilateral_symmetry"].params["target_joint_pos"] = dict(
        HOME_KEYFRAME.joint_pos
    )
    cfg.rewards["post_bilateral_symmetry"].params["joint_weights"] = {
        "shoulder_pitch": 3.0,
        "shoulder_roll": 3.0,
        "shoulder_yaw": 1.0,
        "elbow": 3.0,
        "wrist_roll": 0.5,
        "wrist_pitch": 0.5,
        "wrist_yaw": 0.5,
        "hip_pitch": 2.5,
        "hip_roll": 2.5,
        "hip_yaw": 2.5,
        "knee": 2.5,
        "ankle_pitch": 1.0,
        "ankle_roll": 1.0,
    }
    # v22: HOME dead-zone, family-keyed (see _HOME_POSE_BANDS_FAMILY).
    cfg.rewards["post_bilateral_symmetry"].params["joint_bands"] = dict(
        _HOME_POSE_BANDS_FAMILY
    )

    # -----------------------------------------------------------------------
    # v23: DISABLE the exact-HOME-pose imitation terms (do NOT delete — they and
    # their override blocks above stay intact as no-ops so HOME imitation can be
    # re-enabled in one line later). These four terms all pull every joint toward
    # the exact HOME keyframe; piled up over v16-v22 they created conflicting
    # gradients around a pose the body may not be able to hold (final pose wrong +
    # topples after standing). Replaced by the v23 FUNCTIONAL terms below.
    # -----------------------------------------------------------------------
    cfg.rewards["post_home_pose_l2"].weight = 0.0
    cfg.rewards["post_standing_posture"].weight = 0.0
    cfg.rewards["post_upper_body_posture"].weight = 0.0
    cfg.rewards["post_bilateral_symmetry"].weight = 0.0  # HOME-anchored — replaced by
    #                                                      relational joint_symmetry_mirror

    # v27: arms_in_front DISABLED (weight 0) — decomposed into the four task-space
    # arm terms below (arm_hand_box / arm_elbow_flexion / arm_overextension /
    # arm_clearance). Kept as a no-op per the v23 convention, not deleted. The
    # forward-only band it scored is now the X axis of arm_hand_box's 3D box.
    cfg.rewards["arms_in_front"].weight = 0.0
    cfg.rewards["arms_in_front"].params["asset_cfg"].site_names = (
        "left_palm", "right_palm",
    )

    # v27 palm-site wiring for the three arm terms that read the palm sites (ORDER:
    # left, right — arm_hand_box's signed lateral depends on site 0 = LEFT). The
    # elbow-flexion term uses joints only (no site wiring); shoulder/elbow bodies
    # resolve inside the reward via _cached_body_ids/_cached_joint_ids.
    _palm_sites = ("left_palm", "right_palm")
    cfg.rewards["arm_hand_box"].params["asset_cfg"].site_names = _palm_sites
    cfg.rewards["arm_overextension"].params["asset_cfg"].site_names = _palm_sites
    cfg.rewards["arm_clearance"].params["asset_cfg"].site_names = _palm_sites
    # v24: joint_symmetry_mirror REMOVED (no symmetry reward — see getup_env_cfg.py).

    # ===================================================================
    # v31: BANDED joint-space FINAL-POSE groups (pose_legs / pose_waist / pose_upper).
    # target = HOME_KEYFRAME (same pose the whole reward chain anchors on). Per-joint
    # weights + dead-zone bands (radians) per the group design: legs weight hip-roll/
    # yaw/knee (leg-crossing / stance / symmetry), moderate hip-pitch, WIDE ankle bands
    # (balance actuators); waist narrow roll/pitch, wider yaw; arms weight shoulders+
    # elbow, WIDE wrist bands. pose_upper also reads the palm sites (hand-workspace).
    # These REPLACE the disabled joint-space imitation terms and the weak task-space
    # arm bands (arm_hand_box/arm_elbow_flexion/arm_overextension now weight 0).
    # ===================================================================
    _home = dict(HOME_KEYFRAME.joint_pos)
    # -- LEGS --
    cfg.rewards["pose_legs"].params["target_joint_pos"] = _home
    cfg.rewards["pose_legs"].params["joint_weights"] = {
        ".*_hip_roll_joint": 3.0, ".*_hip_yaw_joint": 3.0, ".*_knee_joint": 3.0,
        ".*_hip_pitch_joint": 1.5, ".*_ankle_pitch_joint": 1.0, ".*_ankle_roll_joint": 1.0,
    }
    cfg.rewards["pose_legs"].params["joint_bands"] = {
        ".*_hip_roll_joint": 0.08, ".*_hip_yaw_joint": 0.08, ".*_knee_joint": 0.10,
        ".*_hip_pitch_joint": 0.12, ".*_ankle_pitch_joint": 0.15, ".*_ankle_roll_joint": 0.15,
    }
    # -- WAIST --
    cfg.rewards["pose_waist"].params["target_joint_pos"] = _home
    cfg.rewards["pose_waist"].params["joint_weights"] = {
        "waist_roll_joint": 3.0, "waist_pitch_joint": 3.0, "waist_yaw_joint": 1.5,
    }
    cfg.rewards["pose_waist"].params["joint_bands"] = {
        "waist_roll_joint": 0.06, "waist_pitch_joint": 0.06, "waist_yaw_joint": 0.15,
    }
    # -- UPPER (arms) --  target HOME: shoulder_pitch 0.20, roll ±0.20, elbow 0.60
    _arm_weights = {
        ".*_shoulder_pitch_joint": 3.0, ".*_shoulder_roll_joint": 3.0,
        ".*_shoulder_yaw_joint": 2.0, ".*_elbow_joint": 3.0,
        ".*_wrist_roll_joint": 1.0, ".*_wrist_pitch_joint": 1.0, ".*_wrist_yaw_joint": 1.0,
    }
    _arm_bands = {
        ".*_shoulder_pitch_joint": 0.15, ".*_shoulder_roll_joint": 0.15,
        ".*_shoulder_yaw_joint": 0.20, ".*_elbow_joint": 0.15,
        ".*_wrist_roll_joint": 0.30, ".*_wrist_pitch_joint": 0.30, ".*_wrist_yaw_joint": 0.30,
    }
    cfg.rewards["pose_upper"].params["target_joint_pos"] = _home
    cfg.rewards["pose_upper"].params["joint_weights"] = _arm_weights
    cfg.rewards["pose_upper"].params["joint_bands"] = _arm_bands
    cfg.rewards["pose_upper"].params["asset_cfg"].site_names = _palm_sites
    # pose_upper stashes its gated r_arms / r_hand halves each step → logged as the
    # metrics pose/upper_arms and pose/upper_hand (see metrics registration below), so
    # a stuck upper score is attributable to arm ANGLES vs hand PLACEMENT.

    # Both-feet grounded: needs the foot site names to check height.
    cfg.rewards["post_stand_on_feet"].params["asset_cfg"].site_names = site_names

    # v26 task-space terminal region: leg_width_ordering + post_terminal_core both
    # read the two foot SITES (left, right) — for the pelvis-yaw-frame lateral
    # ordering and the CoM-over-support / planted-feet checks. Order matters: the
    # signed separation is y_L - y_R, so site 0 must be the LEFT foot.
    cfg.rewards["leg_width_ordering"].params["asset_cfg"].site_names = site_names
    cfg.rewards["post_terminal_core"].params["asset_cfg"].site_names = site_names

    # GETUP_ANTI_JUMP_RAMP_FROM: opt-in override, extends anti_jump_velocity's gate
    # below its default floor (_POST_GATE_RAMP_FROM = H_STAGE1 = 0.45m). Below that
    # height — the entire supine-to-crouch phase — there is otherwise ZERO velocity
    # shaping anywhere in the reward stack; anti_jump_velocity is the only term that
    # penalizes base speed during the rise at all. Unset by default (identical to prior
    # behavior) — set for an ISOLATED test resume only, not on a live/in-progress run,
    # since this is a genuine reward-shape change (same risk class as the alpha edit
    # that caused the v13 resume shock) and needs a clean before/after comparison.
    # Example: GETUP_ANTI_JUMP_RAMP_FROM=0.15 makes the dead-zone-above-1.5m/s penalty
    # live almost from the floor instead of only from the crouch upward.
    _anti_jump_ramp_from = os.environ.get("GETUP_ANTI_JUMP_RAMP_FROM")
    if _anti_jump_ramp_from not in (None, ""):
        cfg.rewards["anti_jump_velocity"].params["ramp_from"] = float(_anti_jump_ramp_from)
        print(f"[getup] anti_jump_velocity ramp_from overridden to "
              f"{float(_anti_jump_ramp_from):.3f} (GETUP_ANTI_JUMP_RAMP_FROM)")

    # Domain randomization: mass offset on torso, friction on feet.
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.events["foot_friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
            "operation": "abs",
            "ranges": (0.3, 1.6),
            "shared_random": True,
        },
    )

    # ------------------------------------------------------------------
    # Assistance curriculum — decaying upward support force on the torso.
    # Registered as a per-step event (class-based: __call__ applies the force
    # every step, .reset() decays it for envs that stood). The metric
    # curriculum/assistance_force tracks the mean support force, which should
    # fall toward zero as the policy learns to stand unaided.
    # ------------------------------------------------------------------
    if _ASSIST_CURRICULUM_ENABLE:
        # The curriculum's per-env force tensor (events.py AssistanceCurriculum.__init__,
        # self._force = torch.full(..., self._initial_force)) is plain in-memory env
        # state, NOT part of the RSL-RL checkpoint — runner.load() only restores the
        # actor-critic, so a bare resume would restart the assist force at the initial
        # value (confirmed: pre-fix resumed runs' TensorBoard logs all show
        # assistance_force=120.0 at their first logged iteration; v7_arm_exp resumed
        # at it2000 with 120 N instead of the source run's decayed value).
        # scripts/train.py now handles this automatically on --agent.resume=True: it
        # reads the source run's logged assistance_force at the resumed checkpoint's
        # iteration and overrides initial_force_n before the env is built.
        # GETUP_TRAIN_INITIAL_ASSIST_FORCE remains as a manual override and takes
        # precedence over the automatic lookup, e.g.:
        #   GETUP_TRAIN_INITIAL_ASSIST_FORCE=70 bash launch.sh ... --agent.resume=True ...
        # Unlike the GETUP_EVAL_ASSIST_FORCE eval override below, both paths only change
        # the STARTING point — decay/ramp stay live so the curriculum keeps annealing
        # normally from there (eval instead freezes the force constant).
        _train_force = os.environ.get("GETUP_TRAIN_INITIAL_ASSIST_FORCE")
        _initial_force_n = (
            float(_train_force) if _train_force not in (None, "") else _ASSIST_INITIAL_FORCE_N
        )
        if _train_force not in (None, ""):
            print(f"[getup] TRAIN assist force curriculum starting at {_initial_force_n:.1f} N "
                  "(GETUP_TRAIN_INITIAL_ASSIST_FORCE override, continues decaying normally)")
        cfg.events["assistance_curriculum"] = EventTermCfg(
            mode="step",
            func=AssistanceCurriculum,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "torso_link",
                "initial_force_n": _initial_force_n,
                "force_decay_per_success": _ASSIST_FORCE_DECAY_PER_SUCCESS,
                "force_min": _ASSIST_FORCE_MIN,
                "success_height": _ASSIST_SUCCESS_HEIGHT,
                "hold_steps": _ASSIST_HOLD_STEPS,
                "decay_cooldown_episodes": _ASSIST_DECAY_COOLDOWN_EPISODES,
                "ramp_up_after_failures": _ASSIST_RAMP_UP_AFTER_FAILURES,
                "ramp_up_step_n": _ASSIST_RAMP_UP_STEP_N,
                "unactuated_steps": _ASSIST_UNACTUATED_STEPS,
                # Feet-planted qualifier: assist-decay credit only accrues while
                # both feet are genuinely planted, not on torso-lean height spikes.
                "feet_sensor_name": "feet_ground_contact",
                "foot_site_names": site_names,  # ("left_foot", "right_foot")
                "foot_height_threshold": 0.1,
            },
        )
        cfg.metrics["curriculum/assistance_force"] = MetricsTermCfg(
            func=mdp.assistance_force
        )

    # ------------------------------------------------------------------
    # Action-rescaler (beta) curriculum — HoST per-env action-scale annealing.
    # Class-based step event: __call__ tracks whether the episode reached the
    # success head height; reset() decrements beta (0.02) for envs that did, down
    # to the 0.25 floor. The action term and beta_rescaler obs read the live beta;
    # curriculum/beta_rescaler logs its per-env mean (should fall 1.0 -> 0.25).
    # ------------------------------------------------------------------
    if _BETA_CURRICULUM_ENABLE:
        # GETUP_TRAIN_INITIAL_BETA: manual override of the beta the curriculum
        # STARTS from (mirrors GETUP_TRAIN_INITIAL_ASSIST_FORCE above). Needed on
        # resume: the per-env beta tensor is plain env state, not part of the
        # RSL-RL checkpoint, so a bare resume restarts beta at _BETA_INITIAL.
        # Decay/ramp/breaker stay live from the given value (unlike the eval
        # --eval-beta pin, which freezes it).
        _train_beta = os.environ.get("GETUP_TRAIN_INITIAL_BETA")
        _initial_beta = (
            float(_train_beta) if _train_beta not in (None, "") else _BETA_INITIAL
        )
        # v20: GETUP_TRAIN_FREEZE_BETA REMOVED. The freeze env var (added v13,
        # 45aecfd) leaked across shell sessions and silently froze THREE runs that
        # were meant to anneal (v18, v18b, v19) — the var survives in the terminal
        # and any truthy leftover pinned beta with no error. Training beta now
        # ALWAYS anneals (decay/ramp/breaker live); a stationary-beta reconverge,
        # if ever needed again, must be a deliberate code change, not an env var.
        # The eval-side --eval-beta pin is unaffected (separate code path).
        if os.environ.get("GETUP_TRAIN_FREEZE_BETA") not in (None, ""):
            print("[getup] WARNING: GETUP_TRAIN_FREEZE_BETA is set but IGNORED "
                  "(freeze support removed in v20 — beta always anneals in training)")
        if _train_beta not in (None, ""):
            print(f"[getup] TRAIN beta curriculum starting at {_initial_beta:.3f} "
                  "(GETUP_TRAIN_INITIAL_BETA override)")
        if _BETA_FREEZE:
            print(f"[getup] beta curriculum FROZEN at {_initial_beta:.3f} "
                  "(_BETA_FREEZE=True — no auto decay/ramp/breaker; step down "
                  "manually via GETUP_TRAIN_INITIAL_BETA between runs)")
        _decrement = _BETA_DECREMENT
        _ramp_up_step = _BETA_RAMP_UP_STEP
        _pause_below = _BETA_PAUSE_BELOW
        _recovery_step = _BETA_RECOVERY_STEP
        cfg.events["beta_rescaler_curriculum"] = EventTermCfg(
            mode="step",
            func=BetaRescalerCurriculum,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "torso_link",
                "initial_beta": _initial_beta,
                "freeze": _BETA_FREEZE,
                "decrement": _decrement,
                "beta_min": _BETA_MIN,
                "success_head_height": _BETA_SUCCESS_HEAD_HEIGHT,
                "hold_steps": _BETA_HOLD_STEPS,
                "ramp_up_after_failures": _BETA_RAMP_UP_AFTER_FAILURES,
                "ramp_up_step": _ramp_up_step,
                "decay_cooldown_episodes": _BETA_DECAY_COOLDOWN_EPISODES,
                # Fix 4 — population circuit breaker (see constants above).
                "success_ema_alpha": _BETA_EMA_ALPHA,
                "pause_below": _pause_below,
                "resume_above": _BETA_RESUME_ABOVE,
                "recovery_step": _recovery_step,
                # Feet-planted qualifier: beta only decays on a GENUINE stand
                # (head height reached WHILE both feet planted), not a torso-lean
                # spike — same qualifier as the assistance curriculum above.
                "feet_sensor_name": "feet_ground_contact",
                "foot_site_names": site_names,  # ("left_foot", "right_foot")
                "foot_height_threshold": 0.1,
            },
        )
        cfg.metrics["curriculum/beta_rescaler"] = MetricsTermCfg(
            func=mdp.beta_rescaler_value
        )
        cfg.metrics["curriculum/beta_success_ema"] = MetricsTermCfg(
            func=mdp.beta_success_ema
        )
        cfg.metrics["curriculum/beta_paused"] = MetricsTermCfg(
            func=mdp.beta_paused
        )

    # ------------------------------------------------------------------
    # P1.4 — Motor strength randomization (scales effort limits per env)
    # Disabled by default; enable by setting _DR_MOTOR_STRENGTH_ENABLE=True
    # above. With range=[1.0,1.0] behaviour is identical to baseline.
    # ------------------------------------------------------------------
    if _DR_MOTOR_STRENGTH_ENABLE:
        cfg.events["motor_strength"] = EventTermCfg(
            mode="reset",
            func=dr.effort_limits,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "effort_limit_range": (0.90, 1.10),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

    # ------------------------------------------------------------------
    # P1.5 — PD gain randomization (scales Kp and Kd per env at reset)
    # Disabled by default. Requires actuator_gainprm/actuator_biasprm.
    # ------------------------------------------------------------------
    if _DR_PD_GAINS_ENABLE:
        cfg.events["pd_gains"] = EventTermCfg(
            mode="reset",
            func=dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": (0.85, 1.15),
                "kd_range": (0.85, 1.15),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

    # ------------------------------------------------------------------
    # P1.6 — Action delay (per-env random lag applied to position target)
    # Disabled by default. Requires the actuators to be wrapped with
    # DelayedActuatorCfg (done in get_g1_supine_robot_cfg_with_delay()).
    # When disabled the robot uses the standard (no-delay) actuator config.
    #
    # TODO: to activate, replace get_g1_supine_robot_cfg() above with
    # get_g1_supine_robot_cfg_with_delay() from g1_constants.py, then
    # set _DR_ACTION_DELAY_ENABLE = True.
    # ------------------------------------------------------------------
    if _DR_ACTION_DELAY_ENABLE:
        cfg.events["action_delay"] = EventTermCfg(
            mode="reset",
            func=dr.sync_actuator_delays,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "lag_range": (0, 3),  # physics steps; 0 = no delay, 3 = 3*2ms = 6ms
            },
        )

    if not play:
        # Default parallel environments for training. Override on the CLI with
        # --env.scene.num-envs. (Episode horizon is 500 steps: episode_length_s=10.0
        # at decimation=4 x 0.005s = 0.02s control period.)
        cfg.scene.num_envs = 4096

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        # Assist force at eval time.
        #  - Default (env var unset): pop the curriculum → 0 N, the true unaided policy
        #    (deployment target).
        #  - GETUP_EVAL_ASSIST_FORCE=<N>: PIN a constant N-Newton lift so the video
        #    reproduces the exact training conditions of a given checkpoint (the assist
        #    the policy actually had at that iteration). Set it to the run's
        #    curriculum/assistance_force at the checkpoint's iteration to make the video
        #    match the metrics. Decay/ramp are zeroed so the force stays constant.
        _eval_force = os.environ.get("GETUP_EVAL_ASSIST_FORCE")
        _eval_force_f = float(_eval_force) if _eval_force not in (None, "") else 0.0
        if _eval_force_f > 0.0 and "assistance_curriculum" in cfg.events:
            _p = cfg.events["assistance_curriculum"].params
            _p["initial_force_n"] = _eval_force_f
            _p["force_min"] = _eval_force_f
            _p["force_decay_per_success"] = 0.0
            _p["ramp_up_step_n"] = 0.0
            print(f"[getup] EVAL assist force pinned to {_eval_force_f:.1f} N "
                  "(video matches training conditions)")
        else:
            # No support force at eval time: show the true, unaided policy.
            cfg.events.pop("assistance_curriculum", None)
            cfg.metrics.pop("curriculum/assistance_force", None)
        # Pin beta to its converged floor (0.25 — HoST's fixed eval/ablation value)
        # instead of the training start of 1.0: the curriculum is per-env and does
        # not persist into a fresh eval env, so without this the policy would be
        # evaluated at full action authority it never deploys with.
        if "beta_rescaler_curriculum" in cfg.events:
            cfg.events["beta_rescaler_curriculum"].params["initial_beta"] = _BETA_MIN
            # Disable the population circuit breaker at eval time: pause_below=-1
            # can never trigger, so the recovery nudge cannot lift the pinned beta.
            cfg.events["beta_rescaler_curriculum"].params["pause_below"] = -1.0
            cfg.events["beta_rescaler_curriculum"].params["recovery_step"] = 0.0
        cfg.curriculum = {}

    return cfg
