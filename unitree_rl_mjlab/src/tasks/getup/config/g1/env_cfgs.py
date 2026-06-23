"""Unitree G1 get-up environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from src.tasks.getup.mdp.actions import LowPassJointPositionActionCfg
from src.tasks.getup.mdp.events import AssistanceCurriculum, BetaRescalerCurriculum
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
import src.tasks.getup.mdp as mdp

from src.assets.robots import get_g1_supine_robot_cfg_host
from src.assets.robots.unitree_g1.g1_constants import (
    HOME_KEYFRAME,
    PRONE_KEYFRAME,
    SEATED_KEYFRAME,
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
# keyframe pose (no joint noise, no root roll/pitch noise). All 5 canonical
# poses (SUPINE, PRONE, SIDE_LEFT, SIDE_RIGHT, SEATED) are always used for
# episode diversity; this flag only controls whether random noise is added
# around each pose. Set to True once the policy reliably stands from all poses.
# ---------------------------------------------------------------------------
_ADD_POSE_PERTURBATION = False

# ---------------------------------------------------------------------------
# Assistance curriculum (HoST-style decaying upward support force on the torso).
# Bootstraps get-up learning: a per-env force helps the robot rise and decays
# toward zero as each env succeeds, so the help anneals away on its own.
# ---------------------------------------------------------------------------
_ASSIST_CURRICULUM_ENABLE      = True
_ASSIST_INITIAL_FORCE_N        = 120.0   # ~30% of G1 weight (~400 N); gentle lift
_ASSIST_FORCE_DECAY_PER_SUCCESS = 5.0    # N removed per successful episode (per env)
_ASSIST_FORCE_MIN              = 0.0     # fully unassisted floor
_ASSIST_SUCCESS_HEIGHT         = 0.65    # pelvis height counting as "stood" (task threshold)
_ASSIST_PROGRESS_FLOOR         = 0.45   # min height for partial-credit decay; must be above
                                         # the robot's natural lying height (~0.20 m for PRONE
                                         # + fall_height) so a failed episode earns no credit
_ASSIST_UNACTUATED_STEPS       = 30      # no assist force during the unactuated/settle
                                         # window (HoST gates pull_force by
                                         # real_episode_length_buf > unactuated_time=30,
                                         # i.e. 30 steps = 0.6 s @ 50 Hz). Matches _SETTLE_STEPS.

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
_BETA_INITIAL             = 1.0
_BETA_DECREMENT           = 0.02   # removed per successful episode (per env)
_BETA_MIN                 = 0.25   # floor (HoST fixed-beta ablation value)
_BETA_SUCCESS_HEAD_HEIGHT = 0.90   # torso_link (head proxy) height counting as "stood up"

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
_SETTLE_STEPS = 30
_MASK_STEPS = 10


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
    cfg.sim.nconmax = 150
    cfg.sim.njmax = 600

    # Robot spawns from SUPINE by default; the reset event below overrides the
    # pose every episode by sampling from the reference set. Uses the HoST PD
    # variant (hip Kp=200, knee Kp=275; all other gains/damping unchanged).
    cfg.scene.entities = {"robot": get_g1_supine_robot_cfg_host()}

    # All 5 canonical poses for episode diversity (SUPINE, PRONE, SIDE_LEFT,
    # SIDE_RIGHT, SEATED). SEATED is the closest to standing and gives the
    # curriculum its earliest success signals; PRONE is important for real-world
    # forward falls. Perturbation noise is disabled until _ADD_POSE_PERTURBATION=True.
    cfg.events["reset_pose"].params["keyframes"] = (
        SUPINE_KEYFRAME,
        PRONE_KEYFRAME,
        SIDE_LEFT_KEYFRAME,
        SIDE_RIGHT_KEYFRAME,
        SEATED_KEYFRAME,
    )
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
    #   3. impact penalties faded in over _MASK_STEPS (reward params, set below).
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
    # Settling phase: hold the policy for the first _SETTLE_STEPS steps so the robot
    # accommodates onto the floor before it takes control.
    joint_pos_action.settle_steps = _SETTLE_STEPS

    # Viewer follows torso.
    cfg.viewer.body_name = "torso_link"

    # Wire up reward params that need robot-specific body/site names (HoST reward set).
    # Head-height task uses torso_link as the head-height proxy (no separate head body).
    cfg.rewards["task_head_height"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["style_foot_displacement"].params["asset_cfg"].site_names = site_names
    cfg.rewards["style_foot_distance"].params["asset_cfg"].site_names = site_names
    cfg.rewards["supine_rising_prep"].params["asset_cfg"].site_names = site_names

    # Post-task upper-body posture: target = HOME standing pose, masked to the
    # upper-body joints (arms + waist); legs/ankles get weight 0 (default).
    cfg.rewards["post_upper_body_posture"].params["target_joint_pos"] = dict(
        HOME_KEYFRAME.joint_pos
    )
    cfg.rewards["post_upper_body_posture"].params["joint_weights"] = {
        ".*_shoulder_.*": 1.0,
        ".*_elbow_joint": 1.0,
        ".*_wrist_.*": 1.0,
        "waist_.*": 1.0,
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
        # Waist — fix crooked torso
        "waist_.*": 2.0,
        # Arms — stop random movement; HOME sets shoulder_pitch/roll + elbow targets
        ".*_shoulder_.*": 1.5,
        ".*_elbow_joint": 1.5,
        ".*_wrist_.*": 0.5,
    }

    # Both-feet grounded: needs the foot site names to check height.
    cfg.rewards["post_stand_on_feet"].params["asset_cfg"].site_names = site_names

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
        cfg.events["assistance_curriculum"] = EventTermCfg(
            mode="step",
            func=AssistanceCurriculum,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "torso_link",
                "initial_force_n": _ASSIST_INITIAL_FORCE_N,
                "force_decay_per_success": _ASSIST_FORCE_DECAY_PER_SUCCESS,
                "force_min": _ASSIST_FORCE_MIN,
                "success_height": _ASSIST_SUCCESS_HEIGHT,
                "progress_floor": _ASSIST_PROGRESS_FLOOR,
                "unactuated_steps": _ASSIST_UNACTUATED_STEPS,
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
        cfg.events["beta_rescaler_curriculum"] = EventTermCfg(
            mode="step",
            func=BetaRescalerCurriculum,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "torso_link",
                "initial_beta": _BETA_INITIAL,
                "decrement": _BETA_DECREMENT,
                "beta_min": _BETA_MIN,
                "success_head_height": _BETA_SUCCESS_HEAD_HEIGHT,
            },
        )
        cfg.metrics["curriculum/beta_rescaler"] = MetricsTermCfg(
            func=mdp.beta_rescaler_value
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
        # No support force at eval time: show the true, unaided policy.
        cfg.events.pop("assistance_curriculum", None)
        cfg.metrics.pop("curriculum/assistance_force", None)
        # Pin beta to its converged floor (0.25 — HoST's fixed eval/ablation value)
        # instead of the training start of 1.0: the curriculum is per-env and does
        # not persist into a fresh eval env, so without this the policy would be
        # evaluated at full action authority it never deploys with.
        if "beta_rescaler_curriculum" in cfg.events:
            cfg.events["beta_rescaler_curriculum"].params["initial_beta"] = _BETA_MIN
        cfg.curriculum = {}

    return cfg
