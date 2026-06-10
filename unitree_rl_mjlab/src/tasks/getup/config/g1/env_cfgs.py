"""Unitree G1 get-up environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from src.tasks.getup.mdp.actions import LowPassJointPositionActionCfg
from src.tasks.getup.mdp.events import AssistanceCurriculum
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
import src.tasks.getup.mdp as mdp

from src.assets.robots import G1_ACTION_SCALE, get_g1_supine_robot_cfg
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
# Assistance curriculum (HoST-style decaying upward support force on the torso).
# Bootstraps get-up learning: a per-env force helps the robot rise and decays
# toward zero as each env succeeds, so the help anneals away on its own.
# ---------------------------------------------------------------------------
_ASSIST_CURRICULUM_ENABLE      = True
_ASSIST_INITIAL_FORCE_N        = 120.0   # ~30% of G1 weight (~400 N); gentle lift
_ASSIST_FORCE_DECAY_PER_SUCCESS = 5.0    # N removed per successful episode (per env)
_ASSIST_FORCE_MIN              = 0.0     # fully unassisted floor
_ASSIST_SUCCESS_HEIGHT         = 0.65    # pelvis height counting as "stood" (task threshold)
_ASSIST_UNACTUATED_STEPS       = 15      # initial settle steps with no assist force

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
_SETTLE_STEPS = 10
_MASK_STEPS = 20


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
    # pose every episode by sampling from the reference set.
    cfg.scene.entities = {"robot": get_g1_supine_robot_cfg()}

    # Reference poses for the multi-pose randomized reset. Add more poses here as
    # they are validated. To test a SINGLE fixed pose first, set this to
    # (SUPINE_KEYFRAME,) and zero the ranges in getup_env_cfg.py.
    cfg.events["reset_pose"].params["keyframes"] = (
        SUPINE_KEYFRAME,
        PRONE_KEYFRAME,
        SIDE_LEFT_KEYFRAME,
        SIDE_RIGHT_KEYFRAME,
        SEATED_KEYFRAME,
    )

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

    # Set per-robot action scale.
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, LowPassJointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE
    # Settling phase: hold the policy for the first _SETTLE_STEPS steps so the robot
    # accommodates onto the floor before it takes control.
    joint_pos_action.settle_steps = _SETTLE_STEPS

    # Viewer follows torso.
    cfg.viewer.body_name = "torso_link"

    # Wire up reward params that need robot-specific body/site names.
    # (torso_height_exp removed: height/uprightness now live in the multiplicative
    # task_stand reward, which uses the pelvis root — no per-robot body needed.)
    cfg.rewards["stand_on_feet"].params["asset_cfg"].site_names = site_names
    cfg.rewards["feet_slip"].params["asset_cfg"].site_names = site_names
    cfg.rewards["feet_distance"].params["asset_cfg"].site_names = site_names
    cfg.rewards["com_over_support"].params["asset_cfg"].site_names = site_names
    cfg.rewards["supine_rising_prep"].params["asset_cfg"].site_names = site_names

    # standing_posture target = HOME standing pose, with per-group joint weights:
    # legs / waist / ankles HIGH (stance-critical), arms MEDIUM, wrists LOW
    # (the distal "hand" joints — keep their influence small).
    cfg.rewards["standing_posture"].params["target_joint_pos"] = dict(
        HOME_KEYFRAME.joint_pos
    )
    cfg.rewards["standing_posture"].params["joint_weights"] = {
        ".*_hip_.*": 1.0,
        ".*_knee_joint": 1.0,
        ".*_ankle_.*": 1.0,
        "waist_.*": 1.0,
        ".*_shoulder_.*": 0.3,
        ".*_elbow_joint": 0.3,
        ".*_wrist_.*": 0.1,
    }

    # Reward masking: fade in the head impact penalties over the first _MASK_STEPS
    # steps so the spawn-landing contact during the settling window is not penalized.
    cfg.rewards["head_contact_penalty"].params["ramp_steps"] = _MASK_STEPS
    cfg.rewards["head_impact_penalty"].params["ramp_steps"] = _MASK_STEPS

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
                "unactuated_steps": _ASSIST_UNACTUATED_STEPS,
            },
        )
        cfg.metrics["curriculum/assistance_force"] = MetricsTermCfg(
            func=mdp.assistance_force
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

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        # No support force at eval time: show the true, unaided policy.
        cfg.events.pop("assistance_curriculum", None)
        cfg.metrics.pop("curriculum/assistance_force", None)
        cfg.curriculum = {}

    return cfg
