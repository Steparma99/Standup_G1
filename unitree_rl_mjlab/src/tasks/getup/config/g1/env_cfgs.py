"""Unitree G1 get-up environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from src.tasks.getup.mdp.actions import LowPassJointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.robots import G1_ACTION_SCALE, get_g1_supine_robot_cfg
from src.assets.robots.unitree_g1.g1_constants import (
    PRONE_KEYFRAME,
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


def unitree_g1_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 29-DOF get-up task configuration."""
    cfg = make_getup_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 400  # increased for body-part contact sensors
    cfg.sim.nconmax = 100                  # increased: get-up generates many contacts
    cfg.sim.njmax = 450                    # observed overflow at 300 during violent early contacts

    # Robot spawns from SUPINE by default; the reset event below overrides the
    # pose every episode by sampling from the reference set.
    cfg.scene.entities = {"robot": get_g1_supine_robot_cfg()}

    # Reference poses for the multi-pose randomized reset. Add more poses here as
    # they are validated. To test a SINGLE fixed pose first, set this to
    # (SUPINE_KEYFRAME,) and zero the ranges in getup_env_cfg.py.
    cfg.events["reset_pose"].params["keyframes"] = (SUPINE_KEYFRAME, PRONE_KEYFRAME)

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

    # Viewer follows torso.
    cfg.viewer.body_name = "torso_link"

    # Wire up reward params that need robot-specific body/site names.
    cfg.rewards["torso_height_exp"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["stand_on_feet"].params["asset_cfg"].site_names = site_names
    cfg.rewards["feet_slip"].params["asset_cfg"].site_names = site_names
    cfg.rewards["feet_distance"].params["asset_cfg"].site_names = site_names
    cfg.rewards["com_over_support"].params["asset_cfg"].site_names = site_names

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
        cfg.curriculum = {}

    return cfg
