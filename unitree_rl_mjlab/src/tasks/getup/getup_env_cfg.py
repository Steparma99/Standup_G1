"""Get-up task base configuration.

The robot starts lying on its back (supine) and must learn to stand up.
No velocity commands; no terrain; no gait phase — just rise.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.getup.mdp as mdp


_ACTOR_INCLUDE_BODY_HEIGHT = False
_ACTOR_INCLUDE_FEET_CONTACT = False
_ACTOR_INCLUDE_IMU_LIN_ACC = False

_HEAD_IMPACT_PENALTY_THRESHOLD = 250.0
_HEAD_IMPACT_PENALTY_SCALE = 250.0
_HEAD_IMPACT_TERMINATION_THRESHOLD = 1200.0

_FORBIDDEN_ACTOR_TERMS = {
    "base_height",
    "body_height",
    "head_height",
    "base_lin_vel",
    "whole_body_com",
    "feet_positions",
    "feet_contact",
    "contact_states",
    "contact_forces",
    "stage_gates",
    "foot_friction",
}


def _get_actor_obs_terms() -> dict[str, ObservationTermCfg]:
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "last_action": ObservationTermCfg(func=mdp.last_action),
        "pd_tracking_error": ObservationTermCfg(
            func=mdp.pd_tracking_error,
            params={"action_name": "joint_pos"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
    }

    # Validate the *mandatory*, deployable actor terms only. The optional terms
    # below are deliberate opt-ins (guarded by flags) and may intentionally tap
    # otherwise-privileged signals; they are appended after this check so they do
    # not trip the forbidden-term assert.
    _assert_actor_terms_no_privileged(actor_terms)

    if _ACTOR_INCLUDE_IMU_LIN_ACC:
        # IMU linear acceleration (specific force / g) — deployable on real G1.
        # Sensor: accelerometer at imu_in_pelvis in g1.xml.
        # Output: [B, 3] in units of g.  Static: ≈ -projected_gravity_b.
        # Dynamic component carries push-off / impact info not in gyro/proj_grav.
        #
        # Noise: ±0.5 g = ±4.9 m/s² (uniform). Real G1 BMI088 noise ≈ 0.2 m/s².
        # The extra margin closes the sim2real gap conservatively.
        #
        # TODO — IMU acc bias DR: add an events["imu_acc_bias"] EventTermCfg
        #   (mode="startup") that samples per-env offsets in ±0.05 g and stores
        #   them; a custom obs function would add the bias before returning.
        #   This simulates slowly-varying bias drift seen on real hardware.
        #
        # TODO — Low-pass filter: add a per-env EMA state (analogous to
        #   LowPassJointPositionAction) if raw noise causes training instability.
        actor_terms["imu_lin_acc"] = ObservationTermCfg(
            func=mdp.imu_lin_acc,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
    if _ACTOR_INCLUDE_BODY_HEIGHT:
        actor_terms["body_height"] = ObservationTermCfg(
            func=mdp.body_height,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
    if _ACTOR_INCLUDE_FEET_CONTACT:
        actor_terms["feet_contact"] = ObservationTermCfg(
            func=mdp.feet_contact,
            params={"sensor_name": "feet_ground_contact"},
        )

    return actor_terms


def _get_privileged_critic_obs_terms() -> dict[str, ObservationTermCfg]:
    return {
        "base_height": ObservationTermCfg(
            func=mdp.base_height,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "head_height": ObservationTermCfg(
            func=mdp.head_height,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
        ),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "whole_body_com": ObservationTermCfg(
            func=mdp.whole_body_com,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "feet_positions": ObservationTermCfg(
            func=mdp.feet_positions,
            params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
        ),
        "feet_contact": ObservationTermCfg(
            func=mdp.feet_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "feet_contact_forces": ObservationTermCfg(
            func=mdp.contact_force,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "head_contact_state": ObservationTermCfg(
            func=mdp.contact_state,
            params={"sensor_name": "contact_head"},
        ),
        "head_contact_force": ObservationTermCfg(
            func=mdp.contact_force,
            params={"sensor_name": "contact_head"},
        ),
        "torso_contact_state": ObservationTermCfg(
            func=mdp.contact_state,
            params={"sensor_name": "contact_torso"},
        ),
        "torso_contact_force": ObservationTermCfg(
            func=mdp.contact_force,
            params={"sensor_name": "contact_torso"},
        ),
        "knee_left_contact_state": ObservationTermCfg(
            func=mdp.contact_state,
            params={"sensor_name": "contact_knee_left"},
        ),
        "knee_left_contact_force": ObservationTermCfg(
            func=mdp.contact_force,
            params={"sensor_name": "contact_knee_left"},
        ),
        "knee_right_contact_state": ObservationTermCfg(
            func=mdp.contact_state,
            params={"sensor_name": "contact_knee_right"},
        ),
        "knee_right_contact_force": ObservationTermCfg(
            func=mdp.contact_force,
            params={"sensor_name": "contact_knee_right"},
        ),
        "forearm_left_contact_state": ObservationTermCfg(
            func=mdp.contact_state,
            params={"sensor_name": "contact_forearm_left"},
        ),
        "forearm_left_contact_force": ObservationTermCfg(
            func=mdp.contact_force,
            params={"sensor_name": "contact_forearm_left"},
        ),
        "forearm_right_contact_state": ObservationTermCfg(
            func=mdp.contact_state,
            params={"sensor_name": "contact_forearm_right"},
        ),
        "forearm_right_contact_force": ObservationTermCfg(
            func=mdp.contact_force,
            params={"sensor_name": "contact_forearm_right"},
        ),
        "hand_left_contact_state": ObservationTermCfg(
            func=mdp.contact_state,
            params={"sensor_name": "contact_hand_left"},
        ),
        "hand_left_contact_force": ObservationTermCfg(
            func=mdp.contact_force,
            params={"sensor_name": "contact_hand_left"},
        ),
        "hand_right_contact_state": ObservationTermCfg(
            func=mdp.contact_state,
            params={"sensor_name": "contact_hand_right"},
        ),
        "hand_right_contact_force": ObservationTermCfg(
            func=mdp.contact_force,
            params={"sensor_name": "contact_hand_right"},
        ),
        "stage_gates": ObservationTermCfg(
            func=mdp.stage_gates,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "foot_friction": ObservationTermCfg(
            func=mdp.foot_friction,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    geom_names=tuple(
                        f"{side}_foot{i}_collision"
                        for side in ("left", "right")
                        for i in range(1, 8)
                    ),
                )
            },
        ),
    }


def _get_critic_obs_terms() -> dict[str, ObservationTermCfg]:
    actor_terms = _get_actor_obs_terms()
    privileged_terms = _get_privileged_critic_obs_terms()
    # Actor terms flow into the critic via the merge below. If an optional actor
    # flag enables a term the privileged set also provides (e.g. feet_contact),
    # drop the privileged copy so it is not duplicated in the critic observation.
    privileged_terms = {
        k: v for k, v in privileged_terms.items() if k not in actor_terms
    }
    return {**actor_terms, **privileged_terms}


def _assert_actor_terms_no_privileged(actor_terms: dict[str, ObservationTermCfg]) -> None:
    overlap = _FORBIDDEN_ACTOR_TERMS & set(actor_terms)
    assert not overlap, f"Actor observations contain privileged terms: {sorted(overlap)}"


def make_getup_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base get-up task configuration."""

    ##
    # Observations
    ##

    actor_terms = _get_actor_obs_terms()
    critic_terms = _get_critic_obs_terms()

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            # Get-up contact transitions are partially observable; a short history
            # gives the policy temporal context (spec: very short history hurts).
            history_length=4,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
    }

    ##
    # Metrics (P1.2 stage/success/termination + P1.3 action/torque saturation)
    ##

    metrics = {
        # --- Legacy ---
        "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
        # --- P1.2: Stage occupancy ---
        "stage/stage0": MetricsTermCfg(func=mdp.stage0_fraction),
        "stage/stage1": MetricsTermCfg(func=mdp.stage1_fraction),
        "stage/stage2": MetricsTermCfg(func=mdp.stage2_fraction),
        "stage/stage3": MetricsTermCfg(func=mdp.stage3_fraction),
        # --- P1.2: Success metrics ---
        "success/candidate": MetricsTermCfg(func=mdp.success_candidate),
        "success/stable_hold": MetricsTermCfg(func=mdp.stable_success_active),
        "success/ever_stood": MetricsTermCfg(func=mdp.ever_stood_fraction),
        "success/fall_after_success": MetricsTermCfg(func=mdp.fall_after_success_active),
        # --- P1.2: Termination reason distribution ---
        "termination/timeout": MetricsTermCfg(func=mdp.termination_timeout),
        "termination/failure": MetricsTermCfg(func=mdp.termination_failure),
        "termination/fall_after_success": MetricsTermCfg(func=mdp.termination_fall_after_success),
        "termination/head_impact": MetricsTermCfg(
            func=mdp.termination_head_impact,
            params={
                "sensor_name": "contact_head",
                "force_threshold": _HEAD_IMPACT_TERMINATION_THRESHOLD,
            },
        ),
        # --- P1.3: Action saturation / smoothness ---
        "action/norm_mean": MetricsTermCfg(func=mdp.action_norm_mean),
        "action/saturation_fraction": MetricsTermCfg(func=mdp.action_saturation_fraction),
        "action/rate_mean": MetricsTermCfg(func=mdp.action_rate_mean),
        "action/acc_mean": MetricsTermCfg(func=mdp.action_acc_mean),
        # --- P1.3: Torque saturation ---
        "torque/norm_mean": MetricsTermCfg(
            func=mdp.torque_norm_mean,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "torque/saturation_fraction": MetricsTermCfg(
            func=mdp.torque_saturation_fraction,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "torque/power_mean": MetricsTermCfg(
            func=mdp.torque_power_mean,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        # --- P1.3: Stage-aware saturation ---
        "stage0/action_saturation": MetricsTermCfg(func=mdp.stage0_action_saturation),
        "stage3/action_saturation": MetricsTermCfg(func=mdp.stage3_action_saturation),
        "stage0/torque_saturation": MetricsTermCfg(
            func=mdp.stage0_torque_saturation,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "stage3/torque_saturation": MetricsTermCfg(
            func=mdp.stage3_torque_saturation,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        # --- Contact logging — sensor_name params set in robot-specific cfg ---
        # Feet: from existing feet_ground_contact (subtree, netforce).
        "contact/feet": MetricsTermCfg(
            func=mdp.contact_found,
            params={"sensor_name": "feet_ground_contact"},
        ),
        # Head — critical for detecting harmful impacts.
        "contact/head": MetricsTermCfg(
            func=mdp.contact_found,
            params={"sensor_name": "contact_head"},
        ),
        "contact/head_force_max": MetricsTermCfg(
            func=mdp.contact_force_max,
            params={"sensor_name": "contact_head"},
        ),
        "contact/head_impact": MetricsTermCfg(
            func=mdp.contact_impact,
            params={
                "sensor_name": "contact_head",
                "force_threshold": _HEAD_IMPACT_PENALTY_THRESHOLD,
            },
        ),
        # Torso + pelvis (back/hips on floor).
        "contact/torso": MetricsTermCfg(
            func=mdp.contact_found,
            params={"sensor_name": "contact_torso"},
        ),
        "contact/torso_force_max": MetricsTermCfg(
            func=mdp.contact_force_max,
            params={"sensor_name": "contact_torso"},
        ),
        # Knees (shin + linkage-brace capsules).
        "contact/knee_left": MetricsTermCfg(
            func=mdp.contact_found,
            params={"sensor_name": "contact_knee_left"},
        ),
        "contact/knee_right": MetricsTermCfg(
            func=mdp.contact_found,
            params={"sensor_name": "contact_knee_right"},
        ),
        # Forearms / elbows (elbow capsule + wrist capsule).
        "contact/forearm_left": MetricsTermCfg(
            func=mdp.contact_found,
            params={"sensor_name": "contact_forearm_left"},
        ),
        "contact/forearm_right": MetricsTermCfg(
            func=mdp.contact_found,
            params={"sensor_name": "contact_forearm_right"},
        ),
        "contact/forearm_left_force_max": MetricsTermCfg(
            func=mdp.contact_force_max,
            params={"sensor_name": "contact_forearm_left"},
        ),
        "contact/forearm_right_force_max": MetricsTermCfg(
            func=mdp.contact_force_max,
            params={"sensor_name": "contact_forearm_right"},
        ),
        # Hands (hand capsule on wrist_yaw_link).
        "contact/hand_left": MetricsTermCfg(
            func=mdp.contact_found,
            params={"sensor_name": "contact_hand_left"},
        ),
        "contact/hand_right": MetricsTermCfg(
            func=mdp.contact_found,
            params={"sensor_name": "contact_hand_right"},
        ),
        "contact/hand_left_force_max": MetricsTermCfg(
            func=mdp.contact_force_max,
            params={"sensor_name": "contact_hand_left"},
        ),
        "contact/hand_right_force_max": MetricsTermCfg(
            func=mdp.contact_force_max,
            params={"sensor_name": "contact_hand_right"},
        ),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        # Residual-on-default joint position target into the PD actuators, with an
        # EMA low-pass on the command (alpha; lower = smoother, more deployable).
        "joint_pos": mdp.LowPassJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.25,  # Override per-robot.
            use_default_offset=True,
            alpha=0.5,
        )
    }

    ##
    # Events
    ##

    events = {
        # Multi-pose randomized reset: pick a reference pose (SUPINE/PRONE/...)
        # at random, then perturb joints + root around it. `keyframes` is empty
        # here (robot-agnostic base) and MUST be set in the robot-specific config.
        # Set any range to (0, 0) — and keyframes to a single pose — to recover a
        # fixed-pose reset for validating the policy before enabling randomization.
        "reset_pose": EventTermCfg(
            func=mdp.reset_to_random_keyframe,
            mode="reset",
            params={
                "keyframes": (),  # Set per-robot, e.g. (SUPINE_KEYFRAME, PRONE_KEYFRAME).
                # Joint perturbation around the nominal pose, by body group.
                "joint_pos_range": {
                    ".*_hip_.*": (-0.5, 0.5),     # legs: generous
                    ".*_knee_.*": (-0.5, 0.5),
                    ".*_ankle_.*": (-0.5, 0.5),
                    "waist_.*": (-0.3, 0.3),      # waist/torso: moderate
                    ".*_shoulder_.*": (-0.5, 0.5),  # arms
                    ".*_elbow_.*": (-0.5, 0.5),
                    ".*_wrist_.*": (-0.5, 0.5),
                },
                # Root pose: small height + roll/pitch, full random yaw.
                "pose_range": {
                    "z": (-0.03, 0.03),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-3.14, 3.14),
                },
                # Random initial root velocity (so the robot isn't always at rest).
                "velocity_range": {
                    "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.2, 0.2),
                    "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
                },
            },
        ),
        # Resets per-env progress counters (prev_height, standing_counter,
        # ever_stood, fall_counter) for terminated environments.
        "reset_episode_state": EventTermCfg(
            func=mdp.reset_episode_state,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=mdp.dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=mdp.dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
                "operation": "add",
                "ranges": {
                    0: (-0.05, 0.05),
                    1: (-0.05, 0.05),
                    2: (-0.05, 0.05),
                },
            },
        ),
    }

    ##
    # Rewards
    ##

    rewards = {
        "base_height_exp": RewardTermCfg(
            func=mdp.base_height_exp,
            weight=2.0,
            params={
                "target_height": 0.728,  # G1 standing pelvis height
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "torso_height_exp": RewardTermCfg(
            func=mdp.torso_height_exp,
            weight=1.0,
            params={
                "target_height": 1.1,  # G1 torso_link height when standing
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
            },
        ),
        "body_up_exp": RewardTermCfg(
            func=mdp.body_up_exp,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "stand_on_feet": RewardTermCfg(
            func=mdp.stand_on_feet,
            weight=1.0,
            params={
                "sensor_name": "feet_ground_contact",
                "foot_height_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
        ),
        "dof_error_when_standing": RewardTermCfg(
            func=mdp.dof_error_when_standing,
            weight=-0.1,
            params={
                "pelvis_height_threshold": 0.6,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # P1.1: Stage-0 righting reward — encourages leaving flat prone/supine
        # pose toward upright. Gated to early stage only; turns off near standing.
        "prone_supine_righting": RewardTermCfg(
            func=mdp.prone_supine_righting,
            weight=0.5,
            params={
                "height_threshold": 0.40,
                "band": 0.10,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # CoM projection reward — MUST be stage-gated (height + upright + foot-contact).
        # Activates during Stage 1/2 (rising) and stays active during Stage 3.
        # Zero when the robot is flat on the floor so it never fights Stage-0 righting.
        # asset_cfg.site_names must be set per-robot (left_foot, right_foot).
        "com_over_support": RewardTermCfg(
            func=mdp.com_over_support,
            weight=1.0,
            params={
                "foot_sensor_name": "feet_ground_contact",
                "min_height": 0.25,     # gate starts at stage-1 entry
                "height_band": 0.15,    # fully on at 0.40 m (stage-2)
                "dist_scale": 5.0,      # exp(-5 * dist²): 0.1m → 0.95, 0.3m → 0.64
                "upright_lo": 0.3,      # -proj_grav_z lower bound (start ramp)
                "upright_hi": 0.7,      # -proj_grav_z upper bound (full ramp)
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
        ),
        # Dense progress signal: reward upward pelvis movement each step.
        # Crucial in early training when absolute-height rewards are near-zero.
        "height_progress": RewardTermCfg(
            func=mdp.height_progress,
            weight=0.5,
            params={"max_step": 0.05, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Reward staying above standing height for 50 consecutive steps (0.5 s).
        # Prevents the policy from learning to "touch" standing for one frame.
        "stable_success_hold": RewardTermCfg(
            func=mdp.stable_success_hold,
            weight=3.0,
            params={
                "n_hold": 50,
                "height_threshold": 0.65,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # --- Style rewards (stage 1-2: support transition / rising) ---
        "feet_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.2,
            params={
                "height_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
        ),
        "feet_distance": RewardTermCfg(
            func=mdp.feet_distance,
            weight=-0.5,
            params={
                "min_dist": 0.10,
                "max_dist": 0.50,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
        ),
        # Head usage should be discouraged at all times, but only violent impacts
        # should dominate the return / terminate the episode.
        "head_contact_penalty": RewardTermCfg(
            func=mdp.head_contact_penalty,
            weight=-0.5,
            params={"sensor_name": "contact_head"},
        ),
        "head_impact_penalty": RewardTermCfg(
            func=mdp.head_impact_penalty,
            weight=-2.0,
            params={
                "sensor_name": "contact_head",
                "force_threshold": _HEAD_IMPACT_PENALTY_THRESHOLD,
                "force_scale": _HEAD_IMPACT_PENALTY_SCALE,
            },
        ),
        # Penalize commanding targets outside the joint limits (actuator saturation).
        "action_saturation": RewardTermCfg(
            func=mdp.action_saturation,
            weight=-1.0,
            params={"action_name": "joint_pos", "asset_cfg": SceneEntityCfg("robot")},
        ),
        # --- Regularization (kept weak during discovery; raise for deployability) ---
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
        "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-5e-3),
        "joint_vel_l2": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-1e-3,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "joint_acc_l2": RewardTermCfg(
            func=mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "joint_torques_l2": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-2e-5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    }

    ##
    # Terminations: only timeout — bad_orientation excluded because robot starts lying
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan": TerminationTermCfg(func=mdp.nan_detection),
        # Safety nets — cannot fire on a healthy fresh episode (robot starts on
        # the ground at low speed), only on simulation blow-ups.
        "joint_vel_explosion": TerminationTermCfg(
            func=mdp.joint_velocity_explosion,
            params={"max_velocity": 50.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        "ground_penetration": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": -0.3, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Terminate if the robot falls after having stood: avoids wasting episode
        # time on a robot that succeeded and then collapsed.
        "standing_fall_timeout": TerminationTermCfg(
            func=mdp.standing_fall_timeout,
            params={
                "height_threshold": 0.65,
                "n_fall_steps": 30,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "head_impact": TerminationTermCfg(
            func=mdp.head_impact_termination,
            params={
                "sensor_name": "contact_head",
                "force_threshold": _HEAD_IMPACT_TERMINATION_THRESHOLD,
            },
        ),
    }

    ##
    # Assemble and return
    ##

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            # nconmax / njmax are sized per-robot in the robot config (env_cfgs.py),
            # because the get-up motion generates many simultaneous contacts.
            mujoco=MujocoCfg(
                timestep=0.002,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=5,  # control at 100Hz (0.002 * 5 = 0.01s)
        episode_length_s=10.0,
    )
