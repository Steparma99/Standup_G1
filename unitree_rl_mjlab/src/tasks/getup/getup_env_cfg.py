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
_ACTOR_INCLUDE_IMU_LIN_ACC = True

# --- Pre-normalization clip bounds (protect the empirical normalizer from
# outliers / sim blow-ups). Applied at the term level, BEFORE the network's
# running mean/std. Symmetric (±value); generous enough not to touch normal
# motion, tight enough to cap explosions. Tune per observed ranges. ---
_CLIP_JOINT_VEL = 30.0   # rad/s  (joint_vel_explosion terminates at 50)
_CLIP_ANG_VEL = 20.0     # rad/s  (pelvis gyro)
_CLIP_PD_ERR = 6.0       # rad    (target - measured)
_CLIP_IMU_ACC = 6.0      # g      (static ±1 g; impact peaks ~3-5 g; matches a
                         #         realistic BMI088 ±6 g full-scale range)

# --- Accelerometer (IMU linear acceleration) sensor model. See mdp.ImuLinAcc. ---
_IMU_LP_ALPHA = 0.3      # EMA low-pass coefficient (lower = smoother)
_IMU_BIAS_STD = 0.02     # per-axis bias 1-sigma [g], resampled each episode
_IMU_BIAS_CLIP = 0.05    # hard clip on sampled bias [g]
_IMU_ACC_NOISE = 0.5     # white-noise half-range [g] (conservative sim2real margin)
_IMU_DELAY_MAX_LAG = 1   # sensor latency in control steps (0 = none; 1 = 10 ms)

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
            clip=(-_CLIP_ANG_VEL, _CLIP_ANG_VEL),
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
            clip=(-_CLIP_JOINT_VEL, _CLIP_JOINT_VEL),
        ),
        "last_action": ObservationTermCfg(func=mdp.last_action),
        "pd_tracking_error": ObservationTermCfg(
            func=mdp.pd_tracking_error,
            params={"action_name": "joint_pos"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-_CLIP_PD_ERR, _CLIP_PD_ERR),
        ),
    }

    # Validate the *mandatory*, deployable actor terms only. The optional terms
    # below are deliberate opt-ins (guarded by flags) and may intentionally tap
    # otherwise-privileged signals; they are appended after this check so they do
    # not trip the forbidden-term assert.
    _assert_actor_terms_no_privileged(actor_terms)

    if _ACTOR_INCLUDE_IMU_LIN_ACC:
        # IMU linear acceleration (specific force / g) — deployable on real G1.
        # Sensor: accelerometer at imu_in_pelvis in g1.xml. Output [B, 3] in g.
        # Static ≈ -projected_gravity_b; the DYNAMIC component carries push-off /
        # impact info not present in the gyro or projected gravity.
        #
        # Modelled as a real accelerometer via the stateful mdp.ImuLinAcc term:
        #   func : bias (per-env, drifts each episode) + EMA low-pass on true/g
        #   noise: ±0.5 g white noise (real BMI088 ≈ 0.2 m/s²; margin for sim2real)
        #   clip : ±4 g saturation BEFORE the network's running normalizer
        #   delay: up to _IMU_DELAY_MAX_LAG control steps of bus/read latency
        # The CRITIC instead observes the CLEAN ground-truth accelerometer — see
        # the override in _get_critic_obs_terms.
        actor_terms["imu_lin_acc"] = ObservationTermCfg(
            func=mdp.ImuLinAcc,
            params={
                "sensor_name": "robot/imu_lin_acc",
                "lp_alpha": _IMU_LP_ALPHA,
                "bias_std": _IMU_BIAS_STD,
                "bias_clip": _IMU_BIAS_CLIP,
            },
            noise=Unoise(n_min=-_IMU_ACC_NOISE, n_max=_IMU_ACC_NOISE),
            clip=(-_CLIP_IMU_ACC, _CLIP_IMU_ACC),
            delay_min_lag=0,
            delay_max_lag=_IMU_DELAY_MAX_LAG,
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
        # Applied (post-EMA, post-settle) PD position target actually commanded.
        # Combined with the actor's raw `last_action` and measured `joint_pos`,
        # this lets the critic reconstruct the full raw→filter→joint command chain
        # and judge whether the policy's action becomes the correct physical command.
        "applied_joint_target": ObservationTermCfg(
            func=mdp.applied_joint_target,
            params={"action_name": "joint_pos"},
        ),
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
    critic_terms = {**actor_terms, **privileged_terms}
    # The critic must see the CLEAN ground-truth accelerometer, not the actor's
    # bias/low-pass/noise-corrupted model. Override the merged-in actor term with
    # the plain imu_lin_acc function (corruption is already stripped for the critic
    # group, but bias+LP live inside ImuLinAcc, so we must swap the func itself).
    if "imu_lin_acc" in critic_terms:
        critic_terms["imu_lin_acc"] = ObservationTermCfg(
            func=mdp.imu_lin_acc,
            clip=(-_CLIP_IMU_ACC, _CLIP_IMU_ACC),
        )
    return critic_terms


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
            # First massive run: log the term + env ids of any NaN/Inf, then
            # sanitize to 0 (per-term check). Switch to "sanitize" for silent
            # production or "error" for strict debugging once stable.
            nan_policy="warn",
            nan_check_per_term=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
            nan_policy="warn",
            nan_check_per_term=True,
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
        "action/raw_min": MetricsTermCfg(func=mdp.raw_action_min),
        "action/raw_max": MetricsTermCfg(func=mdp.raw_action_max),
        "action/raw_clip_fraction": MetricsTermCfg(func=mdp.raw_action_clip_fraction),
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
        "target/clamp_fraction": MetricsTermCfg(
            func=mdp.joint_target_clamp_fraction,
            params={"action_name": "joint_pos", "asset_cfg": SceneEntityCfg("robot")},
        ),
        "target/over_limit_mean": MetricsTermCfg(
            func=mdp.joint_target_over_limit_mean,
            params={"action_name": "joint_pos", "asset_cfg": SceneEntityCfg("robot")},
        ),
        "target/over_limit_max": MetricsTermCfg(
            func=mdp.joint_target_over_limit_max,
            params={"action_name": "joint_pos", "asset_cfg": SceneEntityCfg("robot")},
        ),
        "pd_tracking/mean": MetricsTermCfg(
            func=mdp.pd_tracking_error_mean,
            params={"action_name": "joint_pos", "asset_cfg": SceneEntityCfg("robot")},
        ),
        "pd_tracking/max": MetricsTermCfg(
            func=mdp.pd_tracking_error_max,
            params={"action_name": "joint_pos", "asset_cfg": SceneEntityCfg("robot")},
        ),
        "pd_tracking/l2_mean": MetricsTermCfg(
            func=mdp.pd_tracking_error_l2_mean,
            params={"action_name": "joint_pos", "asset_cfg": SceneEntityCfg("robot")},
        ),
        "pd_tracking/standing_mean": MetricsTermCfg(
            func=mdp.pd_tracking_error_standing_mean,
            params={"action_name": "joint_pos", "asset_cfg": SceneEntityCfg("robot")},
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
                # HoST resets AT REST: no random root velocity. The reset dynamics
                # come from the drop (fall_height, set per-robot in env_cfgs.py),
                # not from an injected velocity. Keep these at 0 to match HoST.
                "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
                # HoST-style drop height (m) added to every spawn; set per-robot.
                "fall_height": 0.0,
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
        # True torso uprightness (exp(-k·(1+proj_grav_z))): ~0 from any flat pose,
        # sharp gradient only as the torso verticalises. weight 2.0 → max 2.0/step.
        "body_up_exp": RewardTermCfg(
            func=mdp.body_up_exp,
            weight=2.0,
            params={"k": 4.0, "asset_cfg": SceneEntityCfg("robot")},
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
        # Final standing posture: soft (exp), group-weighted attraction toward the
        # HOME standing pose, gated to near-standing (>~0.6m). Replaces the old
        # dof_error_when_standing, which tracked the SUPINE default pose and thus
        # pulled the standing posture toward lying down. target_joint_pos and
        # joint_weights are robot-specific and set per-robot (env_cfgs.py).
        "standing_posture": RewardTermCfg(
            func=mdp.standing_posture,
            weight=0.5,
            params={
                "target_joint_pos": {},  # Set per-robot (HOME pose).
                "joint_weights": {},     # Set per-robot (legs/waist high, arms/wrist low).
                "pelvis_height_threshold": 0.6,
                "band": 0.12,
                "kp": 2.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # Stage-0 righting, split into two distinct, non-redundant signals:
        #  - prone_recovery: PROGRESS reward for actively rolling off the face
        #    (pg_x decreasing). Parking pays ~0; only the act of turning over.
        "prone_recovery": RewardTermCfg(
            func=mdp.prone_recovery,
            weight=0.5,
            params={
                "height_threshold": 0.40,
                "band": 0.10,
                "max_step": 0.05,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        #  - supine_rising_prep: rewards a supine robot for sitting up (torso
        #    verticalising) and bringing its CoM over the feet — a config useful
        #    for rising, NOT merely lying on the back. site_names set per-robot.
        "supine_rising_prep": RewardTermCfg(
            func=mdp.supine_rising_prep,
            weight=0.5,
            params={
                "height_threshold": 0.40,
                "band": 0.10,
                "dist_scale": 3.0,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
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
        # Normalised to [0,1] (delta/max_step) so it actually contributes; weight
        # raised 0.5→1.0. Crucial early when absolute-height rewards are near-zero.
        "height_progress": RewardTermCfg(
            func=mdp.height_progress,
            weight=1.0,
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
                "sensor_name": "feet_ground_contact",  # real contact, not height
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
        # Mechanical power Σ|τ·q̇|: captures energy spent under load that torque
        # norm alone misses (static holds are cheap, motion under load is not).
        "joint_power_l2": RewardTermCfg(
            func=mdp.joint_power_l2,
            weight=-1e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        # --- Post-task (HoST-style): shape the held standing state, gated to
        # standing & upright so they never fight the rise. ---
        "base_ang_vel_penalty": RewardTermCfg(
            func=mdp.base_ang_vel_penalty,
            weight=-0.05,
            params={"height_threshold": 0.65, "asset_cfg": SceneEntityCfg("robot")},
        ),
        "base_lin_vel_penalty": RewardTermCfg(
            func=mdp.base_lin_vel_penalty,
            weight=-0.1,
            params={"height_threshold": 0.65, "asset_cfg": SceneEntityCfg("robot")},
        ),
        "standing_balance_hold": RewardTermCfg(
            func=mdp.standing_balance_hold,
            weight=1.0,
            params={
                "target_height": 0.728,
                "height_threshold": 0.65,
                "k_height": 10.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # Terminal penalty kept SMALL for the first run: get-up fails constantly
        # early, so a large terminal cost (was -200) dwarfs the dense task rewards
        # (~+9 max) and can drive a "freeze to avoid terminating" policy. Raise it
        # once a baseline stands reliably.
        "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-25.0),
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
                timestep=0.002,  # 500 Hz physics
                # Newton solver convergence. The previous (10 / 20) were aggressive
                # cuts from the mjlab defaults (100 / 50) made for CPU dev speed.
                # Get-up is CONTACT-DOMINATED (whole body on the floor) and those
                # contact forces directly feed the reward/termination signals
                # (head/feet/knee/torso/hand forces). Under-resolved contacts give
                # penetration + noisy forces → the policy learns on bad physics and
                # may not transfer (sim2real). For the GPU run we trade some
                # throughput for contact fidelity: 50 / 30 (still half the default
                # iterations). Lower to 30 / 20 if GPU throughput is too low.
                iterations=50,
                ls_iterations=30,
            ),
        ),
        decimation=5,  # control at 100Hz (0.002 * 5 = 0.01s)
        episode_length_s=10.0,
    )
