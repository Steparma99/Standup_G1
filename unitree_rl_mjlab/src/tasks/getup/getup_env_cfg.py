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
# HoST's deployable state vector does NOT include the accelerometer; it is kept on
# the CRITIC only (clean ground truth, see _get_privileged_critic_obs_terms).
_ACTOR_INCLUDE_IMU_LIN_ACC = False

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

# Grace window (env-steps) before the velocity / feet-too-high guards arm, so the
# spawn-drop landing transient during the unactuated/settling phase never trips
# them. Covers the G1 settle (5) + assist-unactuated (8) windows with margin.
# Halved 15->8 for the HoST 50 Hz control rate (step_dt 0.01 s -> 0.02 s) so the
# real-time grace window (~0.15 s) is preserved.
_GRACE_STEPS = 8

# HoST unactuated_time: the DOF-velocity / base-velocity guards arm only AFTER this
# window (30 steps = 0.6 s @ 50 Hz, matching g1_config_ground.py unactuated_timesteps=30).
_UNACTUATED_STEPS = 30

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
        # HoST state uses base ROLL and PITCH as two scalars (r_t, q_t), not the
        # full projected-gravity vector. noise ±0.05 (HoST roll/pitch noise).
        "base_roll_pitch": ObservationTermCfg(
            func=mdp.base_roll_pitch,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
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
        # HoST action-rescaler beta (curriculum-annealed action scale), observed by
        # the policy so it adapts as its authority shrinks. noise ±0.025 (HoST).
        "beta_rescaler": ObservationTermCfg(
            func=mdp.beta_rescaler,
            noise=Unoise(n_min=-0.025, n_max=0.025),
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
        # HoST's actor state was trimmed to roll+pitch and dropped pd_tracking_error
        # / imu_lin_acc to stay deployable. The CRITIC is never deployed, so it keeps
        # the richer signal: the full projected-gravity vector, the PD tracking error,
        # and the CLEAN ground-truth accelerometer. (imu_lin_acc here is already the
        # clean func; the override in _get_critic_obs_terms is a harmless no-op now.)
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "pd_tracking_error": ObservationTermCfg(
            func=mdp.pd_tracking_error,
            params={"action_name": "joint_pos"},
            clip=(-_CLIP_PD_ERR, _CLIP_PD_ERR),
        ),
        "imu_lin_acc": ObservationTermCfg(
            func=mdp.imu_lin_acc,
            clip=(-_CLIP_IMU_ACC, _CLIP_IMU_ACC),
        ),
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
            # HoST stacks the last 5 consecutive states (the ablation's sweet spot:
            # history=0 fails on hard terrain, 10 slightly degrades smoothness). mjlab's
            # circular buffer holds the last N frames INCLUDING the current one, so
            # history_length=5 == "last 5 consecutive states".
            history_length=5,
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
        # --- Per-group reward sums (multi-critic groups): the raw reward each
        # critic sees, before per-group normalization + the group weights. Read
        # task (positive) vs regularization (negative) separately on one panel. ---
        "reward_group/task": MetricsTermCfg(func=mdp.reward_group_task),
        "reward_group/style": MetricsTermCfg(func=mdp.reward_group_style),
        "reward_group/regularization": MetricsTermCfg(func=mdp.reward_group_regularization),
        "reward_group/post_task": MetricsTermCfg(func=mdp.reward_group_post_task),
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
        # ===================================================================
        # TASK group (HoST definitive): high-level objectives, weight 1 each.
        # ===================================================================
        # Stage-0 bootstrap: directional/progress rewards active while the robot is
        # on the floor (h < 0.40 m). All three are progress-based (reward the delta,
        # not the absolute pose), so parking in any static pose earns 0 — they cannot
        # be farmed. Weight=1.0 matches the existing task terms; output [0,1]/step.
        # Weights raised 3x/2x (task must dominate regularization; see plan). This is
        # the primary DIRECTION signal off the floor.
        "height_progress": RewardTermCfg(
            func=mdp.height_progress,
            weight=3.0,  # was 1.0
            params={"max_step": 0.035,  # saturates at a slower rise rate -> less incentive to rocket up
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        "prone_recovery": RewardTermCfg(
            func=mdp.prone_recovery,
            weight=2.0,  # was 1.0
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        # supine_rising_prep needs foot site names (set per-robot in env_cfgs.py).
        # Output [0, 0.5]/step (gated to supine+low height only).
        "supine_rising_prep": RewardTermCfg(
            func=mdp.supine_rising_prep,
            weight=2.0,  # was 1.0
            params={"asset_cfg": SceneEntityCfg("robot", site_names=())},
        ),
        # Head height: f_tol(h_head, [lower, inf), margin=1, value=0.1). h_head uses
        # torso_link as a proxy (no separate head body); set per-robot (env_cfgs.py).
        # lower=0.794: torso_link at target standing (pelvis 0.75 + waist_roll offset 0.044).
        "task_head_height": RewardTermCfg(
            func=mdp.task_head_height,
            weight=3.0,  # was 1.0 — primary HEIGHT signal, task must dominate reg
            params={
                "lower": 0.794,
                "margin": 1.0,
                "value_at_margin": 0.1,
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot (torso_link).
            },
        ),
        # Base orientation: f_tol(-proj_grav_b[z], [0.99, inf), margin=1, value=0.05).
        "task_base_orientation": RewardTermCfg(
            func=mdp.task_base_orientation,
            weight=3.0,  # was 1.0 — primary ORIENTATION signal, task must dominate reg
            params={"lower": 0.99, "margin": 1.0, "value_at_margin": 0.05,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Head protection: penalize any head-terrain contact and escalate on impacts.
        # ramp_steps=20 matches _MASK_STEPS — settling contacts are not penalized.
        "task_head_contact": RewardTermCfg(
            func=mdp.head_contact_penalty,
            weight=-0.5,
            params={"sensor_name": "contact_head", "ramp_steps": 20},
        ),
        "task_head_impact": RewardTermCfg(
            func=mdp.head_impact_penalty,
            weight=-2.0,
            params={
                "sensor_name": "contact_head",
                "force_threshold": _HEAD_IMPACT_PENALTY_THRESHOLD,
                "force_scale": _HEAD_IMPACT_PENALTY_SCALE,
                "ramp_steps": 20,
            },
        ),
        # ===================================================================
        # STYLE group (HoST definitive): shape the motion. Binary deviation
        # penalties carry their magnitude in the function; weight=1.0 for those.
        # ===================================================================
        "style_waist_yaw_deviation": RewardTermCfg(
            func=mdp.style_waist_yaw_deviation,
            weight=1.0,
            params={"limit": 1.4, "penalty": -10.0,
                    "asset_cfg": SceneEntityCfg("robot", joint_names=("waist_yaw_joint",))},
        ),
        # Keep the trunk straight once rising past ~0.5 m (gate_lo=0.5). Soft band penalty
        # on waist pitch+roll; gated so it does NOT block the curl/flexion early in the
        # rise, only straightens the torso as the robot approaches standing.
        "style_waist_upright": RewardTermCfg(
            func=mdp.joint_group_deviation,
            weight=-12.0,  # was -8.0 — stronger upright enforcement once standing
            params={"lower": -0.25, "upper": 0.25, "gate_lo": 0.65, "band": 0.08,
                    "require_upright": True,
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=("waist_pitch_joint", "waist_roll_joint"))},
        ),
        "style_hip_deviation": RewardTermCfg(
            func=mdp.style_hip_deviation,
            weight=2.0,  # was 1.0 — stronger hip alignment during standing
            params={"roll_limit": 1.4, "yaw_limit": 0.9, "penalty": -10.0,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        "style_knee_deviation": RewardTermCfg(
            func=mdp.style_knee_deviation,
            weight=2.0,  # was 1.0 — stronger knee alignment during standing
            params={"hi_limit": 2.85, "lo_limit": -0.06, "penalty": -0.25,  # Ground value
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_knee_joint",))},
        ),
        "style_shoulder_roll_deviation": RewardTermCfg(
            func=mdp.style_shoulder_roll_deviation,
            weight=1.0,
            params={"left_limit": -0.4, "right_limit": 0.4, "penalty": -2.5,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # CoM-in-support; weight 2.5 per foot (function sums both feet). Active > Stage 2.
        # clip_min=0.0: allows reward to reach exp(0)=1.0/foot when feet are centered under pelvis.
        "style_foot_displacement": RewardTermCfg(
            func=mdp.style_foot_displacement,
            weight=4.0,  # was 2.5 — keep feet under pelvis (CoM support)
            params={"scale": 2.0, "clip_min": 0.0,
                    "asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
        ),
        "style_foot_distance": RewardTermCfg(
            func=mdp.style_foot_distance,
            weight=2.0,  # was 1.0 — stronger foot-separation control
            params={"max_dist_sq": 0.9, "penalty": -10.0,
                    "asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
        ),
        # Shank verticality (f_tol), weight 10; active > Stage 1. knee/foot body names
        # default to G1 in rewards.py.
        "style_shank_orientation": RewardTermCfg(
            func=mdp.style_shank_orientation,
            weight=8.0,  # was 5.0 — leg geometry signal during early rise (>H_STAGE1)
            params={"lower": 0.8, "margin": 1.0, "value_at_margin": 0.1,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Low trunk angular velocity during the rise (exp), weight 1; active > Stage 1.
        "style_base_ang_vel": RewardTermCfg(
            func=mdp.style_base_ang_vel,
            weight=3.0,  # was 1.0 — penalize trunk spinning during rise (>H_STAGE1)
            params={"scale": 2.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Ankle parallel: binary +20 (highest style weight) when both feet are flat.
        # The +20 magnitude lives in the function, so weight=1.0 (binary-term convention).
        # tilt_threshold = mean per-foot ||proj_grav_xy||^2; 0.05 ~ within 13 deg of flat
        # (HoST's keypoint-variance proxy is unusable on the G1's lateral sole geoms).
        "style_ankle_parallel": RewardTermCfg(
            func=mdp.style_ankle_parallel,
            weight=1.0,
            params={"tilt_threshold": 0.05, "reward": 20.0,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Feet stumble: 0 on flat Ground (this env) -> weight 0 no-op scaffold. Raise the
        # penalty magnitude when platform/slope/wall terrains are added (Sim2Real phase).
        "style_feet_stumble": RewardTermCfg(
            func=mdp.style_feet_stumble,
            weight=0.0,
            params={"sensor_name": "feet_ground_contact", "ratio": 3.0, "penalty": 0.0,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Hand contact: exponential penalty on contact force, active at ALL stages.
        # Dead zone <= free_force (hand weight ~7.65 N) → 0 penalty; grows as
        # exp(excess/force_scale)-1 above that. Capped internally at 5×force_scale
        # to prevent overflow (raw max ≈ 147; weight -0.05 → max step penalty ≈ -7.35).
        # Weight halved -0.1 → -0.05: at floor level these two terms were the ONLY
        # active style signal (~-35/ep combined), monopolizing the style critic and
        # telling the policy "never touch the floor with hands" during the very phase
        # where a hand push-up may help. Halved so it still discourages hard hand
        # contact without dominating the (otherwise height-gated) style group.
        # Tune force_scale vs. contact/hand_*_force_max metrics once training starts.
        "style_hand_left_contact": RewardTermCfg(
            func=mdp.hand_contact_penalty,
            weight=-0.05,  # was -0.1
            params={"sensor_name": "contact_hand_left",
                    "free_force": 8.0, "force_scale": 10.0, "ramp_steps": _GRACE_STEPS},
        ),
        "style_hand_right_contact": RewardTermCfg(
            func=mdp.hand_contact_penalty,
            weight=-0.05,  # was -0.1
            params={"sensor_name": "contact_hand_right",
                    "free_force": 8.0, "force_scale": 10.0, "ramp_steps": _GRACE_STEPS},
        ),
        # ===================================================================
        # REGULARIZATION group (HoST definitive), group weight 0.1: weak shaping
        # penalties (all 9 terms wired).
        # ===================================================================
        "joint_acc_l2": RewardTermCfg(
            func=mdp.joint_acc_l2, weight=0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-2e-3),
        # Smoothness: dead-zone jerk penalty — only |jerk| > 1.0 is penalized (squared).
        # Dead zone allows normal standup acceleration; only violent oscillation is hit.
        # Max per step: 43 joints × (4-1)² = 387 raw → weight -0.01 → max ≈ -3.87
        # Phase 1: weight lowered -0.01 -> -0.003 (3x). At long episode lengths this grew
        # to -2.56/ep and became a jerk-minimization objective competing with the explosive
        # joint acceleration a floor push-up needs. Restore toward -0.01 in Phase 2 once
        # the robot stands reliably.
        "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2_deadzone, weight=-0.003,
                                        params={"deadzone": 1.0}),
        "joint_torques_l2": RewardTermCfg(
            func=mdp.joint_torques_l2, weight=0, #-2.5e-6,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        # Mechanical power Σ|τ·q̇|.
        "joint_power_l2": RewardTermCfg(
            func=mdp.joint_power_l2, weight=0, #-2.5e-5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "joint_vel_l2": RewardTermCfg(
            func=mdp.joint_vel_l2, weight=0,  # disabled: penalizing all velocity fights standup
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        # Arm velocity soft threshold: penalizes only the excess above a comfortable
        # swing speed (clamp(|q̇| - limit, 0)). Arms are free to move and assist
        # the get-up but are penalized when they exceed the per-joint speed cap.
        # Uses arm_vel_soft_limit (separate cache) to coexist with joint_vel_limits.
        # Arm velocity soft threshold. Raised limits + lowered weight: the old
        # (8/8/5 rad/s, weight -1.0) grew from -0.33 to -3.89 over training because
        # normal standup arm swing exceeded 8 rad/s — the penalty fought the very arm
        # motion needed to rise. Now 12/12/8 rad/s (normal swing free, only violent
        # thrash penalized) at weight -0.3. max_excess=15.0 caps a spike so a fall
        # transient cannot blow up the penalty.
        # Phase 1: FULLY DISABLED (weight 0). At -0.3 this reached -3.88/ep — the single
        # largest penalty — and fired hardest exactly when the robot needs a fast arm swing
        # to push off the floor, creating a discount-asymmetry trap (immediate certain
        # penalty vs delayed uncertain standing reward). The 500-iter run got stuck gently
        # oscillating arms on the floor to stay under the threshold. Re-enable in Phase 2
        # (arm velocity shaping) only AFTER the robot reliably stands.
        "reg_arm_vel": RewardTermCfg(
            func=mdp.arm_vel_soft_limit,
            weight=0,  # was -0.3 — disabled for Phase 1 standup learning
            params={
                "vel_limits": {
                    ".*_shoulder_.*": 12.0,  # was 8.0 — allow normal standup swing
                    ".*_elbow_joint":  12.0,  # was 8.0
                    ".*_wrist_.*":     8.0,   # was 5.0
                },
                "max_excess": 15.0,  # explosion guard: cap per-joint over-limit
            },
        ),
        # Hand velocity penalty: keeps fingers from being ACTIVELY spun during get-up.
        # Was joint_vel_l2 (Σ q̇², no threshold) which penalized the unavoidable passive
        # finger drag on the floor (~1 rad/s) → a constant ~-10/ep noise floor the policy
        # could not reduce. Now a DEADBAND (clamp(|q̇| - 3.0, 0)): 0 while fingers move at
        # normal passive speed, linear only when spun past 3 rad/s. max_excess=5.0 caps a
        # spike at 5 rad/s over-limit → per-step penalty bounded to
        # ~n_finger_joints * 5 * 0.5 (no reward explosion on a sim glitch / fall).
        # Velocity-only (not position) so trapped fingers do not fight contact force
        # through the low-effort (1.4 Nm) actuators. Policy stays 43-DOF, deployable.
        # Phase 1: DISABLED (weight 0). At -0.5 this reached -3.75/ep because the 3 rad/s
        # deadband is too tight — any floor-pushing wrist/finger motion exceeds it, so it
        # penalized the very hand use needed to get up. Hand velocity is not safety-critical
        # during standup learning; re-enable in Phase 2 to stop finger thrash in standing.
        "reg_hand_vel": RewardTermCfg(
            func=mdp.hand_vel_soft_limit,
            weight=0,  # was -0.5 — disabled for Phase 1 standup learning
            params={
                "vel_limits": {r".*_hand_(thumb|middle|index)_\d_joint": 3.0},  # rad/s deadband
                "max_excess": 5.0,  # explosion guard: cap per-joint over-limit
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # Strongest in the group: PD target-vs-measured gap (implicit effort penalty).
        # HoST uses -2.5e-4 here; the previous -2.5e-1 was 1000x harsher, which
        # discouraged moving any joint off its default (esp. the arms). Reduced 10x
        # to -2.5e-2 — still a meaningful effort penalty but no longer pinning the
        # arms in place now that the arm PD gains are responsive (Kp=100).
        "joint_tracking_error": RewardTermCfg(
            func=mdp.joint_tracking_error, weight=0,
            params={"action_name": "joint_pos", "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Joint position limit penalty: linear in radian overshoot, weight -10 (was -100).
        # Keeps gradient information (how far outside = how hard the push back).
        # Max: ~2.05 rad total overshoot × -10 = -20.5. Hands already excluded.
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits, weight=-10,
            # Finger joints (Dex3-1) have their XML range boundary at q=0; the 98%
            # soft-factor moves it inward leaving q=0 just outside, firing a constant
            # penalty regardless of pose. Exclude them: hand_contact_penalty + tiny
            # effort limits (1.4 Nm) already cover finger protection.
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("^(?!.*_hand_).*$",))},
        ),
        # Joint velocity limits: normalized ramp from 85% to 100% of each joint's limit.
        # Each joint contributes at most 1; total bounded by num_joints.
        # Max: 43 joints × 1 × weight -0.15 = -6.45. No penalty below 85% of limit.
        "joint_vel_limits": RewardTermCfg(
            func=mdp.joint_vel_limits,
            weight=-0.15,
            params={
                "vel_limits": {  # G1 actuator velocity limits [rad/s]
                    ".*_hip_pitch_joint": 32.0, ".*_hip_yaw_joint": 32.0, "waist_yaw_joint": 32.0,
                    ".*_hip_roll_joint": 20.0, ".*_knee_joint": 20.0,
                    ".*_wrist_pitch_joint": 22.0, ".*_wrist_yaw_joint": 22.0,
                    ".*_ankle_pitch_joint": 50.0, ".*_ankle_roll_joint": 50.0,
                    "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0,
                    ".*_shoulder_.*": 37.0, ".*_elbow_joint": 37.0, ".*_wrist_roll_joint": 37.0,
                },
                "soft_ratio": 0.85,
                "normalize": True,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # ===================================================================
        # POST-TASK group (HoST definitive), group weight 1.0: shape the held
        # standing state. All terms gated by the HARD indicator 1(h_base > H_STAGE2
        # = 0.65); zero until standing. Weights all 10 except feet_parallel = 2.5.
        # ===================================================================
        "post_base_ang_vel": RewardTermCfg(
            func=mdp.post_base_ang_vel, weight=10.0,
            params={"scale": 2.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        "post_base_lin_vel": RewardTermCfg(
            func=mdp.post_base_lin_vel, weight=15.0,
            params={"scale": 20.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        "post_base_orientation": RewardTermCfg(
            func=mdp.post_base_orientation, weight=10.0,
            params={"scale": 8.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Base height target 0.65 m (standing threshold); tight Gaussian (-20).
        "post_base_height": RewardTermCfg(
            func=mdp.post_base_height, weight=10.0,
            params={"target_height": 0.75, "scale": 20.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # arm-only posture disabled: full-body HOME tracking is handled by post_standing_posture.
        "post_upper_body_posture": RewardTermCfg(
            func=mdp.post_upper_body_posture, weight=0.0,
            params={"target_joint_pos": {}, "joint_weights": {}, "scale": 0.5,
                    "height_threshold": 0.65,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Full-body HOME pose L2 penalty gated to standing (h_pelvis > 0.65 m).
        # Uses direct weighted squared error — constant gradient at any distance from
        # HOME. Fixes the dead-gradient problem of the exp(-kp*err) form used by
        # standing_posture(), which saturates to zero when the robot is far from HOME
        # (exactly when a fine-tuning run needs the signal most).
        # Returns a positive value; use with a NEGATIVE weight.
        # target_joint_pos and joint_weights set per-robot in env_cfgs.py (HOME pose).
        "post_standing_posture": RewardTermCfg(
            func=mdp.home_pose_l2, weight=-3.0,
            params={"target_joint_pos": {}, "joint_weights": {},
                    "height_threshold": 0.65,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Both-feet grounded reward: 0.5 for one foot, 1.0 for both. Directly penalizes
        # the one-foot balancing behaviour. site_names set per-robot in env_cfgs.py.
        # Gate triggers earlier (h=0.50m) than the other post-task terms (0.65m), so
        # this is the bridge signal during the rise. Weight raised 5→8 to make planting
        # both feet more valuable through the mid-rise phase.
        "post_stand_on_feet": RewardTermCfg(
            func=mdp.stand_on_feet, weight=8.0,  # was 5.0
            params={"sensor_name": "feet_ground_contact",
                    "asset_cfg": SceneEntityCfg("robot", site_names=())},
        ),
        "post_feet_parallel": RewardTermCfg(
            func=mdp.post_feet_parallel, weight=2.5,
            params={"scale": 20.0, "clip_min": 0.001, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Feet yaw alignment: exp(-10·(q_yawL²+q_yawR²)); peak when both hip_yaw=0 (toes fwd).
        "post_feet_yaw": RewardTermCfg(
            func=mdp.post_feet_yaw, weight=2.5,
            params={"scale": 10.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Explicit bonus for holding the stand (not just touching it for a frame).
        # Also the only writer of episode_state["ever_stood"]/["standing_counter"],
        # which success/ever_stood, success/stable_hold and no_progress_timeout's
        # post-standing exemption all depend on.
        "stable_success_hold": RewardTermCfg(
            func=mdp.stable_success_hold, weight=10.0,
            params={"n_hold": 15, "height_threshold": 0.65, "asset_cfg": SceneEntityCfg("robot")},
        ),
    }

    ##
    # Terminations: only timeout — bad_orientation excluded because robot starts lying
    ##

    # HoST ground / ground-prone termination set: timeout + DOF-vel limit + base-vel
    # limit (both gated after the unactuated window), plus two numerical safety nets
    # (nan, ground_penetration) that mjlab benefits from. The behavioral guards and
    # the platform/wall/slope height checks are deliberately NOT included here — those
    # belong to the other terrains, to be added in the Sim2Real phase.
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan": TerminationTermCfg(func=mdp.nan_detection),
        # HoST DOF velocity limit: any |q̇| over limit (≡ max|q̇|>limit), armed only
        # after the unactuated window so the spawn-drop transient never trips it.
        "joint_vel_explosion": TerminationTermCfg(
            func=mdp.joint_velocity_explosion,
            params={
                "max_velocity": 300.0,
                "grace_steps": _UNACTUATED_STEPS,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # HoST base_vel_out: floating-base speed over limit, armed after the window.
        "base_vel_explosion": TerminationTermCfg(
            func=mdp.base_velocity_out,
            params={
                "max_velocity": 20.0,
                "grace_steps": _UNACTUATED_STEPS,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # Numerical safety net (not in HoST): catch a base sinking through the floor.
        "ground_penetration": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": -0.3, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Failure termination (no time_out=True): penalises stalling. Tracks the best
        # pelvis height reached this episode; if no new record > min_progress is set
        # for n_stall_steps consecutive steps (past the grace window), the episode
        # ends and is_terminated fires. Prevents the robot from hovering at an
        # intermediate height for 500 steps without penalty.
        "no_progress_timeout": TerminationTermCfg(
            func=mdp.no_progress_timeout,
            params={
                "min_progress": 0.02,
                "n_stall_steps": 200,
                "grace_steps": _UNACTUATED_STEPS + 20,  # settle + first-attempt buffer
                "asset_cfg": SceneEntityCfg("robot"),
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
                # HoST sim fidelity: 200 Hz PD / physics, policy at 50 Hz
                # (decimation=4 below). For deployment this switches to 500 Hz PD /
                # 50 Hz policy (decimation=10). Larger timestep makes the contact
                # problem stiffer per step, so the solver iterations are kept high
                # (50 / 30) rather than reduced.
                timestep=0.005,  # 200 Hz physics
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
        decimation=4,  # control at 50Hz (0.005 * 4 = 0.02s) — HoST policy rate
        episode_length_s=10.0,
    )
