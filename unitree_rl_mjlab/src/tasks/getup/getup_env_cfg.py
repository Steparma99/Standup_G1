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

_HEAD_IMPACT_PENALTY_THRESHOLD = 250.0  # metric/diag threshold (penalty term removed v9)
_HEAD_IMPACT_TERMINATION_THRESHOLD = 1200.0

# Grace window (env-steps) before the velocity / feet-too-high guards arm, so the
# spawn-drop landing transient during the unactuated/settling phase never trips
# them. Covers the G1 settle (5) + assist-unactuated (8) windows with margin.
# Halved 15->8 for the HoST 50 Hz control rate (step_dt 0.01 s -> 0.02 s) so the
# real-time grace window (~0.15 s) is preserved.
_GRACE_STEPS = 8

# HoST unactuated_time: the DOF-velocity / base-velocity guards arm only AFTER this
# window (HoST g1_config_ground.py uses unactuated_timesteps=30 = 0.6 s @ 50 Hz;
# v18 extended 30->40->60 = 1.2 s, matched to config/g1/env_cfgs._SETTLE_STEPS so
# the guards never arm while the robot is still passively settling).
_UNACTUATED_STEPS = 60

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
        # v25 diagnostic: fraction of steps in a STILL stand (height + low
        # velocity). Compare vs success/candidate — the gap is "stands but wobbles".
        "success/hold_still_fraction": MetricsTermCfg(func=mdp.hold_stillness_fraction),
        # v26 diagnostic: mean continuous terminal-stability gate g4 (height ×
        # upright × both-feet-loaded). Fraction of the episode spent in the
        # task-space "settled quiet stand" region; should rise with post_terminal_core.
        "success/terminal_gate": MetricsTermCfg(func=mdp.terminal_gate_fraction),
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
        # --- Left/right arm posture diagnostic (observation only, see metrics.py) ---
        # target_joint_pos / joint_weights set per-robot in env_cfgs.py, mirroring the
        # post_upper_body_posture reward's params pattern but filtered to one side.
        "posture/left_arm_error": MetricsTermCfg(
            func=mdp.left_arm_posture_error,
            params={"target_joint_pos": {}, "joint_weights": {}},
        ),
        "posture/right_arm_error": MetricsTermCfg(
            func=mdp.right_arm_posture_error,
            params={"target_joint_pos": {}, "joint_weights": {}},
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
        # Fingers are EXCLUDED from the policy's action space (negative lookahead,
        # same regex as joint_pos_limits below): 29 body DOF instead of 43. The 14
        # finger joints are PD-held by the hand_hold term below — they were pure
        # exploration noise for standup (~50% of Gaussian samples saturate the ±1
        # clamp on dims nothing shapes), matching HoST's 23-DOF no-hands design.
        "joint_pos": mdp.LowPassJointPositionActionCfg(
            entity_name="robot",
            actuator_names=("^(?!.*_hand_).*$",),
            scale=0.25,  # Override per-robot.
            use_default_offset=True,
            alpha=0.5,
        ),
        # Zero-action-dim hold: writes a fixed finger pose target every substep
        # (and at reset). Target pose set per-robot (HAND_HOLD_JOINT_POS) — several
        # G1 finger joints have q=0 as a range boundary, so they hold slightly
        # curled into range rather than at the hard stop.
        "hand_hold": mdp.HandHoldActionCfg(
            entity_name="robot",
            actuator_names=(r".*_hand_(thumb|middle|index)_\d_joint",),
            target_joint_pos=None,  # Set per-robot in config/g1/env_cfgs.py.
        ),
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
        # BRIDGE reward: dense feet-planted pelvis-height ramp (0.35->0.75 m) that
        # fills the crouch->stand dead zone where the old stack had no climbing
        # gradient (height_progress is a delta; post terms gated at 0.65). Not
        # farmable: requires pelvis height AND upright trunk AND both feet planted
        # simultaneously (kills the glute-bridge and ballistic-jump exploits). Needs
        # the foot sensor + site names (set per-robot in env_cfgs.py). See
        # rewards.pelvis_height_bridge. Weight comparable to the primary task signals.
        "pelvis_height_bridge": RewardTermCfg(
            func=mdp.pelvis_height_bridge,
            weight=5.0,
            params={"sensor_name": "feet_ground_contact", "lower": 0.35, "upper": 0.75,
                    "asset_cfg": SceneEntityCfg("robot", site_names=())},
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
        # Head protection: penalize any head-terrain contact.
        # ramp_steps=20 matches _MASK_STEPS — settling contacts are not penalized.
        # task_head_impact REMOVED (v9 ablation): it only fired above 250 N excess,
        # a severity escalation of this per-step contact penalty — both logged 0 from
        # mid-training on. Head impacts stay observable via the contact/head_impact
        # METRIC below; re-add mdp.head_impact_penalty if hard head slams reappear.
        "task_head_contact": RewardTermCfg(
            func=mdp.head_contact_penalty,
            weight=-0.5,
            params={"sensor_name": "contact_head", "ramp_steps": 20},
        ),
        # ===================================================================
        # STYLE group (HoST definitive): shape the motion. Binary deviation
        # penalties carry their magnitude in the function; weight=1.0 for those.
        # ===================================================================
        # style_waist_yaw_deviation / style_hip_deviation / style_knee_deviation /
        # style_shoulder_roll_deviation REMOVED (v9 ablation): binary HoST safety-limit
        # penalties with loose thresholds that logged ~0 for entire converged runs —
        # they never fired and only added gradient noise. Posture shaping is owned by
        # post_standing_posture / post_upper_body_posture; hard joint limits are still
        # guarded by joint_pos_limits (-10). Functions kept in mdp/rewards.py.
        # Incentivize trunk uprightness THROUGH the rise (gate_lo lowered 0.65 -> 0.45 =
        # H_STAGE1) instead of only once standing, so the robot keeps straightening out of
        # the crouch rather than farming crouch rewards. POSITIVE dead-zone reward (was a
        # -12.0 penalty): v_free=0.1 rad full reward, decays to 0.1 at ~0.5 rad tilt.
        # SAFE to activate early because require_upright=True multiplies by uprightness:
        # while the trunk is still horizontal in the Stage-0 curl the payout is ~0, and it
        # grows only as the trunk actually comes vertical — it never rewards a flat trunk.
        "style_waist_upright": RewardTermCfg(
            func=mdp.waist_upright_reward,
            weight=8.0,  # POSITIVE (was -12.0 penalty)
            params={"v_free": 0.10, "margin": 0.40, "gate_lo": 0.45, "band": 0.08,
                    "require_upright": True,
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=("waist_pitch_joint", "waist_roll_joint"))},
        ),
        # CoM-in-support; weight 2.5 per foot (function sums both feet). Active > Stage 2.
        # clip_min=0.0: allows reward to reach exp(0)=1.0/foot when feet are centered under pelvis.
        "style_foot_displacement": RewardTermCfg(
            func=mdp.style_foot_displacement,
            weight=4.0,  # was 2.5 — keep feet under pelvis (CoM support)
            # height_threshold lowered 0.65 -> 0.45 (H_STAGE1): reward feet-under-CoM
            # THROUGH the rise so the robot sets up a balanced base earlier, supporting
            # the earlier trunk-uprightness incentive.
            params={"scale": 2.0, "clip_min": 0.0, "height_threshold": 0.45,
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
            weight=8.0,  # was 5.0 — leg geometry signal (weight unchanged, only the gate moved)
            # height_threshold 0.0 -> 0.30 (gate ~50% at 0.30 m, full by 0.38 m). The
            # old ground-active gate paid ~full shank reward to a robot LYING DOWN that
            # just tucked its shanks vertical (knees up), a large farmable term (~3.8/ep,
            # the single biggest reward) that ALSO fought the standup: extending the legs
            # to rise from the crouch tilts the shanks and LOSES this reward. Gating it to
            # ~0.30 m means it no longer pays on the floor and no longer penalises the leg
            # extension during the crucial crouch->stand transition, while still strongly
            # rewarding vertical shanks once the robot is genuinely rising/standing.
            params={"lower": 0.8, "margin": 1.0, "value_at_margin": 0.1,
                    "height_threshold": 0.30,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Low trunk angular velocity during the rise (exp), weight 1; active > Stage 1.
        # Rise-phase trunk-spin reward, now DEAD-ZONED (f_tol): a normal rise rotation is
        # free (under v_free rad/s), only violent spin is shaped. Stays rise-gated
        # (h>H_STAGE1). Weight softened 3.0 -> 1.5 (velocity is fine during the rise).
        "style_base_ang_vel": RewardTermCfg(
            func=mdp.style_base_ang_vel,
            weight=1.5,  # was 3.0
            params={"v_free": 0.8, "margin": 2.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Ankle parallel: binary +20 (highest style weight) when both feet are flat.
        # The +20 magnitude lives in the function, so weight=1.0 (binary-term convention).
        # tilt_threshold = mean per-foot ||proj_grav_xy||^2; 0.05 ~ within 13 deg of flat
        # (HoST's keypoint-variance proxy is unusable on the G1's lateral sole geoms).
        # v19: reward 20.0 -> 10.0 — at it14000 this was the single largest term in
        # the entire stack (8.6/ep, vs 4.5 for base_height), for a behavior (feet
        # flat) already double-covered by post_stand_on_feet and post_feet_parallel.
        # At 20 it monopolized the critic's budget and diluted the gradient on the
        # unconverged pose terms; 10 keeps it top-of-style without dominating.
        "style_ankle_parallel": RewardTermCfg(
            func=mdp.style_ankle_parallel,
            weight=1.0,
            params={"tilt_threshold": 0.05, "reward": 10.0,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # style_feet_stumble REMOVED (v9 ablation): weight-0 no-op scaffold on flat
        # Ground. Re-add from mdp.style_feet_stumble when platform/slope/wall terrains
        # arrive (Sim2Real phase).
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
        # latch_height=0.72: penalty is ACTIVE all through the push-off/rise (discourage
        # hand-crutch reliance, as in the previous training) and LATCHES OFF for the rest
        # of the episode once the pelvis first reaches HOME (~0.72 m). A later fall does
        # NOT re-arm it, so a failed episode is never buried under a huge hand-contact
        # penalty (the noise that dominated late in prior runs).
        "style_hand_left_contact": RewardTermCfg(
            func=mdp.hand_contact_penalty,
            weight=-0.05,  # was -0.1
            params={"sensor_name": "contact_hand_left",
                    "free_force": 8.0, "force_scale": 10.0, "ramp_steps": _GRACE_STEPS,
                    "latch_height": 0.72, "asset_cfg": SceneEntityCfg("robot")},
        ),
        "style_hand_right_contact": RewardTermCfg(
            func=mdp.hand_contact_penalty,
            weight=-0.05,  # was -0.1
            params={"sensor_name": "contact_hand_right",
                    "free_force": 8.0, "force_scale": 10.0, "ramp_steps": _GRACE_STEPS,
                    "latch_height": 0.72, "asset_cfg": SceneEntityCfg("robot")},
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
        # Max per step: 29 action dims × (4-1)² = 261 raw → weight -0.01 → max ≈ -2.61
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
            weight=-0.1,  # re-enabled mild (was 0): settle arms at HOME after standup;
            # soft limits stay high (12/12/8 rad/s) so the standup swing is still free
            params={
                "vel_limits": {
                    ".*_shoulder_.*": 12.0,  # was 8.0 — allow normal standup swing
                    ".*_elbow_joint":  12.0,  # was 8.0
                    ".*_wrist_.*":     8.0,   # was 5.0
                },
                "max_excess": 15.0,  # explosion guard: cap per-joint over-limit
            },
        ),
        # reg_hand_vel REMOVED: fingers are no longer policy-actuated (see the
        # hand_hold action term) — they are PD-held at a fixed pose, and the
        # actuator's own damping (G1_ACTUATOR_HAND_HOLD Kd) is now the physical,
        # non-RL regularizer against passive finger motion. A reward on finger
        # velocity has no policy-controllable input left to shape.
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
        # Stillness rewards, DEAD-ZONED (f_tol) + HOME-gated: full reward while the base
        # is slow (small in-place adjustments free), decaying only above the threshold,
        # and active ONLY once fully standing at HOME (_HOME_HEIGHT) — never during the
        # rise. Weights lowered from the old exp-form (which pinned v->0 from the crouch
        # up and fought the push-off). Jump safety moved to anti_jump_velocity below.
        "post_base_ang_vel": RewardTermCfg(
            func=mdp.post_base_ang_vel, weight=4.0,  # was 10.0
            params={"v_free": 0.5, "margin": 1.5, "asset_cfg": SceneEntityCfg("robot")},
        ),
        "post_base_lin_vel": RewardTermCfg(
            func=mdp.post_base_lin_vel, weight=6.0,  # was 15.0
            params={"v_free": 0.3, "margin": 0.7, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Anti ballistic-jump: squared base-speed EXCESS above v_high (1.5 m/s), gated
        # from the crouch (0.45 m) upward so it blocks a launch DURING the rise too.
        # Negative weight; normal rise speeds sit under v_high and are unpenalised.
        "anti_jump_velocity": RewardTermCfg(
            func=mdp.anti_jump_velocity, weight=-3.0,
            params={"v_high": 1.5, "cap": 3.0, "asset_cfg": SceneEntityCfg("robot")},
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
        # Arm/waist HOME-posture reward, ENABLED (weight 2.0). A dedicated upper-body
        # signal so the arms return to the HOME pose after standup instead of trailing
        # behind the trunk (which pushes the whole-body CoM backward and destabilises the
        # stand — the arms-back behaviour seen in the eval videos). Kept SEPARATE from the
        # leg-dominated post_standing_posture so the arm gradient is not drowned out by the
        # high-weight leg terms in that normalised 29-DOF metric.
        "post_upper_body_posture": RewardTermCfg(
            # PENALTY (v5 arm fix): now returns the weighted mean per-joint squared
            # error from HOME (non-negative, 0 = perfect) — weight must be NEGATIVE.
            # History: exp form was gradient-dead (unnormalized 17-joint sum, weight
            # raises 2->4->6 did nothing); the v4 positive linear clamp(1−0.5·mean, 0)
            # had a live gradient but at weight 6.0 was too weak vs com_over_support —
            # arms stuck ~0.8 rad RMS off HOME after 900 iterations (climb rate only
            # ~0.28/1000 it). The penalty form is unbounded: unlike a capped positive
            # reward it keeps hurting at ANY distance, and the quadratic gradient grows
            # with the error instead of flattening.
            # EXPONENTIAL form (v7): linear penalty showed diminishing returns per
            # weight doubling (-10 plateaued at 0.67 rad RMS, -20 at 0.60) and the
            # rollout diagnostic RULED OUT com_over_support as the opposing force
            # (4.94 stuck vs 4.99 HOME — 1.2% of cap, noise): there is no fixed
            # opponent to out-price, the limit is gradient magnitude at the current
            # error. exp(k·metric)−1 grows supralinearly: at weight -2.5, k=3 the
            # gradient at the v6 plateau (metric 0.36) is ~22 (vs linear-20's
            # constant 20) and at the resume start (0.62) ~48 — strongest early
            # pull of any form tried — while the worst clamped transient
            # (metric=1.0) costs 2.5·(e³−1) ≈ 48/step, comparable to one step of
            # the post_task positive budget (~50), so it cannot teach
            # stand-avoidance. Near HOME the pull eases (post_standing_posture
            # handles the final polish).
            func=mdp.post_upper_body_posture, weight=-2.5,
            params={"target_joint_pos": {}, "joint_weights": {}, "scale": 1.0,
                    "k": 3.0, "max_metric": 1.0,
                    "height_threshold": 0.65,
                    # ramp_from=H_STAGE1 (v8 arm fix): fade the penalty in from
                    # Stage 2 entry (h=0.45) instead of the symmetric band
                    # [0.57, 0.73]. The arms-back pose is locked in DURING the
                    # rise, where the band gate was fully closed — v7_arm_exp
                    # showed the post-stand pull alone improves the metric
                    # (-9.8 -> -4.8 over 1000 it) but can't undo the habit in
                    # video. Stage 0/1 (h<0.45, floor righting/push-off) stays
                    # ungated so arm use for push-off is not penalized.
                    "ramp_from": 0.45,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Full-body HOME pose L2 penalty gated to standing (h_pelvis > 0.65 m).
        # Uses direct weighted squared error — constant gradient at any distance from
        # HOME. Fixes the dead-gradient problem of the exp(-kp*err) form used by
        # standing_posture(), which saturates to zero when the robot is far from HOME
        # (exactly when a fine-tuning run needs the signal most).
        # Returns a positive value; use with a NEGATIVE weight.
        # target_joint_pos and joint_weights set per-robot in env_cfgs.py (HOME pose).
        # POSITIVE "climax toward HOME" reward (was the home_pose_l2 penalty, weight -3.0).
        # standing_posture = exp(-kp·Σ w·(q-q*)²) · standing_gate ∈ (0,1]: bounded, peaks
        # at the HOME pose, and NEVER punishes a struggling stand (the old L2 penalty grew
        # with distance from HOME, actively repelling the robot from the standing basin the
        # moment its posture was imperfect — exactly when the assist force was withdrawn).
        # Same target_joint_pos / joint_weights dicts (set per-robot in env_cfgs.py).
        # kp=4.0 tuned for the NORMALIZED metric (mean per-joint weighted sq error):
        # mean 0.1 rad -> 0.96, 0.3 rad -> 0.70, 0.8 rad -> 0.08 — smooth gradient
        # across the full operating range. Before normalization the exp saturated to
        # ~0 for any realistic pose (logged 0.0000 the entire previous run).
        "post_standing_posture": RewardTermCfg(
            func=mdp.standing_posture, weight=10.0,  # 5.0->7.5->10.0: v8 still plateaued
            # at ~32% of cap (2.42/7.5) while base-height/orientation sat at 85-90%;
            # raised to their tier (10.0) so closing the remaining full-body HOME error
            # (bent knees, trailing arm) pays as much as holding height/uprightness.
            params={"target_joint_pos": {}, "joint_weights": {},
                    "pelvis_height_threshold": 0.65, "kp": 4.0,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # v20: constant-gradient HOME-pose pull (home_pose_l2, rewards.py) — the
        # missing counterpart to post_standing_posture's exp form. exp(-kp*err)
        # has near-zero gradient FAR from HOME, which is exactly where the
        # recurring crooked-arm/leg stuck poses live (dead arm behind the back
        # persisted v1->v19 through pure weight tuning). This term returns the
        # RAW weighted squared error (negative weight => penalty), so the pull
        # toward HOME grows with distance and never saturates or dies. Runs
        # ALONGSIDE the exp terms: they reward being close, this punishes being
        # far. Same standing gate (0.65m) as the other final-pose terms.
        # Weight -2.0: worst observed crooked poses (~1-1.5 rad on 1-2 joints)
        # cost ~-3..-5/step vs a ~+55/step healthy standing bundle — a real
        # gradient without making standing net-negative. Halve to -1.0 if
        # success/ever_stood drops after this change; raise toward -4.0 if the
        # pose error plateaus again with this term's magnitude still small.
        # v21: -2.0 -> -4.0 — that predicted plateau happened: the v20 run logged
        # this flat at ~-1.4 for 1800 its while the exp terms sat on their dead
        # near-HOME gradient (diag sweep: post_standing_posture drops only
        # 0.92->0.88 over the first 0.1 rad). This is the only non-saturating
        # HOME pull, so it gets the budget; arm/waist joint weights also raised
        # in env_cfgs.py so the visible residual (arms/waist) is what it bites.
        # target/joint_weights set per-robot in env_cfgs.py.
        "post_home_pose_l2": RewardTermCfg(
            func=mdp.home_pose_l2, weight=-4.0,
            params={"target_joint_pos": {}, "joint_weights": {},
                    "height_threshold": 0.65,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # v21 REWRITE: HOME-anchored WORST-SIDE posture reward (was L-vs-R mirror
        # symmetry, v16-v20). The old form's gradient (`q_left - sign·q_right`)
        # pulled the GOOD limb toward the BAD limb exactly as hard as the reverse
        # and scored a symmetric-but-wrong pose (both arms behind the back)
        # perfectly — the final pose was slightly better BEFORE v16 added it and
        # worse after v19 raised its leg weights, while in the v20 run it paid
        # only 1.12/8 (metric≈0.49: failing at its own job yet reshaping the
        # landscape). New metric: max(weighted mean HOME error of LEFT side,
        # same for RIGHT side) — keeps the anti-masking intent (the max ignores
        # the good side, so it can never hide a bad limb) but the gradient lands
        # 100% on the worst side and always points AT HOME. joint_weights (family
        # names) + target_joint_pos (HOME) set per-robot in env_cfgs.py.
        # Weight 5.0 -> 2.0: keep the anti-masking diagnostic pressure, but let the
        # more specific HOME-pose pulls dominate the next A/B test. At this point the
        # residual defects are not "generic asymmetry" problems; they are specific
        # shoulder-yaw / lower-body alignment errors that the targeted posture terms can
        # address more directly.
        "post_bilateral_symmetry": RewardTermCfg(
            func=mdp.post_bilateral_symmetry, weight=2.0,
            params={"joint_weights": {}, "target_joint_pos": {}, "scale": 1.0,
                    "kp": 4.0, "height_threshold": 0.65, "ramp_from": 0.45,
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
        # Feet yaw alignment: exp(-2·(q_yawL²+q_yawR²)); peak when both hip_yaw=0 (toes fwd).
        # scale lowered 10 -> 2.0: at 10 a duck stance (hip_yaw ±0.4 rad) gave
        # exp(-10·2·0.16)=0.04 ≈ saturated (logged ~0.005 all run, no gradient to
        # straighten feet). At 2.0 the same stance gives 0.53 — real gradient.
        # v9: weight 2.5 -> 4.0 and ramp_from=0.45 (same rise-phase fade-in as
        # post_base_height/orientation). The v8 run logged this term FALLING
        # 0.40 -> 0.096 over training — hip yaw drifted (legs crossing) while every
        # other post term climbed, i.e. at 2.5 with a standing-only band gate it was
        # losing the trade against com_over_support/stable stance rewards.
        # v19: weight 4.0 -> 6.0 — logged 0.36/4 (~9% of cap) at it14000, the least
        # saturated post term by far; the visible "crooked legs" at hold is largely
        # hip-yaw, and at 4.0 it was still losing the trade (same failure mode the
        # v9 2.5->4.0 bump addressed, one tier short).
        "post_feet_yaw": RewardTermCfg(
            func=mdp.post_feet_yaw, weight=6.0,
            params={"scale": 2.0, "ramp_from": 0.45,
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # CoM-over-support-polygon: exp(-dist_scale·d²) between the TRUE whole-body CoM
        # (mujoco subtree_com — captures ARM position, ~2.6 cm forward of pelvis at HOME
        # and shifts further as the arms move) and the foot-midpoint. DIRECT geometric
        # balance signal — targets the arms-back CoM problem (CoM behind the feet because
        # arms never reposition), which a pelvis-only proxy cannot see. Three gates:
        # height (ramp 0.25->0.40 m), uprightness (ramp on -proj_grav_z 0.3->0.7, then
        # FULLY ON at full upright — it does NOT switch off when standing), and
        # ANY-foot-contact (support polygon undefined without a planted foot).
        # Strengthened for the posture pass: weight 3->5 (balance matters more relative to
        # the other post-task terms) and dist_scale 5->8 (sharper Gaussian → stronger
        # gradient pulling the CoM/arms forward over the feet; at 8 a 20 cm backward CoM
        # offset costs ~27% vs ~18% at 5, so arms-back is disciplined harder).
        # site_names + feets sensor set per-robot in env_cfgs.py.
        # v30: margin-based static + capture-point balance over the double-foot
        # support polygon (replaces the old feet-midpoint Gaussian). Rewards a signed
        # MARGIN into the support box (not merely "CoM inside"), blends in a
        # capture-point margin near the settled stand, and requires a hysteretic
        # double-support contact. site_names (left_foot,right_foot) set in env_cfgs.py.
        "com_over_support": RewardTermCfg(
            func=mdp.com_over_support, weight=5.0,
            params={"foot_sensor_name": "feet_ground_contact",
                    "m_safe_com": 0.03, "sigma_com": 0.05,
                    "m_safe_cp": 0.02, "sigma_cp": 0.05,
                    "f_on": 60.0, "f_off": 30.0,
                    "asset_cfg": SceneEntityCfg("robot", site_names=())},
        ),
        # Explicit bonus for holding the stand (not just touching it for a frame).
        # Also the only writer of episode_state["ever_stood"]/["standing_counter"],
        # which success/ever_stood, success/stable_hold and no_progress_timeout's
        # post-standing exemption all depend on.
        "stable_success_hold": RewardTermCfg(
            func=mdp.stable_success_hold, weight=10.0,
            # v25: the hold now requires STILLNESS, not just height — the counter
            # resets on gross wobble (base/joint speed above these cutoffs), so the
            # weight-10 success reward can no longer be farmed by a standing-but-
            # oscillating pose. Cutoffs are generous (kill gross wobble, not small
            # settling); loosen if fall_after_success spikes, tighten if the final
            # pose still visibly drifts. Watch success/hold_still_fraction.
            params={
                "n_hold": 15, "height_threshold": 0.65,
                "max_base_lin_vel": 0.5, "max_base_ang_vel": 1.0,
                "max_joint_speed": 1.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # v23 FUNCTIONAL final-pose terms (replace the exact-HOME imitation terms,
        # which are set to weight 0 in config/g1/env_cfgs.py). Specify the objective —
        # arms in front, whole-body stillness, stay-standing, L/R symmetry — with
        # generous tolerance and let the policy find its own stable pose. Palm site
        # names for arms_in_front are set per-robot in env_cfgs.py.
        # v24: band [0.12, 0.30] m (margin 0.10) = palm IN FRONT, clear of the body
        # (>0.12, torso surface ~0.10 forward), NOT extended (<0.30). HOME's natural
        # palm sits at 0.157 (inside). ramp_from=None → active ONLY near standing (the
        # v23 ramp_from=0.45 wrongly shaped the arms from the rise phase onward).
        "arms_in_front": RewardTermCfg(
            func=mdp.arms_in_front, weight=5.0,
            params={"fwd_lo": 0.12, "fwd_hi": 0.30, "margin": 0.10,
                    "ramp_from": None,
                    "asset_cfg": SceneEntityCfg("robot", site_names=())},
        ),
        # Stillness is SECONDARY to being stably up (user directive) — modest weight.
        # v24: function fixed to RMS per-joint velocity (v23 L2-norm form never fired).
        "post_joint_stillness": RewardTermCfg(
            func=mdp.post_joint_stillness, weight=3.0,
            params={"v_free": 0.5, "margin": 1.5, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # Gentle 'stay standing' survival bonus (low weight — post_base_height already
        # rewards being AT height; this rewards DURATION of standing).
        "standing_alive_bonus": RewardTermCfg(
            func=mdp.standing_alive_bonus, weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        # v24: symmetry reward REMOVED (user directive — it was the term that started
        # the plateau at v16, and v23's weakened relational version did not help either;
        # asymmetry is emergent, not a code bug — a full audit found no L/R structural
        # asymmetry). post_bilateral_symmetry stays disabled (weight 0) for easy re-add.
        # ===================================================================
        # v26 TASK-SPACE TERMINAL "quiet standing" region (DeepMimic-style).
        # Define WHAT a settled stand is (trunk vertical, both feet loaded, CoM
        # over support, feet correctly L/R ordered) with generous tolerance and
        # combine the factors MULTIPLICATIVELY, instead of prescribing an exact
        # HOME joint pose. Added ALONGSIDE the existing overlapping post_task
        # terms (post_base_orientation / post_stand_on_feet / com_over_support are
        # left untouched) for an A/B comparison — prune the redundant ones later.
        # site_names ("left_foot", "right_foot") + the dmin/dmax band are set
        # per-robot in config/g1/env_cfgs.py.
        # ===================================================================
        # Crossed-legs fix: SIGNED lateral foot separation (y_L - y_R) in the
        # pelvis yaw-frame, scored against a comfortable band. Deliberately
        # PROMINENT weight — the user's analysis names this the single most
        # important reward for crossed legs, and a plain foot-DISTANCE term (which
        # uses the absolute gap) is structurally blind to which foot is on which
        # side. NEGATIVE separations (feet crossed) score ~0.
        "leg_width_ordering": RewardTermCfg(
            func=mdp.leg_width_ordering, weight=4.0,
            params={"dmin": 0.10, "dmax": 0.30, "margin": 0.10,
                    "asset_cfg": SceneEntityCfg("robot", site_names=())},
        ),
        # Geometric-mean CORE: (upright · double_support · support_margin ·
        # leg_ordering)^(1/4). No factor can be farmed while another is ~0. Modest
        # weight (3.0, same tier as post_joint_stillness / standing_alive_bonus) —
        # this is a NEW, unvalidated signal, not started at the post_base_height
        # tier. Raise once the A/B confirms it moves the final pose the right way.
        "post_terminal_core": RewardTermCfg(
            func=mdp.post_terminal_core, weight=3.0,
            params={"foot_sensor_name": "feet_ground_contact",
                    "dmin": 0.10, "dmax": 0.30, "width_margin": 0.10,
                    "upright_k": 4.0, "support_dist_scale": 8.0,
                    "asset_cfg": SceneEntityCfg("robot", site_names=())},
        ),
        # Worst-leg shank verticality penalty (min over legs, not a sum like the
        # rise-phase shank_orientation) — a vertical shank under each hip so the
        # final stand stacks instead of A-framing. NEGATIVE weight; gated to the
        # standing HOLD window only (never fights the bent-shank crouch).
        "shank_worst_leg": RewardTermCfg(
            func=mdp.shank_worst_leg, weight=-3.0,
            params={"min_cos": 0.9, "asset_cfg": SceneEntityCfg("robot")},
        ),
        # ===================================================================
        # v27 TASK-SPACE ARM/HAND terminal-pose terms. Decompose the coarse
        # arms_in_front (forward-X only, now weight 0 in config/g1) into four
        # unambiguous, separately-logged factors — mirroring the v26 leg
        # terminal-region family. Bands calibrated to the measured HOME arm pose
        # (torso frame: palm x≈0.157, |y|≈0.227, z≈-0.124, shoulder→palm≈0.422,
        # radial≈0.276). Palm site names ("left_palm","right_palm") are wired
        # per-robot in config/g1/env_cfgs.py; elbow/shoulder bodies resolve inside
        # the reward. All final-hold gated (ramp_from=None). Symmetry DEFERRED.
        # ===================================================================
        # Hand PLACEMENT: worst-hand 3D box (forward / outward-lateral / height) in
        # the torso frame. Signed lateral enforces left-hand-left / right-hand-right;
        # geometric mean per hand + worst-of so no axis and no hand can be farmed.
        # Inherits the old arms_in_front weight (5.0) — the primary arm term.
        "arm_hand_box": RewardTermCfg(
            func=mdp.arm_hand_box, weight=5.0,
            params={"x_lo": 0.12, "x_hi": 0.30, "y_lo": 0.12, "y_hi": 0.32,
                    "z_lo": -0.30, "z_hi": 0.00, "margin": 0.10,
                    "asset_cfg": SceneEntityCfg("robot", site_names=())},
        ),
        # ARM rest-pose band: ELBOW flexion (HOME 0.60) + v29 SHOULDER pitch (HOME
        # 0.20) + SHOULDER outward-roll (HOME ±0.20). Geometric mean of the three,
        # folded into ONE term (no new Episode_Reward entry). The shoulder factors
        # pin the redundant DOF that let the arm sweep back / roll against the torso
        # (the persistent left-arm-back-and-touching defect); roll factor is the
        # smooth reincarnation of the v9-ablated style_shoulder_roll_deviation.
        "arm_elbow_flexion": RewardTermCfg(
            func=mdp.arm_elbow_flexion, weight=2.0,
            params={"e_lo": 0.35, "e_hi": 0.95, "margin": 0.15,
                    "sp_lo": 0.00, "sp_hi": 0.45,   # shoulder pitch band (HOME 0.20)
                    "sr_lo": 0.05, "sr_hi": 0.60,   # shoulder OUTWARD-roll band (HOME 0.20)
                    "asset_cfg": SceneEntityCfg("robot")},
        ),
        # OVEREXTENSION penalty: squared hinge on shoulder→palm distance past l_max
        # (just above HOME reach ≈0.42). NEGATIVE weight; final-hold gated.
        "arm_overextension": RewardTermCfg(
            func=mdp.arm_overextension, weight=-2.0,
            params={"l_max": 0.50,
                    "asset_cfg": SceneEntityCfg("robot", site_names=())},
        ),
        # CLEARANCE penalty: torso-frame radial proxy, predictive anti-self-collision
        # (palm pinned to chest/back). NEGATIVE weight; final-hold gated so it never
        # fights the push-off phase where hands are legitimately near the body.
        # v29: d_safe 0.15 → 0.22 (closer to HOME radial ≈0.276 so the margin
        # actually engages before contact — at 0.15 it stayed pinned at ~0 all of
        # v28, teaching nothing) and weight -3 → -5 to give it real gradient.
        "arm_clearance": RewardTermCfg(
            func=mdp.arm_clearance, weight=-5.0,
            params={"d_safe": 0.22,
                    "asset_cfg": SceneEntityCfg("robot", site_names=())},
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
        # v28: after-standing fall termination (was defined but never registered).
        # Failure termination (no time_out=True): once the robot has stood (ever_stood)
        # and then stays below 0.65 m for n_fall_steps (~0.3 s), end the episode and
        # fire is_terminated. Stops post-fall ground-farming and gives a clear "don't
        # fall after standing" signal. Expect mean_episode_length to DROP for the
        # stood-then-fell envs — that is the intended failure signal, not a regression.
        "standing_fall_timeout": TerminationTermCfg(
            func=mdp.standing_fall_timeout,
            params={
                "height_threshold": 0.65,
                "n_fall_steps": 30,
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
