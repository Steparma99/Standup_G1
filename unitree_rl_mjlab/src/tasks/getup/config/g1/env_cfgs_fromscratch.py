"""From-scratch staged-curriculum (V2) env config — PARALLEL to the legacy getup config.

Builds on the known-good `unitree_g1_getup_env_cfg()` and applies the from-scratch
OVERRIDES (M1 launchable baseline): incremental action semantics, the simplified 8-term
post-task reward set, recovery-reward gating, fixed canonical SUPINE + all randomization
OFF, and publishes the FromScratchCurriculumCfg on the env. The legacy task is untouched.

M2 wires the GLOBAL COUPLED assist-force + beta curriculum: one global lambda_F and one
global beta that advance TOGETHER on a conservative rolling success-rate trigger (K=128
episode terminations, tau_s=0.85, success = a sustained STABLE HOLD == the shared χ_t
is_stable indicator held for the final window, stable_counter >= success_window_steps).
No single lucky episode can retire the help — the whole population must sustain the
success rate before both the support force and the action authority step down. Still a
single canonical SUPINE pose with ALL randomization OFF (pose expansion = later milestone).
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.getup.mdp as mdp
from src.assets.robots.unitree_g1.g1_constants import HOME_KEYFRAME, SUPINE_KEYFRAME
from src.tasks.getup.curriculum_cfg import FromScratchCurriculumCfg
from .env_cfgs import unitree_g1_getup_env_cfg


# ---- banded nominal-pose (reward E) importance + tolerance bands --------------------
# high legs/waist, medium shoulders/elbows, low wrists, ZERO fingers (held). Wider bands
# for the balance joints (ankles, hips, knees, waist pitch/roll) so small corrections are
# free. target = HOME.
_FS_POSE_WEIGHTS = {
    ".*_hip_pitch_joint": 3.0, ".*_hip_roll_joint": 3.0, ".*_hip_yaw_joint": 3.0,
    ".*_knee_joint": 3.0, ".*_ankle_pitch_joint": 1.5, ".*_ankle_roll_joint": 1.5,
    "waist_yaw_joint": 2.0, "waist_roll_joint": 3.0, "waist_pitch_joint": 3.0,
    ".*_shoulder_pitch_joint": 2.0, ".*_shoulder_roll_joint": 2.0,
    ".*_shoulder_yaw_joint": 1.5, ".*_elbow_joint": 2.0,
    ".*_wrist_roll_joint": 0.5, ".*_wrist_pitch_joint": 0.5, ".*_wrist_yaw_joint": 0.5,
}
_FS_POSE_BANDS = {
    ".*_hip_pitch_joint": 0.15, ".*_hip_roll_joint": 0.12, ".*_hip_yaw_joint": 0.10,
    ".*_knee_joint": 0.15, ".*_ankle_pitch_joint": 0.20, ".*_ankle_roll_joint": 0.20,
    "waist_yaw_joint": 0.15, "waist_roll_joint": 0.10, "waist_pitch_joint": 0.10,
    ".*_shoulder_pitch_joint": 0.20, ".*_shoulder_roll_joint": 0.20,
    ".*_shoulder_yaw_joint": 0.25, ".*_elbow_joint": 0.20,
    ".*_wrist_roll_joint": 0.35, ".*_wrist_pitch_joint": 0.35, ".*_wrist_yaw_joint": 0.35,
}

# post-task legacy terms zeroed in the clean baseline (spec 13.5) + anti_jump (replaced).
_DISABLE_POST_TASK = (
    "post_base_ang_vel", "post_base_lin_vel", "post_base_orientation", "post_base_height",
    "post_stand_on_feet", "post_feet_parallel", "post_feet_yaw", "stable_success_hold",
    "post_joint_stillness", "standing_alive_bonus", "leg_width_ordering",
    "post_terminal_core", "shank_worst_leg", "arm_clearance", "pose_legs", "pose_waist",
    "pose_upper", "pose_legs_l2", "pose_waist_l2", "pose_upper_l2",
    "post_upper_body_posture", "post_standing_posture", "post_home_pose_l2",
    "post_bilateral_symmetry", "arms_in_front", "arm_hand_box", "arm_elbow_flexion",
    "arm_overextension", "anti_jump_velocity",
)
# style terms zeroed in the clean baseline (spec 13.2).
_DISABLE_STYLE = ("style_foot_displacement", "style_foot_distance", "style_ankle_parallel")


def unitree_g1_getup_fromscratch_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_getup_env_cfg(play=play)
    curr = FromScratchCurriculumCfg()

    # --- publish curriculum cfg on the env (startup event) -----------------------
    cfg.events["attach_fromscratch_cfg"] = EventTermCfg(
        func=mdp.attach_fromscratch_cfg, mode="startup", params={"cfg": curr},
    )

    # --- 7. incremental action semantics -----------------------------------------
    old = cfg.actions["joint_pos"]
    a = curr.action
    cfg.actions["joint_pos"] = mdp.IncrementalJointPositionActionCfg(
        entity_name=old.entity_name,
        actuator_names=old.actuator_names,
        scale=1.0,                       # unused by increment, kept for base ctor
        use_default_offset=False,
        alpha=a.ema_alpha,               # 1.0 -> EMA off
        max_rate=a.slew_rate_limit,      # None -> normal slew off
        settle_steps=old.settle_steps,   # keep the settling window
        liftoff_steps=0,                 # no liftoff phase in the incremental branch
        default_pos_override=None,       # incremental anchors on q_current, not HOME
        delta_scale=dict(a.delta_scale),
        default_delta_scale=a.default_delta_scale,
        action_ramp_steps=a.action_ramp_steps,
        emergency_rate_limit_rad_s=a.emergency_rate_limit_rad_s,
    )

    # --- 5/9. fixed canonical SUPINE, ALL randomization OFF -----------------------
    cfg.events["reset_pose"].params["keyframes"] = (SUPINE_KEYFRAME,)
    cfg.events["reset_pose"].params["joint_pos_range"] = {}
    cfg.events["reset_pose"].params["pose_range"] = {}
    for ev in ("encoder_bias", "base_com", "foot_friction"):
        cfg.events.pop(ev, None)         # remove stochastic startup randomization

    # --- 6/7. GLOBAL COUPLED assist-force + beta curriculum (M2) -------------------
    # Replace the per-env HoST AssistanceCurriculum + BetaRescalerCurriculum with ONE
    # global lambda_F and ONE global beta that advance TOGETHER on a conservative
    # rolling success-rate trigger (K=128 episode-ends, tau_s=0.85). The policy learns
    # a robust stand at each level before ANY help is withdrawn — no single lucky
    # episode can retire the support (the HoST failure mode). Both the assist metric
    # (curriculum/assistance_force) and beta metric (curriculum/beta_rescaler) read
    # the global scalars via the same env attrs, so their plumbing is unchanged.
    for ev in ("assistance_curriculum", "beta_rescaler_curriculum"):
        cfg.events.pop(ev, None)
    # The per-env breaker metrics are meaningless for the global curriculum.
    for m in ("curriculum/beta_success_ema", "curriculum/beta_paused"):
        cfg.metrics.pop(m, None)

    co = curr.coupled
    cfg.events["fromscratch_assist_beta_curriculum"] = EventTermCfg(
        mode="step",
        func=mdp.FromScratchAssistBetaCurriculum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "head_body": co.head_body,
            "success_window_steps": co.success_window_steps,
            "window_K": co.window_K,
            "success_rate_threshold": co.success_rate_threshold,
            "lambda_F_init": co.lambda_F_init,
            "lambda_F_min": co.lambda_F_min,
            "delta_lambda_F": co.delta_lambda_F,
            "beta_init": co.beta_init,
            "beta_min": co.beta_min,
            "delta_beta": co.delta_beta,
            "assist_unactuated_steps": co.assist_unactuated_steps,
        },
    )
    # assist force + beta level metrics (read the global broadcast tensors).
    cfg.metrics["curriculum/assistance_force"] = MetricsTermCfg(func=mdp.assistance_force)
    cfg.metrics["curriculum/beta_rescaler"] = MetricsTermCfg(func=mdp.beta_rescaler_value)
    # coupled-trigger diagnostics.
    cfg.metrics["curriculum/fs_success_rate"] = MetricsTermCfg(func=mdp.fs_curriculum_success_rate)
    cfg.metrics["curriculum/fs_buffer_fill"] = MetricsTermCfg(func=mdp.fs_curriculum_buffer_fill)
    cfg.metrics["curriculum/fs_advances"] = MetricsTermCfg(func=mdp.fs_curriculum_advances)

    # ============================================================================
    # 13. Reward simplification
    # ============================================================================
    R = cfg.rewards

    # 13.1 recovery rewards -> gated wrappers (zero after standing latched)
    R["height_progress"].func = mdp.fs_height_progress
    R["supine_rising_prep"].func = mdp.fs_supine_rising_prep
    R["pelvis_height_bridge"].func = mdp.fs_pelvis_height_bridge
    R["prone_recovery"].func = mdp.fs_prone_recovery
    R["prone_recovery"].weight = 0.0     # enabled only once PRONE is introduced (M3)

    # 13.2 style: keep 5, zero 3
    for name in _DISABLE_STYLE:
        R[name].weight = 0.0

    # 13.4/13.5 post-task: zero the redundant legacy set, reuse com_over_support (+6)
    for name in _DISABLE_POST_TASK:
        if name in R:
            R[name].weight = 0.0
    R["com_over_support"].weight = 6.0   # C. reuse existing implementation

    # 13.4 new 8-term post-task set (A,B,D,E,F,G,H) — added as new terms
    R["fs_upright"] = RewardTermCfg(
        func=mdp.fs_upright, weight=6.0,
        params={"asset_cfg": SceneEntityCfg("robot")})
    R["fs_double_support"] = RewardTermCfg(
        func=mdp.fs_double_support, weight=4.0,
        params={"sensor_name": "feet_ground_contact", "asset_cfg": SceneEntityCfg("robot")})
    R["fs_dynamic_stillness"] = RewardTermCfg(
        func=mdp.fs_dynamic_stillness, weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot")})
    R["fs_banded_pose"] = RewardTermCfg(
        func=mdp.fs_banded_pose, weight=3.0,
        params={"target_joint_pos": dict(HOME_KEYFRAME.joint_pos),
                "joint_weights": dict(_FS_POSE_WEIGHTS), "joint_bands": dict(_FS_POSE_BANDS),
                "scale": 1.0, "k": 1.0, "asset_cfg": SceneEntityCfg("robot")})
    R["fs_progressive_hold"] = RewardTermCfg(
        func=mdp.fs_progressive_hold, weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot")})
    R["fs_success_bonus"] = RewardTermCfg(
        func=mdp.fs_success_bonus, weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot")})
    R["fs_anti_jump"] = RewardTermCfg(
        func=mdp.fs_anti_jump, weight=-2.0,
        params={"v_thresh": 1.5, "asset_cfg": SceneEntityCfg("robot")})

    return cfg
