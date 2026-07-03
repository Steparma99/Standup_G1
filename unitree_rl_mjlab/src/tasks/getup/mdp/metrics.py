"""Get-up task diagnostic metrics.

All functions return a [B,] float tensor per control step. The MetricsManager
accumulates episode sums and emits per-episode averages (step-count-normalised)
into extras["log"] under the key prefix "Episode_Metrics/".

Keys that contain "/" are forwarded to tensorboard/wandb by the rsl_rl logger.

Naming convention mirrors the P1.2/P1.3/contact-logging spec:
  stage/stage0..3        — soft stage occupancy fractions
  success/...            — per-episode success signals
  action/...             — action saturation / smoothness
  torque/...             — actuator force saturation
  contact/...            — per-body-part ground contact (binary + force / impact)
  dr/...                 — domain-randomization sanity checks (logged via events)

The stage gate thresholds here MUST match (or be looser than) the thresholds
used in the rewards (getup_env_cfg.py) to avoid mislabelling stages.

Contact sensors used below (defined in config/g1/env_cfgs.py):
  feet_ground_contact  — left+right ankle subtrees (existing)
  contact_head         — head_collision geom
  contact_torso        — torso_collision + pelvis_collision geoms
  contact_knee_left/right  — shin + linkage-brace geoms
  contact_forearm_left/right — elbow + wrist geoms
  contact_hand_left/right   — hand geoms

ContactSensor data shapes (after build):
  found : [B, N_primaries] or [B]
  force : [B, N_primaries, 3] or [B, 3]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .events import _ASSIST_FORCE_ATTR, _BETA_RESCALER_ATTR, get_episode_state

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_action_term(env: "ManagerBasedRlEnv", action_name: str = "joint_pos"):
    return env.action_manager.get_term(action_name)


def assistance_force(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Per-env assistance-curriculum support force [B,] in Newtons.

    Reads the live per-env force buffer published by AssistanceCurriculum. The
    episode average should DECREASE over training as envs learn to stand and
    their support force decays toward zero. Returns zeros when the curriculum is
    disabled (attribute absent).
    """
    force = getattr(env, _ASSIST_FORCE_ATTR, None)
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    return force


def beta_rescaler_value(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Per-env action-rescaler beta [B,] (HoST action-scale curriculum).

    Reads the live per-env beta published by BetaRescalerCurriculum. The episode
    average should DECREASE over training (1.0 -> 0.25 floor) as envs learn to
    stand and their action authority anneals. Returns ones when disabled.
    """
    beta = getattr(env, _BETA_RESCALER_ATTR, None)
    if beta is None:
        return torch.ones(env.num_envs, device=env.device)
    return beta

# ---------------------------------------------------------------------------
# Stage boundaries (pelvis height in metres, matching reward gates)
# ---------------------------------------------------------------------------
_H_STANDING = 0.65   # robot is "standing" (stable_success threshold)
_H_RISING   = 0.40   # robot is actively rising (Stage 2 entry)
_H_SUPPORT  = 0.20   # robot has left the floor (Stage 1 entry)


def _soft_gate(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Linearly ramp 0→1 as x goes from lo to hi."""
    return torch.clamp((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)


# ---------------------------------------------------------------------------
# P1.2 — Stage occupancy
# ---------------------------------------------------------------------------

def stage0_fraction(
    env: "ManagerBasedRlEnv",
    h_support: float = _H_SUPPORT,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Fraction of envs in Stage 0 (fallen / righting, h < h_support) [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    return (h < h_support).float()


def stage1_fraction(
    env: "ManagerBasedRlEnv",
    h_support: float = _H_SUPPORT,
    h_rising: float = _H_RISING,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Fraction of envs in Stage 1 (support-transition, h_support ≤ h < h_rising) [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    return ((h >= h_support) & (h < h_rising)).float()


def stage2_fraction(
    env: "ManagerBasedRlEnv",
    h_rising: float = _H_RISING,
    h_standing: float = _H_STANDING,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Fraction of envs in Stage 2 (rising, h_rising ≤ h < h_standing) [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    return ((h >= h_rising) & (h < h_standing)).float()


def stage3_fraction(
    env: "ManagerBasedRlEnv",
    h_standing: float = _H_STANDING,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Fraction of envs in Stage 3 (standing, h ≥ h_standing) [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    return (h >= h_standing).float()


# ---------------------------------------------------------------------------
# P1.2 — Success metrics
# ---------------------------------------------------------------------------

def success_candidate(
    env: "ManagerBasedRlEnv",
    height_threshold: float = _H_STANDING,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """1 when robot is above standing threshold (success candidate) [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    return (h > height_threshold).float()


def stable_success_active(
    env: "ManagerBasedRlEnv",
    n_hold: int = 50,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """1 when robot has held standing for n_hold consecutive steps [B,].

    Reads the standing_counter from episode_state (written by the
    stable_success_hold reward). Returns 1 if counter ≥ n_hold.
    """
    asset: Entity = env.scene[asset_cfg.name]
    state = get_episode_state(env, asset)
    return (state["standing_counter"] >= n_hold).float()


def ever_stood_fraction(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """1 if the robot has ever reached standing height this episode [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    state = get_episode_state(env, asset)
    return state["ever_stood"].float()


def fall_after_success_active(
    env: "ManagerBasedRlEnv",
    height_threshold: float = _H_STANDING,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """1 if robot has ever stood AND is currently below standing threshold [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    state = get_episode_state(env, asset)
    h = asset.data.root_link_pos_w[:, 2]
    return (state["ever_stood"] & (h < height_threshold)).float()


# ---------------------------------------------------------------------------
# P1.2 — Termination reason distribution
#
# Each function returns 1 on the step this env terminates for that reason,
# 0 otherwise. Episode average ≈ 1/episode_length when it fires exactly once,
# giving a relative measure of which reason dominates.
# ---------------------------------------------------------------------------

def termination_timeout(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """1 on the step an env terminates due to time_out [B,]."""
    return env.reset_time_outs.float()


def termination_failure(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """1 on the step an env terminates due to a non-timeout failure [B,]."""
    return env.reset_terminated.float()


def termination_fall_after_success(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """1 on the step an env terminates and the robot had previously stood [B,].

    Subset of termination_failure — specifically the case where the robot
    succeeded but then fell (the standing_fall_timeout termination).
    """
    asset: Entity = env.scene[asset_cfg.name]
    state = get_episode_state(env, asset)
    return (env.reset_terminated & state["ever_stood"]).float()


def termination_head_impact(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    force_threshold: float = 1200.0,
) -> torch.Tensor:
    """1 on the step an env terminates while head-impact force exceeds threshold."""
    return (
        env.reset_terminated
        & (contact_force_max(env, sensor_name) > force_threshold)
    ).float()


# ---------------------------------------------------------------------------
# P1.3 — Action saturation / smoothness metrics
# ---------------------------------------------------------------------------

def action_norm_mean(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Mean L2 norm of the raw policy action [B,]."""
    a = env.action_manager.action  # [B, A] raw (unscaled) action
    return torch.norm(a, dim=-1)


def action_saturation_fraction(
    env: "ManagerBasedRlEnv",
    threshold: float = 0.95,
) -> torch.Tensor:
    """Fraction of action dimensions near ±1 (raw action, before scaling) [B,].

    If threshold=0.95, fires for any joint whose raw action abs value > 0.95.
    The policy outputs actions in roughly [-1, 1]; this captures over-saturation.
    """
    a = env.action_manager.action  # [B, A]
    saturated = (a.abs() > threshold).float()
    return saturated.mean(dim=-1)  # [B]


def action_rate_mean(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Mean per-joint action rate ||a_t - a_{t-1}|| [B,]."""
    diff = env.action_manager.action - env.action_manager.prev_action
    return torch.norm(diff, dim=-1)


def action_acc_mean(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Mean per-joint action acceleration ||a_t - 2a_{t-1} + a_{t-2}|| [B,]."""
    acc = (
        env.action_manager.action
        - 2.0 * env.action_manager.prev_action
        + env.action_manager.prev_prev_action
    )
    return torch.norm(acc, dim=-1)


def raw_action_min(
    env: "ManagerBasedRlEnv",
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Minimum raw policy action before defensive clamp [B,]."""
    term = _get_action_term(env, action_name)
    return term.raw_policy_actions.min(dim=-1).values


def raw_action_max(
    env: "ManagerBasedRlEnv",
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Maximum raw policy action before defensive clamp [B,]."""
    term = _get_action_term(env, action_name)
    return term.raw_policy_actions.max(dim=-1).values


def raw_action_clip_fraction(
    env: "ManagerBasedRlEnv",
    threshold: float = 1.0,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Fraction of raw policy dimensions outside [-threshold, threshold] [B,]."""
    term = _get_action_term(env, action_name)
    raw = term.raw_policy_actions
    return (raw.abs() > threshold).float().mean(dim=-1)


# ---------------------------------------------------------------------------
# P1.3 — Torque / actuator force saturation
# ---------------------------------------------------------------------------

def _get_effort_limits(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    """Return actuator force limits as a [nu,] tensor.

    Reads from sim.model.actuator_forcerange (per-env; take env 0 as reference
    since limits are shared unless DR is active).
    """
    ctrl_ids = asset.indexing.ctrl_ids  # local actuator IDs [nu]
    # forcerange shape: [n_envs, n_actuators_total, 2] or [n_actuators_total, 2]
    fr = env.sim.model.actuator_forcerange
    if fr.ndim == 3:
        limits = fr[0, ctrl_ids, :]  # [nu, 2]
    else:
        limits = fr[ctrl_ids, :]     # [nu, 2]
    return limits[:, 1]  # positive side


def torque_norm_mean(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mean L2 norm of actuator forces [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force  # [B, nu]
    return torch.norm(tau, dim=-1)


def torque_saturation_fraction(
    env: "ManagerBasedRlEnv",
    threshold: float = 0.90,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Fraction of actuators near their effort limit [B,].

    A motor is "saturated" when |tau| / tau_limit > threshold.
    """
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force  # [B, nu]
    limits = _get_effort_limits(env, asset)  # [nu]
    ratio = tau.abs() / (limits.unsqueeze(0) + 1e-6)  # [B, nu]
    saturated = (ratio > threshold).float()
    return saturated.mean(dim=-1)


def torque_power_mean(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mean mechanical power |tau * qdot| across actuated joints [B,].

    Crude sim2real proxy: excessively high power consumption is suspicious.
    """
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force  # [B, nu]
    # joint_vel is ordered by joint/DOF id; use the actuator ctrl_ids subset.
    ctrl_ids = asset.indexing.ctrl_ids
    qvel = asset.data.joint_vel[:, ctrl_ids]  # [B, nu]
    power = (tau * qvel).abs()
    return power.mean(dim=-1)


def joint_target_clamp_fraction(
    env: "ManagerBasedRlEnv",
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Fraction of target dimensions that exceeded joint limits before clamp [B,]."""
    term = _get_action_term(env, action_name)
    target = getattr(term, "filtered_target_unclamped", term.applied_target)
    asset: Entity = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits[:, term.target_ids]
    clamped = (target < limits[..., 0]) | (target > limits[..., 1])
    return clamped.float().mean(dim=-1)


def joint_target_over_limit_mean(
    env: "ManagerBasedRlEnv",
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mean absolute target exceedance beyond joint limits [B,]."""
    term = _get_action_term(env, action_name)
    target = getattr(term, "filtered_target_unclamped", term.applied_target)
    asset: Entity = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits[:, term.target_ids]
    over = torch.clamp(target - limits[..., 1], min=0.0) + torch.clamp(
        limits[..., 0] - target, min=0.0
    )
    return over.mean(dim=-1)


def joint_target_over_limit_max(
    env: "ManagerBasedRlEnv",
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Maximum absolute target exceedance beyond joint limits [B,]."""
    term = _get_action_term(env, action_name)
    target = getattr(term, "filtered_target_unclamped", term.applied_target)
    asset: Entity = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits[:, term.target_ids]
    over = torch.clamp(target - limits[..., 1], min=0.0) + torch.clamp(
        limits[..., 0] - target, min=0.0
    )
    return over.max(dim=-1).values


def pd_tracking_error_mean(
    env: "ManagerBasedRlEnv",
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mean absolute PD tracking error |q_des - q| across joints [B,]."""
    term = _get_action_term(env, action_name)
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, term.target_ids]
    err = (term.applied_target - q).abs()
    return err.mean(dim=-1)


def pd_tracking_error_max(
    env: "ManagerBasedRlEnv",
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Maximum absolute PD tracking error |q_des - q| across joints [B,]."""
    term = _get_action_term(env, action_name)
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, term.target_ids]
    err = (term.applied_target - q).abs()
    return err.max(dim=-1).values


def pd_tracking_error_l2_mean(
    env: "ManagerBasedRlEnv",
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L2 norm of PD tracking error, normalised by sqrt(num_joints) [B,]."""
    term = _get_action_term(env, action_name)
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, term.target_ids]
    err = term.applied_target - q
    return torch.norm(err, dim=-1) / (err.shape[-1] ** 0.5)


def pd_tracking_error_standing_mean(
    env: "ManagerBasedRlEnv",
    action_name: str = "joint_pos",
    h_standing: float = _H_STANDING,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mean absolute PD tracking error while standing; zero otherwise [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    standing = (asset.data.root_link_pos_w[:, 2] >= h_standing).float()
    return pd_tracking_error_mean(env, action_name=action_name, asset_cfg=asset_cfg) * standing


# ---------------------------------------------------------------------------
# P1.3 — Stage-aware saturation (stage × action/torque saturation)
# ---------------------------------------------------------------------------

def stage0_action_saturation(
    env: "ManagerBasedRlEnv",
    threshold: float = 0.95,
    h_support: float = _H_SUPPORT,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Action saturation fraction for envs in Stage 0 [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    mask = (h < h_support).float()
    a = env.action_manager.action
    sat = (a.abs() > threshold).float().mean(dim=-1)
    return sat * mask


def stage3_action_saturation(
    env: "ManagerBasedRlEnv",
    threshold: float = 0.95,
    h_standing: float = _H_STANDING,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Action saturation fraction for envs in Stage 3 [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    mask = (h >= h_standing).float()
    a = env.action_manager.action
    sat = (a.abs() > threshold).float().mean(dim=-1)
    return sat * mask


def stage0_torque_saturation(
    env: "ManagerBasedRlEnv",
    threshold: float = 0.90,
    h_support: float = _H_SUPPORT,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Torque saturation fraction for envs in Stage 0 [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    mask = (h < h_support).float()
    tau = asset.data.actuator_force
    limits = _get_effort_limits(env, asset)
    ratio = tau.abs() / (limits.unsqueeze(0) + 1e-6)
    sat = (ratio > threshold).float().mean(dim=-1)
    return sat * mask


def stage3_torque_saturation(
    env: "ManagerBasedRlEnv",
    threshold: float = 0.90,
    h_standing: float = _H_STANDING,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Torque saturation fraction for envs in Stage 3 [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    mask = (h >= h_standing).float()
    tau = asset.data.actuator_force
    limits = _get_effort_limits(env, asset)
    ratio = tau.abs() / (limits.unsqueeze(0) + 1e-6)
    sat = (ratio > threshold).float().mean(dim=-1)
    return sat * mask


# ---------------------------------------------------------------------------
# Contact logging (spec 1.1)
#
# Two generic helpers + named wrappers per body part.
#
# contact_found():      1 if ANY primary geom in the sensor group contacts
#                       the terrain this step, 0 otherwise. [B,]
#                       Episode average = fraction of steps in contact.
#
# contact_force_max():  L2 magnitude of the strongest contact force across
#                       all primaries in the sensor group. [B,]
#                       Episode average = mean peak force per step.
#
# Sensor shapes (produced by ContactSensorCfg mode="geom", reduce="maxforce"):
#   found : [B, N_primaries]     (0=no contact, >0=count)
#   force : [B, N_primaries, 3]  (force vector of max contact)
# ---------------------------------------------------------------------------

def contact_found(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
) -> torch.Tensor:
    """Binary floor contact for a body part group [B,].

    Returns 1.0 when ANY primary geom in the named sensor has terrain contact,
    0.0 otherwise. Episode average = fraction of steps in contact with floor.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found  # [B, N_primaries]
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    if found.ndim == 1:
        return (found > 0).float()
    return (found > 0).any(dim=1).float()


def contact_force_max(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
) -> torch.Tensor:
    """Maximum contact force magnitude across all primaries [B,] in Newtons.

    Uses the force vector of the strongest contact (reduce="maxforce") for
    each primary, then takes the maximum L2 norm across primaries.
    Episode average ≈ mean peak impact force per control step.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    force = sensor.data.force
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    mag = torch.norm(force, dim=-1)
    if mag.ndim == 1:
        return mag
    return mag.max(dim=1).values


def contact_impact(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    force_threshold: float = 250.0,
) -> torch.Tensor:
    """Binary metric for contacts whose force exceeds a threshold [B,]."""
    return (contact_force_max(env, sensor_name) > force_threshold).float()


# ---------------------------------------------------------------------------
# Per-group reward sums (the four multi-critic groups). Each returns its group's
# summed reward RATE; the MetricsManager episode-averages it, so the logged
# Episode_Metrics/reward_group/<g> equals the SUM of that group's per-term
# Episode_Reward entries — i.e. the raw reward each critic sees BEFORE per-group
# normalization and the (2.5/1.9/0.1/1.0) group weights. Lets you read the task
# (positive) vs regularization (negative) signals separately on one panel.
# ---------------------------------------------------------------------------

_GROUP_ONEHOT_ATTR = "_metric_group_onehot"


def _grouped_step_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """[B, G] per-group summed reward rate; columns follow reward_groups.GROUP_ORDER.

    Applies the same settling gate as the training signal (gate_settling_rewards): this
    metric runs at env.step line ~389, BEFORE that step-event zeroes _step_reward, so we
    reproduce the gate here to keep the logged reward_group/* panels consistent with what
    the policy actually trains on. metrics runs BEFORE _reset_idx, so every env still holds
    its true episode_length_buf and no reset-guard is needed (unlike the event).
    """
    rm = env.reward_manager
    onehot = getattr(env, _GROUP_ONEHOT_ATTR, None)
    if onehot is None:
        # Deferred import: avoids triggering the rl package at mdp-import time.
        from src.tasks.getup.rl.reward_groups import build_group_onehot

        onehot = build_group_onehot(list(rm.active_terms), device=env.device)
        setattr(env, _GROUP_ONEHOT_ATTR, onehot)
    grouped = rm._step_reward @ onehot  # [B, num_terms] @ [num_terms, G] -> [B, G]
    # Deferred import (runtime) avoids the env_cfgs <-> mdp circular import at load time.
    from src.tasks.getup.config.g1.env_cfgs import _SETTLE_STEPS

    if _SETTLE_STEPS > 0:
        gate = (env.episode_length_buf > _SETTLE_STEPS).float().unsqueeze(1)  # [B, 1]
        grouped = grouped * gate
    return grouped


# Column order matches reward_groups.GROUP_ORDER = (task, style, regularization, post_task).
def reward_group_task(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _grouped_step_reward(env)[:, 0]


def reward_group_style(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _grouped_step_reward(env)[:, 1]


def reward_group_regularization(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _grouped_step_reward(env)[:, 2]


def reward_group_post_task(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return _grouped_step_reward(env)[:, 3]
