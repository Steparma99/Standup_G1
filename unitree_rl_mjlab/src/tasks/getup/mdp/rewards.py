"""Get-up task reward functions.

Translated from HumanUP legged_gym (Isaac Gym API) to mjlab (MuJoCo API).
Source: simulation/legged_gym/legged_gym/envs/g1waist/g1waist_up.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .events import get_episode_state

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _contact_force_norm(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return binary contact flag and contact-force norm [B,] for a named sensor."""
    sensor: ContactSensor = env.scene[sensor_name]

    found = sensor.data.found
    if found is None:
        in_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    elif found.ndim == 1:
        in_contact = found > 0
    else:
        in_contact = (found > 0).any(dim=1)

    force = sensor.data.force
    if force is None:
        force_norm = torch.zeros(env.num_envs, device=env.device)
    elif force.ndim == 2:
        force_norm = torch.norm(force, dim=-1)
    else:
        force_norm = torch.norm(force, dim=-1).max(dim=1).values

    return in_contact, force_norm


def base_height_exp(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward pelvis height rising toward target.

    Equivalent to HumanUP _reward_base_height_exp. Uses exp(clamp(z, 0, target)) - 1
    so the reward is zero at ground level and positive as the robot rises.
    """
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    z_rwd = torch.clamp(z, min=0.0, max=target_height)
    return torch.exp(z_rwd) - 1.0


def torso_height_exp(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward torso_link height rising toward target.

    Equivalent to HumanUP _reward_head_height_exp (head is highest body in G1
    without an explicit head link; torso_link is the uppermost tracked body).
    """
    asset: Entity = env.scene[asset_cfg.name]
    # body_link_pos_w: [B, N_bodies, 3] — body_ids selects the torso body.
    z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)  # [B]
    z_rwd = torch.clamp(z, min=0.0, max=target_height)
    return torch.exp(z_rwd) - 1.0


def body_up_exp(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the body being oriented upright (gravity vector pointing down in body frame).

    projected_gravity_b[:, 2]: negative = body Z points up (good), positive = body Z points down (bad).
    exp(-proj_grav_z) is high when proj_grav_z is negative (upright).
    Equivalent to HumanUP _reward_body_up_exp.
    """
    asset: Entity = env.scene[asset_cfg.name]
    z_axis = asset.data.projected_gravity_b[:, 2]  # [B]; +1 = facing down, -1 = facing up
    return torch.exp(-z_axis)


def stand_on_feet(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_height_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward when both feet are in ground contact and not elevated.

    Equivalent to HumanUP _reward_stand_on_feet.
    asset_cfg must specify site_names for left and right foot sites.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    in_contact = contact_sensor.data.found > 0  # [B, N_feet]
    both_feet_contact = torch.all(in_contact, dim=1)  # [B]

    asset: Entity = env.scene[asset_cfg.name]
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N_feet]
    feet_on_ground = torch.all(foot_heights < foot_height_threshold, dim=1)  # [B]

    return (both_feet_contact & feet_on_ground).float()


def head_contact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Binary penalty when the head touches the terrain [B,]."""
    in_contact, _ = _contact_force_norm(env, sensor_name)
    return in_contact.float()


def head_impact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 250.0,
    force_scale: float = 250.0,
) -> torch.Tensor:
    """Strong penalty for head impacts above a force threshold [B,].

    Returns 0 below threshold and then grows linearly with the excess force.
    """
    _, force_norm = _contact_force_norm(env, sensor_name)
    excess = torch.clamp(force_norm - force_threshold, min=0.0)
    return excess / max(force_scale, 1e-6)


def _standing_gate(
    pelvis_height: torch.Tensor, threshold: float, band: float = 0.12
) -> torch.Tensor:
    """Smooth [0,1] gate: 0 well below `threshold`, 1 well above.

    This is the minimal 'soft staging' primitive: instead of a hard on/off switch
    at a height threshold (a brittle state machine), rewards that only make sense
    once the robot is rising/standing are ramped in smoothly as a function of the
    pelvis height. Maps loosely to the spec's Stage 2/3 gating.
    """
    return torch.clamp(
        (pelvis_height - (threshold - band)) / (2.0 * band), 0.0, 1.0
    )


def dof_error_when_standing(
    env: ManagerBasedRlEnv,
    pelvis_height_threshold: float = 0.6,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint deviation from default pose, ramped in as the robot stands.

    Equivalent to HumanUP _reward_dof_error (version 3) but with a SMOOTH stage gate
    (see `_standing_gate`) instead of a hard cutoff. Avoids penalizing unusual joint
    poses while the robot is still getting up (which would fight the height reward),
    while not introducing a discontinuity at the threshold.
    """
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    dof_error = torch.sum(
        torch.square(asset.data.joint_pos - default_joint_pos), dim=1
    )  # [B]
    # Use pelvis height as standing proxy (no explicit head link in G1).
    pelvis_height = asset.data.root_link_pos_w[:, 2]  # [B]
    return dof_error * _standing_gate(pelvis_height, pelvis_height_threshold)


def feet_slip(
    env: ManagerBasedRlEnv,
    height_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize horizontal sliding of the feet while they are on the ground.

    Spec: "Foot slip / lateral force penalty". Uses foot height as the ground proxy
    (no dependency on contact-sensor ordering): for each foot near the ground, sum
    its horizontal speed. asset_cfg must specify the two foot site_names.
    """
    asset: Entity = env.scene[asset_cfg.name]
    heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
    vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
    speed = torch.norm(vel_xy, dim=-1)  # [B, N]
    on_ground = (heights < height_threshold).float()
    return torch.sum(speed * on_ground, dim=1)  # [B]


def feet_distance(
    env: ManagerBasedRlEnv,
    min_dist: float = 0.10,
    max_dist: float = 0.50,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the horizontal foot-to-foot gap being outside [min_dist, max_dist].

    Spec: "Feet distance range: evita piedi troppo larghi". Yaw-invariant (uses the
    Euclidean gap between the two foot sites). asset_cfg must specify two foot sites.
    """
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
    gap = torch.norm(pos[:, 0, :] - pos[:, 1, :], dim=-1)  # [B]
    too_wide = torch.clamp(gap - max_dist, min=0.0)
    too_narrow = torch.clamp(min_dist - gap, min=0.0)
    return too_wide + too_narrow  # [B], use with a negative weight


def height_progress(
    env: ManagerBasedRlEnv,
    max_step: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the upward movement of the pelvis since the last control step [B].

    Spec: "height progress — premia aumento progressivo altezza". The absolute-
    height rewards (base_height_exp) tell the policy WHERE to go; this one tells
    it it is going in the RIGHT DIRECTION. Especially important in the early
    training phases when the policy is still random and absolute-height rewards
    are near-zero everywhere.

    `max_step` caps the per-step reward to avoid sensitivity to timestep size
    and to prevent a single large jump from dominating the episode return.
    """
    asset: Entity = env.scene[asset_cfg.name]
    state = get_episode_state(env, asset)
    h = asset.data.root_link_pos_w[:, 2]
    delta = torch.clamp(h - state["prev_height"], min=0.0, max=max_step)
    state["prev_height"] = h.clone()  # update for next step
    return delta


def stable_success_hold(
    env: ManagerBasedRlEnv,
    n_hold: int = 50,
    height_threshold: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward staying above the standing height for n_hold consecutive steps [B].

    Spec: "stable success hold — deve restare in piedi per N step, non solo
    raggiungere altezza per un frame". Returns 1.0 only after the robot has
    been continuously above `height_threshold` for `n_hold` steps (0.5 s at
    100 Hz). Also tracks whether the robot has ever stood (used by the
    standing_fall_timeout termination).
    """
    asset: Entity = env.scene[asset_cfg.name]
    state = get_episode_state(env, asset)
    h = asset.data.root_link_pos_w[:, 2]
    above = h > height_threshold
    # Increment counter while above, reset to 0 the moment the robot dips below.
    state["standing_counter"] = torch.where(
        above,
        state["standing_counter"] + 1,
        torch.zeros_like(state["standing_counter"]),
    )
    # Once the robot has reached standing height in this episode, remember it.
    state["ever_stood"] = state["ever_stood"] | above
    return (state["standing_counter"] >= n_hold).float()


def prone_supine_righting(
    env: ManagerBasedRlEnv,
    height_threshold: float = 0.40,
    band: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward orienting the torso upright, gated to early-stage only [B,].

    Spec P1.1: incentivises leaving flat (supine/prone) poses. Uses the same
    orientation signal as body_up_exp but multiplied by an early-stage gate
    that fades to zero as the robot rises toward standing height.

    - proj_grav_b[:, 2] == -1 → body Z points up = upright (high reward)
    - proj_grav_b[:, 2] ==  0 → body flat (moderate reward, gradient exists)
    - proj_grav_b[:, 2] == +1 → body inverted (low reward)

    The gate is `1 - _standing_gate(h, threshold)`, so the reward is ~1 when
    the robot is low and ~0 when it is near-standing, preventing it from
    fighting the stand_on_feet / body_up_exp rewards at stage 3.
    """
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    proj_grav_z = asset.data.projected_gravity_b[:, 2]  # [B]; -1=up, +1=down
    orientation_reward = torch.exp(-proj_grav_z)  # same signal as body_up_exp
    early_gate = 1.0 - _standing_gate(h, height_threshold, band=band)
    return orientation_reward * early_gate


def com_over_support(
    env: ManagerBasedRlEnv,
    foot_sensor_name: str,
    min_height: float = 0.25,
    height_band: float = 0.15,
    dist_scale: float = 5.0,
    upright_lo: float = 0.3,
    upright_hi: float = 0.7,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the whole-body CoM projecting over the feet support area [B,].

    Stage-gated: only active when the robot is rising (Stage 1-3). Zero when
    the robot is flat on the ground so it cannot interfere with Stage-0 righting.

    Signal: Gaussian reward centred on zero offset between CoM XY and the
    midpoint of the two foot sites.  reward = exp(-dist_scale * dist²).

    Three multiplicative gates (all in [0, 1]):
      1. Height gate   — smooth ramp from min_height to min_height + height_band.
                         Zero when pelvis is flat on the floor, full at ~Stage 2.
      2. Upright gate  — smooth ramp as the torso rotates toward vertical.
                         Uses -proj_grav_z: 0 when flat, 1 when upright.
                         Activates between -proj_grav_z in [upright_lo, upright_hi].
      3. Foot contact  — hard gate: any foot must touch the terrain.
                         Without ground contact the support polygon is undefined.

    asset_cfg must have site_names = (left_foot, right_foot) set per-robot.
    foot_sensor_name must be a ContactSensor with found field, e.g. feet_ground_contact.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # --- Whole-body CoM XY (mujoco_warp subtree_com of root body) ----------
    # subtree_com[:, root_body_id] = CoM of the whole robot subtree. Offset
    # from the pelvis by ~9cm (G1 CoM is anterior to and slightly below pelvis).
    root_id = asset.data.indexing.root_body_id
    com_xy = asset.data.data.subtree_com[:, root_id, :2]  # [B, 2]

    # --- Feet midpoint XY ---------------------------------------------------
    foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
    feet_mid_xy = foot_pos_w.mean(dim=1)  # [B, 2]

    # --- Gaussian reward on CoM-to-midpoint distance ------------------------
    dist = torch.norm(com_xy - feet_mid_xy, dim=-1)          # [B], metres
    base_reward = torch.exp(-dist_scale * dist.pow(2))        # [B], in (0, 1]

    # --- Gate 1: height ------------------------------------------------------
    h = asset.data.root_link_pos_w[:, 2]
    height_gate = torch.clamp((h - min_height) / height_band, 0.0, 1.0)

    # --- Gate 2: uprightness ------------------------------------------------
    # proj_grav_z = -1 → upright; 0 → flat.  We use -proj_grav_z ∈ [0, 1].
    pg_z = asset.data.projected_gravity_b[:, 2]      # [B]
    uprightness = -pg_z                               # [B]; 1=upright, 0=flat
    upright_gate = torch.clamp(
        (uprightness - upright_lo) / max(upright_hi - upright_lo, 1e-6),
        0.0, 1.0,
    )

    # --- Gate 3: any foot contact -------------------------------------------
    foot_sensor = env.scene[foot_sensor_name]
    found = foot_sensor.data.found
    if found is not None:
        any_foot = (found > 0).any(dim=1).float()   # [B]
    else:
        any_foot = torch.ones(env.num_envs, device=env.device)

    return base_reward * height_gate * upright_gate * any_foot


def action_saturation(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize commanding a joint position target outside the physical joint limits.

    Spec: "penalty su saturation". A target beyond the soft joint limits is one the
    PD cannot reach without saturating torque, i.e. a non-deployable command. Reads
    the commanded (filtered) target from the action term and measures how far it
    exceeds the soft limits. (This is the sound version of the spec's actuator
    saturation penalty; a naive |target - q| penalty would instead punish motion.)
    """
    term = env.action_manager.get_term(action_name)
    target = term.applied_target  # [B, A] in target_ids order
    asset: Entity = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits[:, term.target_ids]  # [B, A, 2]
    over = torch.clamp(target - limits[..., 1], min=0.0) + torch.clamp(
        limits[..., 0] - target, min=0.0
    )
    return torch.sum(torch.square(over), dim=1)  # [B]
