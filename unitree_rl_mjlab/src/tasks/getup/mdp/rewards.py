"""Get-up task reward functions.

Translated from HumanUP legged_gym (Isaac Gym API) to mjlab (MuJoCo API).
Source: simulation/legged_gym/legged_gym/envs/g1waist/g1waist_up.py
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.string import resolve_expr

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
    k: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward TRUE torso uprightness (torso vertical), not merely "belly up".

    proj_grav_b[:, 2] = -cos(tilt): -1 when the torso Z axis is vertical (upright),
    0 when the torso is horizontal (supine/prone/side), +1 when inverted.

    The previous `exp(-proj_grav_z)` gave a large baseline (1.0) to ALL flat poses
    and saturated near vertical (a 14° seated pose scored 2.64 vs 2.72 standing —
    almost no gradient). It also could not tell upright from flat.

    New form `exp(-k·(1 + proj_grav_z))` = exp(-k·(1 - cos(tilt))):
      - upright (proj_grav_z=-1) -> 1.0
      - flat    (proj_grav_z= 0) -> exp(-k) ~ 0.018  (k=4)
      - inverted(proj_grav_z=+1) -> exp(-2k) ~ 0      (inversion-safe, unlike a
        plain exp(-k·tilt_xy²) which would wrongly score an inverted pose ~1.0)
    so it is ~0 from any flat pose and rises sharply only as the torso verticalises.
    """
    asset: Entity = env.scene[asset_cfg.name]
    z_axis = asset.data.projected_gravity_b[:, 2]  # [B]; -1 = upright, +1 = inverted
    return torch.exp(-k * (1.0 + z_axis))


def task_stand(
    env: ManagerBasedRlEnv,
    target_height: float = 1.1,
    k_up: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """MULTIPLICATIVE get-up task reward: upright AND tall, in [0, 1] [B,].

    Replaces the additive trio (base_height_exp + torso_height_exp + body_up_exp).
    Those summed terms could be farmed by satisfying ONE of them — the policy rolled
    into a half-propped pose (body_up≈0.8) and parked, banking the sum for the whole
    episode without ever rising. HoST avoids this by MULTIPLYING the task factors
    (orientation × head_height):

        reward = up_term · height_term

      - height_term = clamp(z, 0, target)/target  ∈ [0, 1]  (HEAD/TORSO height, like
                      HoST's head_height — uses the upper body, not the pelvis, so it
                      tracks "head up"; asset_cfg.body_ids selects torso_link)
      - up_term     = exp(-k·(1 + proj_grav_z))   ∈ [0, 1]  (torso uprightness;
                      1 upright, ~0.018 flat, ~0 inverted — same form as body_up_exp)

    The product is high ONLY when the robot is upright AND tall: you cannot raise it
    by improving one factor while the other stays near zero, so the half-propped
    local optimum is no longer rewarded. NOTE: the product has a near-zero gradient
    from a fully flat pose (both factors ~0), so the additive stage-0 bootstrap
    rewards (height_progress / prone_recovery / supine_rising_prep) and the assist
    curriculum are what get the robot off the floor; this term takes over the rise.
    """
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)  # head/torso height [B]
    height_term = torch.clamp(z, 0.0, target_height) / target_height
    up_term = torch.exp(-k_up * (1.0 + asset.data.projected_gravity_b[:, 2]))
    return up_term * height_term


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

    asset: Entity = env.scene[asset_cfg.name]
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N_feet]
    foot_low = foot_heights < foot_height_threshold  # [B, N_feet]

    # Graded instead of binary: a foot counts as "planted" when it is both in
    # contact AND low. During the rising transition the policy gets partial
    # credit for one planted foot (0.5) and full credit for both (1.0), giving a
    # smoother gradient than the old all-or-nothing both-feet flag.
    planted = in_contact & foot_low  # [B, N_feet]
    any_foot = planted.any(dim=1).float()  # [B]
    both_feet = planted.all(dim=1).float()  # [B]
    return 0.5 * any_foot + 0.5 * both_feet


def _reset_ramp(env: ManagerBasedRlEnv, ramp_steps: int) -> torch.Tensor | float:
    """Per-env temporal mask in [0, 1] that ramps 0 -> 1 over the first `ramp_steps`
    env-steps of an episode (linear in episode_length_buf).

    Used to fade IN destructive/impact penalties so the unavoidable spawn-landing
    contact during the settling window does not unfairly penalize the policy. With
    ramp_steps <= 0 it is a no-op (constant 1.0).
    """
    if ramp_steps <= 0:
        return 1.0
    return torch.clamp(
        env.episode_length_buf.float() / float(ramp_steps), 0.0, 1.0
    )


def head_contact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    ramp_steps: int = 0,
) -> torch.Tensor:
    """Binary penalty when the head touches the terrain [B,].

    Masked over the first `ramp_steps` steps (see _reset_ramp) so settling contacts
    are not penalized.
    """
    in_contact, _ = _contact_force_norm(env, sensor_name)
    return in_contact.float() * _reset_ramp(env, ramp_steps)


def head_impact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 250.0,
    force_scale: float = 250.0,
    ramp_steps: int = 0,
) -> torch.Tensor:
    """Strong penalty for head impacts above a force threshold [B,].

    Returns 0 below threshold and then grows linearly with the excess force. Masked
    over the first `ramp_steps` steps (see _reset_ramp) so the spawn-landing impact
    during the settling window is not penalized.
    """
    _, force_norm = _contact_force_norm(env, sensor_name)
    excess = torch.clamp(force_norm - force_threshold, min=0.0)
    return (excess / max(force_scale, 1e-6)) * _reset_ramp(env, ramp_steps)


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


_POSTURE_CACHE_ATTR = "_standing_posture_cache"


def standing_posture(
    env: ManagerBasedRlEnv,
    target_joint_pos: dict[str, float],
    joint_weights: dict[str, float],
    pelvis_height_threshold: float = 0.6,
    band: float = 0.12,
    kp: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward adopting the desired FINAL standing posture, gated to near-standing.

    Replaces `dof_error_when_standing`, which tracked `default_joint_pos` — but for
    the get-up task the model's default keyframe is the SUPINE pose, so the old
    reward actually pulled the standing posture back toward lying down. Here:

    - Target is an explicit standing pose (HOME), passed as a {joint_regex: value}
      dict and resolved once (cached) against the joint names.
    - Per-joint weights (also a {joint_regex: weight} dict) let us prioritise the
      joints that matter for a stable stance — legs / waist / ankles HIGH, arms
      MEDIUM, wrists LOW — instead of an equal 29-DOF L2 that over-constrains the
      arms / "hand" (wrist) joints.
    - SOFT, not rigid: exp(-kp · Σ w·(q-q*)²) ∈ (0, 1], a positive reward that
      peaks at the target pose, rather than an unbounded L2 penalty.
    - Gated by `_standing_gate` so it only acts once the robot is ~standing
      (>~0.6 m), i.e. it shapes the final posture and never fights the rise.

    target_joint_pos / joint_weights are robot-specific and set per-robot.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cache = getattr(env, _POSTURE_CACHE_ATTR, None)
    if cache is None:
        names = asset.joint_names
        tgt = torch.tensor(
            [resolve_expr(target_joint_pos or {}, names, 0.0)], device=env.device
        )
        w = torch.tensor(
            [resolve_expr(joint_weights or {}, names, 0.0)], device=env.device
        )
        cache = {"tgt": tgt, "w": w}
        setattr(env, _POSTURE_CACHE_ATTR, cache)
    err = asset.data.joint_pos - cache["tgt"]  # [B, J]
    metric = torch.sum(cache["w"] * err.pow(2), dim=1)  # [B], weighted L2
    pelvis_height = asset.data.root_link_pos_w[:, 2]  # [B]
    gate = _standing_gate(pelvis_height, pelvis_height_threshold, band=band)
    return torch.exp(-kp * metric) * gate  # positive reward in [0, 1] * gate


def feet_slip(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize horizontal sliding of the feet while they are actually loaded.

    Spec: "Foot slip / lateral force penalty". Now gated on the REAL contact state
    (`feet_ground_contact` found per foot) instead of foot height: a foot that is
    merely low but not in contact (e.g. skimming the floor, unloaded) no longer
    counts as slipping, which removes the false positives the height proxy gave.
    For each foot in contact, sum its horizontal speed. The sensor's per-foot
    columns align with site order (left, right). asset_cfg must specify the two
    foot site_names.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    asset: Entity = env.scene[asset_cfg.name]
    vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
    speed = torch.norm(vel_xy, dim=-1)  # [B, N]
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    in_contact = (found > 0).float()
    if in_contact.shape[1] != speed.shape[1]:
        # Defensive: collapse to a shared per-step contact flag if shapes differ.
        in_contact = in_contact.any(dim=1, keepdim=True)
    return torch.sum(speed * in_contact, dim=1)  # [B]


def feet_distance(
    env: ManagerBasedRlEnv,
    min_dist: float = 0.10,
    max_dist: float = 0.50,
    height_threshold: float = 0.40,
    band: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the horizontal foot-to-foot gap being outside [min_dist, max_dist].

    Spec: "Feet distance range: evita piedi troppo larghi". Yaw-invariant (uses the
    Euclidean gap between the two foot sites). asset_cfg must specify two foot sites.

    Now STAGE-GATED (`_standing_gate`): the foot-gap constraint only makes sense
    while rising/standing. On the floor (stage 0) the legs are sprawled by design,
    so the penalty is ramped to zero below the height threshold to avoid fighting
    the early get-up motion.
    """
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
    gap = torch.norm(pos[:, 0, :] - pos[:, 1, :], dim=-1)  # [B]
    too_wide = torch.clamp(gap - max_dist, min=0.0)
    too_narrow = torch.clamp(min_dist - gap, min=0.0)
    h = asset.data.root_link_pos_w[:, 2]
    gate = _standing_gate(h, height_threshold, band=band)
    return (too_wide + too_narrow) * gate  # [B], use with a negative weight


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

    `max_step` caps the per-step rise and NORMALISES the reward to [0, 1] so the
    signal is scale/timestep-independent (potential-based progress, à la Ng et
    al. 1999) and contributes meaningfully regardless of dt — instead of the old
    raw-metre value (~0.025 max) that was effectively negligible. Use a modest
    weight (the per-step reward is already O(1) when rising at the cap rate).
    """
    asset: Entity = env.scene[asset_cfg.name]
    state = get_episode_state(env, asset)
    h = asset.data.root_link_pos_w[:, 2]
    delta = torch.clamp(h - state["prev_height"], min=0.0, max=max_step) / max_step
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


def prone_recovery(
    env: ManagerBasedRlEnv,
    height_threshold: float = 0.40,
    band: float = 0.10,
    max_step: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward ACTIVELY rolling off the face (prone -> side/supine) [B,].

    Replaces the old `prone_supine_righting` for the prone case. The earlier
    reward gave a flat pose a constant ~1.0 just for lying there; here we use a
    PROGRESS signal (potential-based): pg_x = +1 prone, 0 on the side, -1 supine,
    so a DECREASE in pg_x means the belly is turning away from the ground.

    reward = clamp(prev_pg_x - pg_x, 0, max_step) / max_step  ∈ [0, 1]
    gated to low height (early stage). Parking in any pose pays ~0; only the act
    of rolling off the face is rewarded. Once supine (pg_x≈-1) it can no longer
    decrease, so this naturally fades and `supine_rising_prep` takes over.
    """
    asset: Entity = env.scene[asset_cfg.name]
    state = get_episode_state(env, asset)
    h = asset.data.root_link_pos_w[:, 2]
    pg_x = asset.data.projected_gravity_b[:, 0]
    delta = torch.clamp(state["prev_pg_x"] - pg_x, min=0.0, max=max_step) / max_step
    state["prev_pg_x"] = pg_x.clone()
    early_gate = 1.0 - _standing_gate(h, height_threshold, band=band)
    return delta * early_gate


def supine_rising_prep(
    env: ManagerBasedRlEnv,
    height_threshold: float = 0.40,
    band: float = 0.10,
    dist_scale: float = 3.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward a supine robot for adopting a configuration USEFUL for rising [B,].

    Replaces the old `prone_supine_righting` for the supine case. Lying flat on
    the back should NOT be rewarded per se; what we want is the robot to start
    sitting up and bring its mass over its feet. Two components (each in [0,1]):

      1. sit-up:  -proj_grav_z, i.e. the torso tilting from horizontal toward
                  vertical (0 flat -> 1 upright). Clamped to [0, 1].
      2. com-over-feet:  exp(-dist_scale · ||CoM_xy - feet_mid_xy||²), rewarding
                  the whole-body CoM moving over the feet midpoint.

    reward = supine_gate · low_gate · 0.5·(sit_up + com_term)

    Gates:
      - supine_gate = clamp(-pg_x, 0, 1): 1 supine, 0 on the side/prone (smooth,
        no discontinuity with prone_recovery which lives in the pg_x>0 hemisphere).
      - low_gate = 1 - standing_gate(h): only while low (hands off near standing).

    asset_cfg must provide foot site_names (left_foot, right_foot), set per-robot.
    """
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    pg = asset.data.projected_gravity_b  # [B,3]

    sit_up = torch.clamp(-pg[:, 2], 0.0, 1.0)  # 0 flat -> 1 upright

    root_id = asset.data.indexing.root_body_id
    com_xy = asset.data.data.subtree_com[:, root_id, :2]
    foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B,2,2]
    feet_mid_xy = foot_xy.mean(dim=1)
    dist = torch.norm(com_xy - feet_mid_xy, dim=-1)
    com_term = torch.exp(-dist_scale * dist.pow(2))

    supine_gate = torch.clamp(-pg[:, 0], 0.0, 1.0)
    low_gate = 1.0 - _standing_gate(h, height_threshold, band=band)
    return supine_gate * low_gate * 0.5 * (sit_up + com_term)


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
    target = getattr(term, "filtered_target_unclamped", term.applied_target)  # [B, A]
    asset: Entity = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits[:, term.target_ids]  # [B, A, 2]
    over = torch.clamp(target - limits[..., 1], min=0.0) + torch.clamp(
        limits[..., 0] - target, min=0.0
    )
    return torch.sum(torch.square(over), dim=1)  # [B]


def joint_power_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize mechanical power Σ|τ·q̇| across actuated joints [B,].

    Complements `joint_torques_l2`: torque norm alone does not capture how much
    the robot is actually SPENDING — a large torque at zero velocity (static hold)
    costs little power, while torque under motion does. Penalising power discourages
    high-effort dynamic motions and aligns with HoST's emphasis on motion economy /
    smoothness for sim2real. Same quantity as the `torque/power_mean` metric.
    """
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force  # [B, nu]
    ctrl_ids = asset.indexing.ctrl_ids
    qvel = asset.data.joint_vel[:, ctrl_ids]  # [B, nu]
    return torch.sum((tau * qvel).abs(), dim=-1)  # [B]


def _post_task_gate(
    env: ManagerBasedRlEnv,
    asset: Entity,
    height_threshold: float,
    band: float = 0.10,
) -> torch.Tensor:
    """[0,1] gate active only once the robot is standing AND upright.

    Used by the HoST-style post-task rewards: they should shape behaviour only
    AFTER the robot has gotten up (maintain a still, upright stance), never during
    the rise. gate = standing_gate(h) · clamp(-proj_grav_z, 0, 1).
    """
    h = asset.data.root_link_pos_w[:, 2]
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return _standing_gate(h, height_threshold, band=band) * upright


def base_ang_vel_penalty(
    env: ManagerBasedRlEnv,
    height_threshold: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task: penalize base angular velocity once standing [B,].

    HoST post-task term — a stood-up robot should be still, not spinning/rocking
    the torso. Gated so it never penalizes the angular motion needed to rise.
    """
    asset: Entity = env.scene[asset_cfg.name]
    w = asset.data.root_link_ang_vel_b  # [B, 3]
    return torch.sum(w.pow(2), dim=1) * _post_task_gate(env, asset, height_threshold)


def base_lin_vel_penalty(
    env: ManagerBasedRlEnv,
    height_threshold: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task: penalize horizontal base linear velocity once standing [B,].

    HoST post-task term — a stood-up robot should hold position, not drift/step.
    Horizontal only (vertical velocity near standing is small and not the concern).
    """
    asset: Entity = env.scene[asset_cfg.name]
    v_xy = asset.data.root_link_lin_vel_b[:, :2]  # [B, 2]
    return torch.sum(v_xy.pow(2), dim=1) * _post_task_gate(env, asset, height_threshold)


def standing_balance_hold(
    env: ManagerBasedRlEnv,
    target_height: float = 0.728,
    height_threshold: float = 0.65,
    k_height: float = 10.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task: reward HOLDING base orientation + height once standing [B,].

    HoST post-task term — a positive bonus that peaks when the robot is exactly
    upright at the target pelvis height: exp(-(k·(h-h*)² + (1+proj_grav_z))).
    Unlike base_height_exp (monotonic in h), this peaks AT the target and so
    discourages over/under-shoot and wobble. Gated by standing_gate so it only
    rewards a held standing pose, complementing the binary stable_success_hold.
    """
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    pg_z = asset.data.projected_gravity_b[:, 2]  # -1 upright
    err = k_height * (h - target_height).pow(2) + (1.0 + pg_z)
    return torch.exp(-err) * _standing_gate(h, height_threshold)


# ===========================================================================
# HoST-style STYLE + REGU shaping penalties (soft, phase-gated)
# ---------------------------------------------------------------------------
# All return a NON-negative quantity meant for a NEGATIVE weight. Unlike HoST's
# binary thresholds, the joint/orientation terms are SOFT (quadratic past a band)
# and gated to the relevant get-up phase so they protect the motion during the
# rise without fighting stage-0 discovery. Thresholds/weights are starting points.
# ===========================================================================


def _stage_gate(
    h: torch.Tensor,
    lo: float | None = None,
    hi: float | None = None,
    band: float = 0.10,
) -> torch.Tensor:
    """Composable [0,1] phase gate on pelvis height h.

    lo set -> ramps ON above lo; hi set -> ramps OFF above hi. Both -> a window
    (lo, hi). Neither -> constant 1. Built from the existing `_standing_gate`.
    """
    gate = torch.ones_like(h)
    if lo is not None:
        gate = gate * _standing_gate(h, lo, band=band)
    if hi is not None:
        gate = gate * (1.0 - _standing_gate(h, hi, band=band))
    return gate


def _cached_body_ids(
    env: "ManagerBasedRlEnv", asset: Entity, names: tuple[str, ...], attr: str
) -> torch.Tensor:
    """Resolve a tuple of EXACT body names to ids, IN ORDER, cached on env.

    Resolving each name individually (not via SceneEntityCfg) guarantees the
    left/right ordering needed to pair knee↔foot etc., independent of body index.
    """
    cache = getattr(env, attr, None)
    if cache is None:
        ids = [int(asset.find_bodies(n)[0][0]) for n in names]
        cache = torch.as_tensor(ids, device=env.device, dtype=torch.long)
        setattr(env, attr, cache)
    return cache


# --- REGU (group A) --------------------------------------------------------

def joint_tracking_error(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the PD actuation gap Σ(commanded_target − measured_q)² [B,].

    HoST regu `joint_tracking_error`. Reads the (filtered) commanded target like
    `action_saturation` does. Discourages commanding targets the PD cannot reach
    (deployability / torque economy). Use with a tiny negative weight.
    """
    term = env.action_manager.get_term(action_name)
    target = getattr(term, "filtered_target_unclamped", term.applied_target)  # [B, A]
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, term.target_ids]  # [B, A]
    return torch.sum((target - q).pow(2), dim=1)


_VEL_LIMIT_CACHE = "_getup_vel_limit_cache"


def dof_vel_limits(
    env: ManagerBasedRlEnv,
    vel_limits: dict[str, float] | None = None,
    soft_ratio: float = 0.9,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Graded penalty as joint speeds approach their limit (HoST dof_vel_limits) [B,].

    Σ clamp(|q̇| − soft_ratio·limit, 0, 1). Soft companion to the hard
    joint_vel_explosion termination: bites smoothly BEFORE the blow-up. Per-joint
    limits come from a {joint_regex: limit} dict (G1 actuator velocity limits),
    resolved once and cached. Unmatched joints get a huge limit (never penalized).
    """
    asset: Entity = env.scene[asset_cfg.name]
    cache = getattr(env, _VEL_LIMIT_CACHE, None)
    if cache is None:
        lim = torch.tensor(
            [resolve_expr(vel_limits or {}, asset.joint_names, 1e6)], device=env.device
        )
        cache = lim * soft_ratio  # [1, J]
        setattr(env, _VEL_LIMIT_CACHE, cache)
    over = torch.clamp(asset.data.joint_vel.abs() - cache, min=0.0, max=1.0)
    return torch.sum(over, dim=1)


# --- Per-joint deviation (group B) -----------------------------------------

def joint_group_deviation(
    env: ManagerBasedRlEnv,
    lower: float,
    upper: float,
    gate_lo: float | None = None,
    gate_hi: float | None = None,
    band: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Soft penalty for a joint group leaving the allowed band [lower, upper] [B,].

    `over = clamp(q−upper,0)+clamp(lower−q,0)` per joint, Σ over² · phase_gate.
    Soft (quadratic) version of HoST's binary deviation penalties; the joint group
    is selected per-robot via asset_cfg.joint_ids, the active phase via gate_lo/hi.
    """
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    over = torch.clamp(q - upper, min=0.0) + torch.clamp(lower - q, min=0.0)
    pen = torch.sum(over.pow(2), dim=1)
    gate = _stage_gate(asset.data.root_link_pos_w[:, 2], lo=gate_lo, hi=gate_hi, band=band)
    return pen * gate


# --- Leg / foot orientation (group C) --------------------------------------

def shank_orientation(
    env: ManagerBasedRlEnv,
    knee_names: tuple[str, ...] = ("left_knee_link", "right_knee_link"),
    foot_names: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    min_cos: float = 0.8,
    gate_lo: float = 0.40,
    band: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize shanks not vertical enough during the rise (HoST shank_orientation) [B,].

    cos = (z_knee − z_foot)/‖knee − foot‖ ∈ [-1,1] (1 = shank vertical). Penalty
    Σ clamp(min_cos − cos, 0)² over both legs. A vertical shank builds a stable
    base under the pelvis in the kneel/crouch-to-stand phase. Gated above gate_lo.
    """
    asset: Entity = env.scene[asset_cfg.name]
    knee_ids = _cached_body_ids(env, asset, knee_names, "_shank_knee_ids")
    foot_ids = _cached_body_ids(env, asset, foot_names, "_shank_foot_ids")
    vec = (
        asset.data.body_link_pos_w[:, knee_ids, :]
        - asset.data.body_link_pos_w[:, foot_ids, :]
    )  # [B, 2, 3]
    cos = vec[..., 2] / (torch.norm(vec, dim=-1) + 1e-6)  # [B, 2]
    pen = torch.clamp(min_cos - cos, min=0.0).pow(2).sum(dim=1)
    gate = _stage_gate(asset.data.root_link_pos_w[:, 2], lo=gate_lo, band=band)
    return pen * gate


def thigh_orientation(
    env: ManagerBasedRlEnv,
    thigh_names: tuple[str, ...] = ("left_hip_yaw_link", "right_hip_yaw_link"),
    knee_names: tuple[str, ...] = ("left_knee_link", "right_knee_link"),
    min_cos: float = 0.8,
    gate_lo: float = 0.30,
    gate_hi: float = 0.60,
    band: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize thighs not vertical enough during the crouch transition [B,].

    cos = (z_thigh − z_knee)/‖thigh − knee‖. Uses the WORST (min) leg, penalty
    clamp(min_cos − worst, 0)². Helps stack the legs in the crouch/kneel-to-stand
    transition. WINDOW-gated (gate_lo, gate_hi): off while fully prone (thighs are
    legitimately horizontal there) and off at standing — active only mid-rise.
    """
    asset: Entity = env.scene[asset_cfg.name]
    thigh_ids = _cached_body_ids(env, asset, thigh_names, "_thigh_ids")
    knee_ids = _cached_body_ids(env, asset, knee_names, "_thigh_knee_ids")
    vec = (
        asset.data.body_link_pos_w[:, thigh_ids, :]
        - asset.data.body_link_pos_w[:, knee_ids, :]
    )  # [B, 2, 3]
    cos = vec[..., 2] / (torch.norm(vec, dim=-1) + 1e-6)  # [B, 2]
    worst = cos.min(dim=1).values
    pen = torch.clamp(min_cos - worst, min=0.0).pow(2)
    gate = _stage_gate(asset.data.root_link_pos_w[:, 2], lo=gate_lo, hi=gate_hi, band=band)
    return pen * gate


def feet_flat(
    env: ManagerBasedRlEnv,
    foot_names: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    gate_lo: float = 0.40,
    band: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize feet not flat on the ground (HoST feet_ori) [B,].

    Projects gravity into each foot frame; a flat foot has its up-axis vertical,
    i.e. zero horizontal gravity component. Penalty Σ_feet ‖proj_grav_xy‖². Crucial
    for a stable real stance. Gated above gate_lo, fully on by standing.
    """
    asset: Entity = env.scene[asset_cfg.name]
    ids = _cached_body_ids(env, asset, foot_names, "_feet_flat_ids")
    quat = asset.data.body_link_quat_w[:, ids, :]  # [B, F, 4]
    g = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, quat.shape[1], -1)  # [B, F, 3]
    pg = quat_apply_inverse(quat, g)  # [B, F, 3]
    pen = torch.sum(pg[..., :2].pow(2), dim=(1, 2))
    gate = _stage_gate(asset.data.root_link_pos_w[:, 2], lo=gate_lo, band=band)
    return pen * gate


def feet_level(
    env: ManagerBasedRlEnv,
    foot_names: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    gate_lo: float = 0.40,
    band: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the two feet being at different heights (HoST ground_parallel) [B,].

    (z_footL − z_footR)². Keeps the support symmetric / parallel to the ground.
    Gated above gate_lo (rising/standing).
    """
    asset: Entity = env.scene[asset_cfg.name]
    ids = _cached_body_ids(env, asset, foot_names, "_feet_level_ids")
    z = asset.data.body_link_pos_w[:, ids, 2]  # [B, 2]
    pen = (z[:, 0] - z[:, 1]).pow(2)
    gate = _stage_gate(asset.data.root_link_pos_w[:, 2], lo=gate_lo, band=band)
    return pen * gate


# --- Foot placement + base rotation (group D) ------------------------------

def foot_displacement(
    env: ManagerBasedRlEnv,
    max_dist: float = 0.35,
    gate_lo: float = 0.40,
    band: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize feet migrating too far from the base in XY (HoST foot_displacement) [B,].

    Σ_feet clamp(‖base_xy − foot_xy‖ − max_dist, 0)². DIFFERENT from feet_slip
    (which penalizes sliding VELOCITY): this penalizes the feet drifting away from
    under the body. asset_cfg must provide the two foot site_names. Gated above gate_lo.
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_xy = asset.data.root_link_pos_w[:, :2].unsqueeze(1)  # [B, 1, 2]
    foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
    dist = torch.norm(foot_xy - base_xy, dim=-1)  # [B, F]
    over = torch.clamp(dist - max_dist, min=0.0)
    pen = over.pow(2).sum(dim=1)
    gate = _stage_gate(asset.data.root_link_pos_w[:, 2], lo=gate_lo, band=band)
    return pen * gate


def style_ang_vel_xy(
    env: ManagerBasedRlEnv,
    gate_lo: float = 0.20,
    gate_hi: float = 0.65,
    band: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive torso roll/pitch angular velocity DURING the rise [B,].

    Σ ω_base[xy]², gated to the stage 1-2 window (0.20-0.65). Complements
    base_ang_vel_penalty (which is post-task, >0.65): this damps torso tumbling
    while the robot is still getting up.
    """
    asset: Entity = env.scene[asset_cfg.name]
    w_xy = asset.data.root_link_ang_vel_b[:, :2]
    pen = torch.sum(w_xy.pow(2), dim=1)
    gate = _stage_gate(asset.data.root_link_pos_w[:, 2], lo=gate_lo, hi=gate_hi, band=band)
    return pen * gate


# ===========================================================================
# HoST definitive reward set (task + style). Three-stage activation on base
# height: Stage 1 (righting) h < H_STAGE1; Stage 2 (rising) H_STAGE1 <= h <
# H_STAGE2; Stage 3 (standing) h >= H_STAGE2. Reward terms are gated by HARD
# stage indicators 1(h > H_STAGE*) exactly as in the HoST formulas.
# ===========================================================================

H_STAGE1 = 0.45  # righting -> rising boundary [m]
H_STAGE2 = 0.65  # rising -> standing boundary [m]


def f_tol(
    x: torch.Tensor,
    lower: float,
    upper: float,
    margin: float,
    value_at_margin: float,
) -> torch.Tensor:
    """dm_control-style Gaussian tolerance reward.

    Returns 1.0 when ``lower <= x <= upper``; outside the bounds it decays with a
    Gaussian, reaching ``value_at_margin`` at distance ``margin`` from the nearest
    bound. ``upper`` may be ``float('inf')`` (one-sided bound).
    """
    below = torch.clamp(lower - x, min=0.0)
    above = torch.clamp(x - upper, min=0.0) if math.isfinite(upper) else torch.zeros_like(x)
    d = (below + above) / margin
    scale = math.sqrt(-2.0 * math.log(value_at_margin))
    val = torch.exp(-0.5 * (d * scale).pow(2))
    in_bounds = (x >= lower) & (x <= upper)
    return torch.where(in_bounds, torch.ones_like(x), val)


def _cached_joint_ids(
    env: "ManagerBasedRlEnv", asset: Entity, names: tuple[str, ...], attr: str
) -> torch.Tensor:
    """Resolve EXACT joint names to ids, IN ORDER, cached on env (mirror of bodies)."""
    cache = getattr(env, attr, None)
    if cache is None:
        ids = [int(asset.find_joints(n)[0][0]) for n in names]
        cache = torch.as_tensor(ids, device=env.device, dtype=torch.long)
        setattr(env, attr, cache)
    return cache


# --- Task group (HoST) -----------------------------------------------------

def task_head_height(
    env: ManagerBasedRlEnv,
    lower: float = 1.0,
    margin: float = 1.0,
    value_at_margin: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST head-height task reward f_tol(h_head, [1, inf), margin=1, value=0.1) [B,].

    NOTE: G1 exposes no separate head body, so h_head uses the tracked torso_link
    height as a proxy (asset_cfg.body_ids). When standing the torso sits ~1.1 m, so
    it clears the 1.0 lower bound (full reward); it decays as the robot lies down.
    """
    asset: Entity = env.scene[asset_cfg.name]
    h_head = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)
    return f_tol(h_head, lower, float("inf"), margin, value_at_margin)


def task_base_orientation(
    env: ManagerBasedRlEnv,
    lower: float = 0.99,
    margin: float = 1.0,
    value_at_margin: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST base-orientation task reward f_tol(-g_z, [0.99, inf), margin=1, value=0.05) [B,].

    -projected_gravity_b[z] ~ +1 when the base is upright (gravity along -z_base).
    """
    asset: Entity = env.scene[asset_cfg.name]
    upright = -asset.data.projected_gravity_b[:, 2]
    return f_tol(upright, lower, float("inf"), margin, value_at_margin)


# --- Style group (HoST) ----------------------------------------------------

def style_waist_yaw_deviation(
    env: ManagerBasedRlEnv,
    limit: float = 1.4,
    penalty: float = -10.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Binary: penalty if |q_waist_yaw| > limit (HoST waist deviation) [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [B, n]
    viol = (q.abs() > limit).any(dim=1).float()
    return penalty * viol


def style_hip_deviation(
    env: ManagerBasedRlEnv,
    roll_limit: float = 1.4,
    yaw_limit: float = 0.9,
    penalty: float = -10.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST hip deviation: -10 if max(|hip_roll|)>1.4, -10 if min(|hip_yaw|)>0.9 [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    roll_ids = _cached_joint_ids(
        env, asset, ("left_hip_roll_joint", "right_hip_roll_joint"), "_hip_roll_ids"
    )
    yaw_ids = _cached_joint_ids(
        env, asset, ("left_hip_yaw_joint", "right_hip_yaw_joint"), "_hip_yaw_ids"
    )
    q_roll = asset.data.joint_pos[:, roll_ids].abs()  # [B, 2]
    q_yaw = asset.data.joint_pos[:, yaw_ids].abs()  # [B, 2]
    roll_viol = (q_roll.max(dim=1).values > roll_limit).float()
    yaw_viol = (q_yaw.min(dim=1).values > yaw_limit).float()
    return penalty * roll_viol + penalty * yaw_viol


def style_knee_deviation(
    env: ManagerBasedRlEnv,
    hi_limit: float = 2.85,
    lo_limit: float = -0.06,
    penalty: float = -0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST knee deviation (Ground value): -0.25 if max(q_knee)>2.85, -0.25 if min<-0.06 [B,].

    The HoST penalty is terrain-dependent (-0.25 on flat ground, -10 on
    platform/slope/wall). This env is flat-ground only, so the Ground value is used.
    """
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [B, 2]
    hi_viol = (q.max(dim=1).values > hi_limit).float()
    lo_viol = (q.min(dim=1).values < lo_limit).float()
    return penalty * hi_viol + penalty * lo_viol


def style_shoulder_roll_deviation(
    env: ManagerBasedRlEnv,
    left_limit: float = -0.02,
    right_limit: float = 0.02,
    penalty: float = -2.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST shoulder roll: -2.5 if q_left < -0.02, -2.5 if q_right > 0.02 [B,].

    Penalizes arms rolling inward during the stand-up. Left/right resolved by exact
    name so the sign convention is guaranteed.
    """
    asset: Entity = env.scene[asset_cfg.name]
    ids = _cached_joint_ids(
        env, asset, ("left_shoulder_roll_joint", "right_shoulder_roll_joint"), "_shoulder_roll_ids"
    )
    q = asset.data.joint_pos[:, ids]  # [B, 2] -> [:, 0]=left, [:, 1]=right
    left_viol = (q[:, 0] < left_limit).float()
    right_viol = (q[:, 1] > right_limit).float()
    return penalty * left_viol + penalty * right_viol


def style_foot_displacement(
    env: ManagerBasedRlEnv,
    scale: float = 2.0,
    clip_min: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST CoM-in-support: Σ_feet exp(-2·clamp(||base_xy-foot_xy||², 0.3)) · 1(h>H_STAGE2) [B,].

    Summed over both feet; configure weight 2.5 (per-foot). The clamp(.,0.3) floors
    the squared distance, so the reward saturates (~0.55/foot) when the base is well
    centered and only drops once a foot is displaced beyond ~0.3 m. Only active after
    Stage 2. asset_cfg.site_names = (left_foot, right_foot).
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_xy = asset.data.root_link_pos_w[:, :2].unsqueeze(1)  # [B, 1, 2]
    foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
    d2 = ((foot_xy - base_xy).pow(2)).sum(dim=-1)  # [B, F]
    d2 = torch.clamp(d2, min=clip_min)
    r = torch.exp(-scale * d2).sum(dim=1)  # [B]
    gate = (asset.data.root_link_pos_w[:, 2] > H_STAGE2).float()
    return r * gate


def style_foot_distance(
    env: ManagerBasedRlEnv,
    max_dist_sq: float = 0.9,
    penalty: float = -10.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST foot distance: -10 if ||foot_L - foot_R||² > 0.9 (feet too far apart) [B,].

    Horizontal (xy) separation between the two foot sites. asset_cfg.site_names =
    (left_foot, right_foot).
    """
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
    d2 = (pos[:, 0] - pos[:, 1]).pow(2).sum(dim=-1)  # [B]
    return penalty * (d2 > max_dist_sq).float()


def style_shank_orientation(
    env: ManagerBasedRlEnv,
    knee_names: tuple[str, ...] = ("left_knee_link", "right_knee_link"),
    foot_names: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    lower: float = 0.8,
    margin: float = 1.0,
    value_at_margin: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST shank orientation: f_tol(mean(shank_z), [0.8, inf), m=1, v=0.1) · 1(h>H_STAGE1) [B,].

    shank_z = (z_knee - z_foot)/||knee - foot|| per leg (1 = vertical shank), averaged
    over both legs. Encourages upright lower legs during the rise. Active after Stage 1.
    """
    asset: Entity = env.scene[asset_cfg.name]
    knee_ids = _cached_body_ids(env, asset, knee_names, "_shank_tol_knee_ids")
    foot_ids = _cached_body_ids(env, asset, foot_names, "_shank_tol_foot_ids")
    vec = (
        asset.data.body_link_pos_w[:, knee_ids, :]
        - asset.data.body_link_pos_w[:, foot_ids, :]
    )  # [B, 2, 3]
    cos = vec[..., 2] / (torch.norm(vec, dim=-1) + 1e-6)  # [B, 2]
    mean_cos = cos.mean(dim=1)  # [B]
    r = f_tol(mean_cos, lower, float("inf"), margin, value_at_margin)
    gate = _standing_gate(asset.data.root_link_pos_w[:, 2], H_STAGE1, band=0.08)
    return r * gate


def style_base_ang_vel(
    env: ManagerBasedRlEnv,
    scale: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST style base ang-vel: exp(-2·||w_xy||²) · 1(h>H_STAGE1) [B,].

    Positive reward for low horizontal-plane base angular velocity during the rise
    (stabilizes the trunk). A separate, stronger version lives in the post-task group.
    """
    asset: Entity = env.scene[asset_cfg.name]
    w_xy = asset.data.root_link_ang_vel_b[:, :2]
    r = torch.exp(-scale * w_xy.pow(2).sum(dim=1))
    gate = _standing_gate(asset.data.root_link_pos_w[:, 2], H_STAGE1, band=0.04)
    return r * gate


def style_ankle_parallel(
    env: ManagerBasedRlEnv,
    tilt_threshold: float = 0.05,
    reward: float = 20.0,
    foot_names: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST ankle parallel: Gaussian reward peaking at +reward when both feet are flat [B,].

    Gaussian reward (peak default 20) that decays smoothly as feet tilt away from flat.
    Provides dense gradient signal toward flat feet throughout training, not just a binary
    cliff at the threshold.

    HoST measures flatness as the variance of per-foot keypoint heights. The G1's sole
    collision geoms are a thin LATERAL line (no fore-aft extent), so their z-variance is
    blind to foot pitch and unusable as a proxy (measured ~0.0002 in every pose). Instead
    we use the kinematically robust, ISOTROPIC equivalent: project gravity into each foot
    frame and take ‖proj_grav_xy‖² (the squared sine of the foot's tilt from horizontal —
    same quantity as `feet_flat`). It is ~0 when a foot rests flat (foot up-axis vertical,
    the standing target) and grows with pitch OR roll. `tilt_threshold` sets the Gaussian
    width: exp(-tilt / (2*thr)) → at thr the reward is ~61% of peak; lying poses (tilt≥0.76)
    → ~0.
    """
    asset: Entity = env.scene[asset_cfg.name]
    ids = _cached_body_ids(env, asset, foot_names, "_ankle_parallel_ids")
    quat = asset.data.body_link_quat_w[:, ids, :]  # [B, 2, 4]
    g = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, quat.shape[1], -1)  # [B, 2, 3]
    pg = quat_apply_inverse(quat, g)  # [B, 2, 3]
    tilt = pg[..., :2].pow(2).sum(dim=-1).mean(dim=1)  # [B]; 0 flat -> grows with tilt
    return reward * torch.exp(-tilt / (2 * tilt_threshold))


def style_feet_stumble(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    ratio: float = 3.0,
    penalty: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST feet stumble: 1(∃i, |F_xy_i| > ratio·|F_z_i|) · penalty [B,].

    Penalizes a foot whose horizontal contact force exceeds `ratio`× its vertical
    force — i.e. the foot scraping / stumbling against a surface edge rather than
    pressing down. HoST gives this 0 on flat Ground and −25 on platform/slope/wall
    (PSW). This env is flat-ground only, so `penalty` defaults to 0 (a no-op scaffold);
    raise its magnitude when the PSW terrains are added in the Sim2Real phase.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    force = sensor.data.force
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    if force.ndim == 2:  # [B, 3] -> [B, 1, 3]
        force = force.unsqueeze(1)
    f_xy = torch.norm(force[..., :2], dim=-1)  # [B, N]
    f_z = force[..., 2].abs()  # [B, N]
    viol = (f_xy > ratio * f_z).any(dim=1).float()  # [B]
    return penalty * viol


# --- Post-task group (HoST) ------------------------------------------------
# Shape the held standing state. Every term is gated by the HARD indicator
# 1(h_base > H_STAGE2) and is zero until the robot is actually standing.

_POST_UPPER_CACHE_ATTR = "_post_upper_body_cache"
_HOME_L2_CACHE_ATTR = "_home_l2_cache"


def _post_gate(asset: Entity, height_threshold: float = H_STAGE2, band: float = 0.06) -> torch.Tensor:
    """Smooth post-task gate: linear ramp from 0→1 over [threshold-band, threshold+band] [B,].

    Replaces the previous hard step to give the policy gradient signal as it approaches
    standing height. With default band=0.06: ramp spans [0.59m, 0.71m]; all lying poses
    (h<0.37m) still receive ~0; the cliff at H_STAGE2 is removed.
    """
    h = asset.data.root_link_pos_w[:, 2]
    return _standing_gate(h, height_threshold, band=band)


def home_pose_l2(
    env: ManagerBasedRlEnv,
    target_joint_pos: dict[str, float],
    joint_weights: dict[str, float],
    height_threshold: float = H_STAGE2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L2 penalty toward HOME pose, gated to standing [B,].

    Returns Σ w·(q − q_HOME)², a NON-NEGATIVE value — use with a NEGATIVE weight.

    Unlike standing_posture() which uses exp(−kp·err) and has a near-zero gradient
    when the robot is far from HOME (the signal dies exactly when needed most), this
    function returns the raw weighted squared error. The gradient is proportional to
    (q − q_HOME) at ALL distances — it never saturates or dies. This makes it the
    right choice when fine-tuning a policy that already has a wrong posture: the pull
    toward HOME is constant regardless of how far off the current stance is.

    target_joint_pos and joint_weights are {regex: value} dicts resolved once (cached)
    against the robot's joint names. Joints absent from joint_weights get weight 0
    and are not penalised. Set per-robot in env_cfgs.py.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cache = getattr(env, _HOME_L2_CACHE_ATTR, None)
    if cache is None:
        names = asset.joint_names
        tgt = torch.tensor(
            [resolve_expr(target_joint_pos or {}, names, 0.0)], device=env.device
        )
        w = torch.tensor(
            [resolve_expr(joint_weights or {}, names, 0.0)], device=env.device
        )
        cache = {"tgt": tgt, "w": w}
        setattr(env, _HOME_L2_CACHE_ATTR, cache)
    err = asset.data.joint_pos - cache["tgt"]          # [B, J]
    metric = torch.sum(cache["w"] * err.pow(2), dim=1) # [B]
    return metric * _post_gate(asset, height_threshold)


def post_base_ang_vel(
    env: ManagerBasedRlEnv,
    scale: float = 2.0,
    height_threshold: float = H_STAGE2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task base ang-vel: exp(-2·||w_xy||²) · 1(h>H_STAGE2). Still trunk [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    w_xy = asset.data.root_link_ang_vel_b[:, :2]
    return torch.exp(-scale * w_xy.pow(2).sum(dim=1)) * _post_gate(asset, height_threshold)


def post_base_lin_vel(
    env: ManagerBasedRlEnv,
    scale: float = 5.0,
    height_threshold: float = H_STAGE2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task base lin-vel: exp(-5·||v_xy||²) · 1(h>H_STAGE2). Hold position [B,]."""
    asset: Entity = env.scene[asset_cfg.name]
    v_xy = asset.data.root_link_lin_vel_b[:, :2]
    return torch.exp(-scale * v_xy.pow(2).sum(dim=1)) * _post_gate(asset, height_threshold)


def post_base_orientation(
    env: ManagerBasedRlEnv,
    scale: float = 5.0,
    height_threshold: float = H_STAGE2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task base orientation: exp(-5·||theta_xy||²) · 1(h>H_STAGE2) [B,].

    theta_xy = horizontal projected-gravity components (base tilt); 0 when upright.
    """
    asset: Entity = env.scene[asset_cfg.name]
    theta_xy = asset.data.projected_gravity_b[:, :2]
    return torch.exp(-scale * theta_xy.pow(2).sum(dim=1)) * _post_gate(asset, height_threshold)


def post_base_height(
    env: ManagerBasedRlEnv,
    target_height: float = 0.7,
    scale: float = 20.0,
    height_threshold: float = H_STAGE2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task base height: exp(-20·(h-h*)²) · 1(h>H_STAGE2), h*=0.7 m [B,].

    Tight Gaussian: ~0.6 already at 5 cm deviation.
    """
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    return torch.exp(-scale * (h - target_height).pow(2)) * _post_gate(asset, height_threshold)


def post_upper_body_posture(
    env: ManagerBasedRlEnv,
    target_joint_pos: dict[str, float],
    joint_weights: dict[str, float],
    scale: float = 0.1,
    height_threshold: float = H_STAGE2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task upper-body posture: exp(-0.1·Σ w·(q-q*)²) · 1(h>H_STAGE2) [B,].

    Softly encourages the default arm/waist standing pose. ``joint_weights`` is a 0/1
    mask selecting the upper-body joints; ``target_joint_pos`` is the HOME pose. Both
    are {regex: value} dicts resolved once (cached), set per-robot.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cache = getattr(env, _POST_UPPER_CACHE_ATTR, None)
    if cache is None:
        names = asset.joint_names
        tgt = torch.tensor([resolve_expr(target_joint_pos or {}, names, 0.0)], device=env.device)
        w = torch.tensor([resolve_expr(joint_weights or {}, names, 0.0)], device=env.device)
        cache = {"tgt": tgt, "w": w}
        setattr(env, _POST_UPPER_CACHE_ATTR, cache)
    err = asset.data.joint_pos - cache["tgt"]  # [B, J]
    metric = torch.sum(cache["w"] * err.pow(2), dim=1)  # [B]
    return torch.exp(-scale * metric) * _post_gate(asset, height_threshold)


def post_feet_parallel(
    env: ManagerBasedRlEnv,
    scale: float = 20.0,
    clip_min: float = 0.02,
    height_threshold: float = H_STAGE2,
    foot_names: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task feet parallel: exp(-20·|z_L - z_R|.clip(0.02, inf)) · 1(h>H_STAGE2) [B,].

    Feet at equal height. The clip(0.02) floors the absolute height difference, so
    differences below 2 cm incur no marginal penalty (only real asymmetry is shaped).
    """
    asset: Entity = env.scene[asset_cfg.name]
    ids = _cached_body_ids(env, asset, foot_names, "_post_feet_parallel_ids")
    z = asset.data.body_link_pos_w[:, ids, 2]  # [B, 2]
    diff = torch.clamp((z[:, 0] - z[:, 1]).abs(), min=clip_min)
    return torch.exp(-scale * diff) * _post_gate(asset, height_threshold)


def post_feet_yaw(
    env: ManagerBasedRlEnv,
    scale: float = 10.0,
    height_threshold: float = H_STAGE2,
    joint_names: tuple[str, ...] = ("left_hip_yaw_joint", "right_hip_yaw_joint"),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task feet yaw alignment: exp(-scale·(q_yawL² + q_yawR²))·1(h>H_STAGE2) [B,].

    Both feet point straight forward when hip_yaw=0. Decays when the robot toes
    out or in after standing. Peak reward = 1.0 when both hip_yaw = 0.
    """
    asset: Entity = env.scene[asset_cfg.name]
    ids = _cached_joint_ids(env, asset, joint_names, "_post_feet_yaw_joint_ids")
    q = asset.data.joint_pos[:, ids]  # [B, 2]
    return torch.exp(-scale * q.pow(2).sum(dim=1)) * _post_gate(asset, height_threshold)


_JOINT_VEL_LIMIT_CACHE = "_joint_vel_limit_cache"


def joint_vel_limits(
    env: ManagerBasedRlEnv,
    vel_limits: dict[str, float] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST joint velocity limits: Σ_i clamp(|q̇_i| - limit_i, 0) [B,].

    Hard, uncapped over-limit penalty (no soft ratio). Per-joint limits come from a
    {joint_regex: limit} dict, resolved once and cached; unmatched joints get a huge
    limit (never penalized). Returns a NON-NEGATIVE violation amount, so use a
    negative weight (the HoST term is -Σ[...] with magnitude 1 -> weight -1.0).
    """
    asset: Entity = env.scene[asset_cfg.name]
    cache = getattr(env, _JOINT_VEL_LIMIT_CACHE, None)
    if cache is None:
        cache = torch.tensor(
            [resolve_expr(vel_limits or {}, asset.joint_names, 1e6)], device=env.device
        )  # [1, J]
        setattr(env, _JOINT_VEL_LIMIT_CACHE, cache)
    over = torch.clamp(asset.data.joint_vel.abs() - cache, min=0.0)
    return torch.sum(over, dim=1)
