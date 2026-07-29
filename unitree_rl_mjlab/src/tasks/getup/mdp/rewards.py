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
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
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
    height_threshold: float = 0.50,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward when both feet are in ground contact and not elevated.

    Equivalent to HumanUP _reward_stand_on_feet.
    asset_cfg must specify site_names for left and right foot sites.

    `height_threshold` gates the reward via _standing_gate: the reward only
    starts fading in once the pelvis reaches ~(height_threshold - band/2) m and
    is fully active above height_threshold. This prevents the exploit where the
    policy learns to lie on the ground with feet touching the floor.
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

    h = asset.data.root_link_pos_w[:, 2]
    gate = _standing_gate(h, height_threshold, band=0.08)
    return (0.5 * any_foot + 0.5 * both_feet) * gate


def pelvis_height_bridge(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    lower: float = 0.35,
    upper: float = 0.75,
    foot_height_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Dense feet-planted pelvis-height ramp that BRIDGES the crouch->stand gap [B,].

    Motivation. The get-up reward stack had a WALL at pelvis h≈0.65: `height_progress`
    is an upward-*delta* reward (zero the instant the robot stops rising), and every
    "standing" reward (post_base_*, stable_success_hold) is gated at h>0.65 — so a
    robot that reaches an upright-torso CROUCH (~0.4-0.6 m) sits in a dead zone with no
    gradient pulling the pelvis the last stretch up. Training parked there (stage2≈60%,
    stage3≈0.3%). This term fills that gap with a CONTINUOUS position reward.

        reward = height_ramp · upright · feet_planted   ∈ [0, 1]

      - height_ramp = clamp((h - lower)/(upper - lower), 0, 1): 0 at `lower` (deep
        crouch), 1 at `upper` (standing). A monotone climbing gradient across the whole
        band — unlike the delta reward it keeps paying more for every extra cm, so
        extending the legs out of the crouch is always uphill.
      - upright = clamp(-proj_grav_b_z, 0, 1): 1 when the pelvis is vertical, 0 when
        horizontal. Kills the glute-BRIDGE exploit (hips up, back on floor → pelvis
        horizontal → 0) so a high pelvis only pays when the trunk is actually upright.
      - feet_planted = mean over feet of (in_contact AND low): 0/0.5/1. Kills the
        BALLISTIC-JUMP exploit — you cannot farm height by launching, because a jump
        lifts the feet off the floor and zeroes the term. Same planted-foot convention
        as `stand_on_feet`.

    The ONLY way to maximise all three factors at once is to stand tall with both feet
    planted and the trunk upright — i.e. the actual task — so this is a dense signal
    that is not farmable by any of the sub-standing local optima we have seen. Requires
    `asset_cfg.site_names` = the two foot sites (set per-robot). Use a POSITIVE weight.
    """
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    height_ramp = torch.clamp((h - lower) / (upper - lower + 1e-6), 0.0, 1.0)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)

    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    in_contact = contact_sensor.data.found > 0  # [B, N_feet]
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N_feet]
    planted = (in_contact & (foot_heights < foot_height_threshold)).float().mean(dim=1)  # [B]

    return height_ramp * upright * planted


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


def hand_contact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    free_force: float = 8.0,
    force_scale: float = 10.0,
    ramp_steps: int = 0,
    latch_height: float = 0.72,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Exponential penalty for hand-to-terrain contact force [B,], LATCHED off at HOME.

    Dead zone: force_norm <= free_force (hand's own weight ~7.65 N) → 0 penalty.
    Above that: exp(excess / force_scale) - 1, growing exponentially.
    Excess is clamped at 5×force_scale to prevent numerical overflow while
    preserving the exponential shape in the relevant range (default: 8–58 N).
    Use with a small negative weight (e.g. -0.1); at the 5× clamp the raw
    output is exp(5)-1 ≈ 147, giving a max step penalty of ~14.7 at weight=-0.1.

    Activation policy (per user): the penalty is ACTIVE during the whole push-off /
    rise (so the policy is discouraged from leaning on its hands, as in the previous
    training), and LATCHES OFF for the rest of the episode the moment the pelvis first
    reaches `latch_height` (~0.72 m, HOME). Once the stand is achieved a subsequent
    fall does NOT re-arm the penalty, so a failed episode is not buried under a huge
    hand-contact penalty. The latch (`ever_reached_home`) lives in the shared episode
    state and is cleared on episode reset.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _, force_norm = _contact_force_norm(env, sensor_name)
    excess = torch.clamp(force_norm - free_force, min=0.0, max=5.0 * force_scale)
    penalty = (torch.exp(excess / force_scale) - 1.0) * _reset_ramp(env, ramp_steps)
    # Latch: once the pelvis reaches HOME height this episode, disable the penalty.
    state = get_episode_state(env, asset)
    pelvis_h = asset.data.root_link_pos_w[:, 2]
    state["ever_reached_home"] = state["ever_reached_home"] | (pelvis_h >= latch_height)
    active = (~state["ever_reached_home"]).float()
    return penalty * active


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
    joint_bands: dict[str, float] | None = None,
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
        band_t = torch.tensor(
            [resolve_expr(joint_bands or {}, names, 0.0)], device=env.device
        )
        cache = {"tgt": tgt, "w": w, "band": band_t}
        setattr(env, _POSTURE_CACHE_ATTR, cache)
    err = asset.data.joint_pos - cache["tgt"]  # [B, J]
    metric = torch.sum(cache["w"] * _banded_sq_err(err, cache["band"]), dim=1)  # [B]
    # Normalize by the total weight so the exponent is the MEAN per-joint weighted
    # squared error, not a sum that grows with the joint count. Without this, with
    # sum_w~49 across 29 joints even a modest 0.2 rad/joint error gives metric~2.0 ->
    # exp(-kp*2.0) saturates to ~0 and the reward is dead for the entire run.
    n_w = cache["w"].sum().clamp(min=1.0)  # scalar total weight (~49.0)
    pelvis_height = asset.data.root_link_pos_w[:, 2]  # [B]
    gate = _standing_gate(pelvis_height, pelvis_height_threshold, band=band)
    return torch.exp(-kp * (metric / n_w)) * gate  # positive reward in [0, 1] * gate


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
    max_base_lin_vel: float = 0.5,
    max_base_ang_vel: float = 1.0,
    max_joint_speed: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward holding a STILL stand for n_hold consecutive steps [B].

    Spec: "stable success hold — deve restare in piedi per N step, non solo
    raggiungere altezza per un frame". Returns 1.0 only after the robot has held
    a QUIET stand — above ``height_threshold`` AND moving slowly — continuously
    for ``n_hold`` steps.

    v25 change (the point of this run): the hold used to gate on pelvis height
    ALONE, so a robot that reached 0.65 m and then wobbled / bounced / drifted in
    place collected the full weight-10 reward the whole time — the single
    highest-weighted post-task term was structurally BLIND to the "stands but
    never settles" failure the videos show. Now the counter only advances while
    the robot is also still: base linear speed < ``max_base_lin_vel`` (m/s),
    horizontal base angular speed < ``max_base_ang_vel`` (rad/s), and RMS
    per-joint speed < ``max_joint_speed`` (rad/s). Any spike on ANY of these
    resets the counter, exactly as dipping below the height threshold already
    did — so a genuinely held, motionless pose is the only way to bank the
    reward.

    Thresholds are deliberately GENEROUS (well above the post_base_*/
    post_joint_stillness dead-zones) so small settling adjustments still count as
    a hold — the goal here is to kill gross wobble, not to demand perfect
    stillness (that gradient is what the dead-zoned post_* stillness rewards are
    for). They are exposed as params so they can be tuned from the config without
    a code change; loosen them if fall_after_success spikes (the counter is too
    strict), tighten them if the final pose still visibly drifts.

    ``ever_stood`` is intentionally kept on the height-only condition so the
    standing_fall_timeout / no_progress_timeout exemption logic that reads it is
    unchanged by this reward-shaping tweak. A separate ``hold_still_ok`` flag is
    stashed for the diagnostic metric (see metrics.hold_stillness_fraction).
    """
    asset: Entity = env.scene[asset_cfg.name]
    state = get_episode_state(env, asset)
    h = asset.data.root_link_pos_w[:, 2]
    above = h > height_threshold

    # Stillness gate: reuse the exact velocity channels the post_base_* /
    # post_joint_stillness dead-zone rewards read, but as HARD cutoffs here.
    base_lin_speed = asset.data.root_link_lin_vel_b[:, :3].norm(dim=1)
    base_ang_speed = asset.data.root_link_ang_vel_b[:, :2].norm(dim=1)
    ctrl_ids = asset.indexing.ctrl_ids
    joint_speed = asset.data.joint_vel[:, ctrl_ids].pow(2).mean(dim=-1).sqrt()  # RMS
    still = (
        (base_lin_speed < max_base_lin_vel)
        & (base_ang_speed < max_base_ang_vel)
        & (joint_speed < max_joint_speed)
    )
    holding = above & still

    # Increment counter while holding a still stand, reset the moment either the
    # height OR the stillness condition is broken.
    state["standing_counter"] = torch.where(
        holding,
        state["standing_counter"] + 1,
        torch.zeros_like(state["standing_counter"]),
    )
    # Once the robot has reached standing HEIGHT in this episode, remember it
    # (height-only, unchanged — feeds the fall/no-progress terminations).
    state["ever_stood"] = state["ever_stood"] | above
    # Diagnostic flags for metrics.hold_stillness_fraction (no training effect):
    # is the robot above standing height, and if so is it also holding still?
    state["stand_height_ok"] = above
    state["stand_still_ok"] = holding
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
    require_upright: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Soft penalty for a joint group leaving the allowed band [lower, upper] [B,].

    `over = clamp(q−upper,0)+clamp(lower−q,0)` per joint, Σ over² · phase_gate.
    Soft (quadratic) version of HoST's binary deviation penalties; the joint group
    is selected per-robot via asset_cfg.joint_ids, the active phase via gate_lo/hi.
    When require_upright=True the gate is additionally multiplied by clamp(-proj_grav_z,0,1)
    so the penalty only fires when the robot is both high enough AND upright.
    """
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    over = torch.clamp(q - upper, min=0.0) + torch.clamp(lower - q, min=0.0)
    pen = torch.sum(over.pow(2), dim=1)
    gate = _stage_gate(asset.data.root_link_pos_w[:, 2], lo=gate_lo, hi=gate_hi, band=band)
    if require_upright:
        gate = gate * torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return pen * gate


def waist_upright_reward(
    env: ManagerBasedRlEnv,
    v_free: float = 0.10,
    margin: float = 0.40,
    value_at_margin: float = 0.1,
    gate_lo: float | None = None,
    gate_hi: float | None = None,
    band: float = 0.08,
    require_upright: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """POSITIVE dead-zone reward for keeping the trunk upright [B,].

    Positive counterpart of joint_group_deviation for the waist: instead of only
    PENALISING deviation outside a band (which gives no pull toward neutral once
    inside), this REWARDS the waist staying near neutral and decays smoothly as it
    tilts. dev = ‖(q_pitch, q_roll)‖; f_tol(dev, 0, v_free, margin) is 1.0 while the
    trunk is within `v_free` rad of straight and decays to `value_at_margin` at
    `v_free + margin`. Same stage/upright gating as the old penalty. Use POSITIVE weight.
    """
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [B, n_waist]
    dev = torch.norm(q, dim=1)  # [B], deviation from neutral (0)
    reward = f_tol(dev, 0.0, v_free, margin, value_at_margin)
    gate = _stage_gate(asset.data.root_link_pos_w[:, 2], lo=gate_lo, hi=gate_hi, band=band)
    if require_upright:
        gate = gate * torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return reward * gate


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
# "Fully standing at HOME" gate used by the low-velocity STILLNESS rewards
# (post_base_lin_vel / post_base_ang_vel). Higher than H_STAGE2 so those terms only
# demand stillness once the robot is genuinely up at HOME — never during the rise.
_HOME_HEIGHT = 0.72


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
    """Binary: penalty if |q_waist_yaw| > limit (HoST waist deviation) [B,]. Gated h>H_STAGE2 + upright."""
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [B, n]
    viol = (q.abs() > limit).any(dim=1).float()
    return penalty * viol * _post_gate(asset, H_STAGE2)


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
    return (penalty * roll_viol + penalty * yaw_viol) * _post_gate(asset, H_STAGE2)


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
    return (penalty * hi_viol + penalty * lo_viol) * _post_gate(asset, H_STAGE2)


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
    return (penalty * left_viol + penalty * right_viol) * _post_gate(asset, H_STAGE2)


def style_foot_displacement(
    env: ManagerBasedRlEnv,
    scale: float = 2.0,
    clip_min: float = 0.3,
    height_threshold: float = H_STAGE2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST CoM-in-support: Σ_feet exp(-2·clamp(||base_xy-foot_xy||², 0.3)) · gate(h) [B,].

    Summed over both feet; configure weight 2.5 (per-foot). The clamp(.,0.3) floors
    the squared distance, so the reward saturates (~0.55/foot) when the base is well
    centered and only drops once a foot is displaced beyond ~0.3 m. Active above
    `height_threshold` (default H_STAGE2; lower it to reward feet-under-CoM earlier in
    the rise). asset_cfg.site_names = (left_foot, right_foot).
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_xy = asset.data.root_link_pos_w[:, :2].unsqueeze(1)  # [B, 1, 2]
    foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
    d2 = ((foot_xy - base_xy).pow(2)).sum(dim=-1)  # [B, F]
    d2 = torch.clamp(d2, min=clip_min)
    r = torch.exp(-scale * d2).sum(dim=1)  # [B]
    # Ramped gate (was a hard step): this term was part of the F1 reward cliff —
    # several standing rewards all switching on discontinuously at 0.65 m. Same band
    # as _post_gate so all standing payoffs fade in together.
    gate = _standing_gate(asset.data.root_link_pos_w[:, 2], height_threshold, band=0.08)
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
    return penalty * (d2 > max_dist_sq).float() * _post_gate(asset, H_STAGE2)


def style_shank_orientation(
    env: ManagerBasedRlEnv,
    knee_names: tuple[str, ...] = ("left_knee_link", "right_knee_link"),
    foot_names: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    lower: float = 0.8,
    margin: float = 1.0,
    value_at_margin: float = 0.1,
    height_threshold: float = H_STAGE1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST shank orientation: f_tol(mean(shank_z), [0.8, inf), m=1, v=0.1) · gate(h) [B,].

    shank_z = (z_knee - z_foot)/||knee - foot|| per leg (1 = vertical shank), averaged
    over both legs. Encourages upright lower legs during the rise. `height_threshold`
    sets where the _standing_gate ramp centers: with the default H_STAGE1=0.45 the
    reward only turns on mid-rise; set it to 0.0 to make the shank signal active from
    the ground up (band ±0.08 m -> ~50% at h=0, fully on by h=0.08 m), giving the legs
    an early reason to tuck vertically rather than the robot leaning its torso up.
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
    gate = _standing_gate(asset.data.root_link_pos_w[:, 2], height_threshold, band=0.08)
    return r * gate


def style_base_ang_vel(
    env: ManagerBasedRlEnv,
    v_free: float = 0.8,
    margin: float = 2.0,
    value_at_margin: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """HoST style base ang-vel: DEAD-ZONED trunk-spin reward during the rise [B,].

    f_tol(||w_xy||, [0, v_free], margin) · 1(h>H_STAGE1): full reward while the trunk's
    horizontal angular speed stays under v_free rad/s (a normal rise is free), decaying
    only for violent spin. Rise-gated (h>H_STAGE1) so it shapes trunk stability WHILE
    getting up; the stronger HOME-gated stillness version lives in the post-task group.
    Reshaped from the old exp(-scale·||w||²) form, which penalised any rotation.
    """
    asset: Entity = env.scene[asset_cfg.name]
    w = asset.data.root_link_ang_vel_b[:, :2].norm(dim=1)
    r = f_tol(w, 0.0, v_free, margin, value_at_margin)
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
    return reward * torch.exp(-tilt / (2 * tilt_threshold)) * _post_gate(asset, H_STAGE1)


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


# Crouch floor: when a post-task reward opts into `ramp_from`, its height gate fades
# in LINEARLY from this height up to the standing threshold (full when standing),
# instead of a narrow band centered on standing. Chosen at the deep-crouch height so
# the standing rewards start pulling while the robot is still crouched (~0.45 m),
# giving a gradient across the whole crouch->stand band. Pairs with pelvis_height_bridge.
_POST_GATE_RAMP_FROM = 0.45


def _post_gate(
    asset: Entity,
    height_threshold: float = H_STAGE2,
    band: float = 0.08,
    ramp_from: float | None = None,
) -> torch.Tensor:
    """Height × upright gate for post-task and style-standing rewards [B,].

    Two height-ramp modes:
      - ramp_from=None (default): symmetric band ramp 0→1 over [threshold-band,
        threshold+band]. Used by the style-standing PENALTIES so their active region
        is unchanged.
      - ramp_from=<h>: linear ramp 0→1 over [ramp_from, threshold], i.e. fully on when
        standing and fading in from the crouch floor. Used by the height/upright/
        anti-jump post terms so they provide a climbing gradient out of the crouch.
        Requires threshold - ramp_from >= 2*band (else the band ramp is used, so a
        low-threshold caller like H_STAGE1 can never divide by ~0).

    Upright factor (both modes): clamp(-proj_grav_z, 0, 1) — 0 when lying, 1 upright.

    band history (genuine tradeoff — both extremes failed in training):
      - band=0.06 -> ramp [0.59, 0.71]: robot learned to BALLISTICALLY JUMP toward
        ~0.71 m to grab the payoff, then fell (commit 71e6646 shrank it).
      - band=0.02 -> ramp [0.63, 0.67]: near-zero gradient below standing. Run
        2026-07-07_14-50-55 converged to a stable feet-planted CROUCH at ~0.5-0.6 m
        (farming post_stand_on_feet, own gate 0.50) and stopped climbing.
      - band=0.08 -> ramp [0.57, 0.73]: fades in from the top of the crouch. Still left
        a dead zone below 0.57 where the observed crouch (0.4-0.6) mostly sits.
      - ramp_from mode (current, for the height/upright/anti-jump post terms) reaches
        the whole way down to _POST_GATE_RAMP_FROM. Ballistic-jump safe because the
        co-gated post_base_lin_vel penalty (weight 15) rides the SAME ramp and the
        per-metre gradient is GENTLER than the old narrow band. If jumping returns
        (task_head_impact spikes, violent rises in video), raise _POST_GATE_RAMP_FROM.
    """
    h = asset.data.root_link_pos_w[:, 2]
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    if ramp_from is not None and (height_threshold - ramp_from) >= 2.0 * band:
        height_gate = torch.clamp(
            (h - ramp_from) / (height_threshold - ramp_from), 0.0, 1.0
        )
    else:
        height_gate = _standing_gate(h, height_threshold, band=band)
    return height_gate * upright


def _banded_sq_err(err: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """Per-joint squared error with a symmetric dead-zone band [..., J].

    ``err``  : signed joint error (q - target), any leading shape, last dim = joints.
    ``band`` : per-joint tolerance HALF-WIDTH (>= 0), broadcastable over ``err``.

    Returns the squared EXCESS error beyond the band: 0 while |err| <= band (the
    joint may sit anywhere in [target-band, target+band] at no cost), then growing
    quadratically once outside, exactly like the raw err**2 shifted outward by
    ``band``. With band == 0 everywhere this is identical to err**2, so it is a
    drop-in that reproduces the pre-band behavior for any joint given band 0.

    Because the excess is clamped to <= |err|, the resulting metric is always <=
    the un-banded metric — so every posture term built on it can only move toward
    a SMALLER penalty / LARGER bounded reward, never a larger magnitude. This is
    the v22 change: the HOME-pose terms stop pulling toward the EXACT keyframe
    (which may not be a holdable equilibrium) and instead tolerate a window around
    it, letting the policy settle into a nearby stable stance.
    """
    excess = torch.clamp(err.abs() - band, min=0.0)
    return excess.pow(2)


def home_pose_l2(
    env: ManagerBasedRlEnv,
    target_joint_pos: dict[str, float],
    joint_weights: dict[str, float],
    joint_bands: dict[str, float] | None = None,
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
        band = torch.tensor(
            [resolve_expr(joint_bands or {}, names, 0.0)], device=env.device
        )
        cache = {"tgt": tgt, "w": w, "band": band}
        setattr(env, _HOME_L2_CACHE_ATTR, cache)
    err = asset.data.joint_pos - cache["tgt"]                       # [B, J]
    metric = torch.sum(cache["w"] * _banded_sq_err(err, cache["band"]), dim=1)  # [B]
    return metric * _post_gate(asset, height_threshold)


def post_base_ang_vel(
    env: ManagerBasedRlEnv,
    v_free: float = 0.5,
    margin: float = 1.5,
    value_at_margin: float = 0.1,
    height_threshold: float = _HOME_HEIGHT,
    band: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task base ang-vel STILLNESS reward, DEAD-ZONED and HOME-gated [B,].

    f_tol(||w_xy||, [0, v_free], margin) · HOME_gate: full reward 1.0 while the trunk's
    horizontal-plane angular speed stays under v_free rad/s (a small-adjustment dead
    zone), decaying only above it. Gated to _HOME_HEIGHT (not H_STAGE2) so it demands
    stillness ONLY once fully standing at HOME — never during the rise. Reshaped from
    the old exp(-scale·||w||²) form, which peaked only at w=0 and penalised any motion.
    """
    asset: Entity = env.scene[asset_cfg.name]
    w = asset.data.root_link_ang_vel_b[:, :2].norm(dim=1)
    return f_tol(w, 0.0, v_free, margin, value_at_margin) * _post_gate(
        asset, height_threshold, band=band
    )


def post_base_lin_vel(
    env: ManagerBasedRlEnv,
    v_free: float = 0.3,
    margin: float = 0.7,
    value_at_margin: float = 0.1,
    height_threshold: float = _HOME_HEIGHT,
    band: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task base lin-vel STILLNESS reward, DEAD-ZONED and HOME-gated [B,].

    f_tol(||v||, [0, v_free], margin) · HOME_gate: full reward 1.0 while the base speed
    stays under v_free m/s (dead zone for small in-place adjustments), decaying above.
    Gated to _HOME_HEIGHT so stillness is required ONLY when fully standing at HOME and
    NEVER during the rise (the old exp form, gated from the crouch up at weight 15,
    penalised exactly the push-off velocity the robot needs to get up). Jumping is now
    handled separately by anti_jump_velocity (an explicit high-speed penalty active from
    the rise up), not by this reward.
    """
    asset: Entity = env.scene[asset_cfg.name]
    speed = asset.data.root_link_lin_vel_b[:, :3].norm(dim=1)
    return f_tol(speed, 0.0, v_free, margin, value_at_margin) * _post_gate(
        asset, height_threshold, band=band
    )


def anti_jump_velocity(
    env: ManagerBasedRlEnv,
    v_high: float = 1.5,
    cap: float = 3.0,
    height_threshold: float = H_STAGE2,
    ramp_from: float = _POST_GATE_RAMP_FROM,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Anti ballistic-jump penalty: squared base-speed EXCESS above v_high [B,].

    Σ clamp(||v|| - v_high, 0, cap)² · gate. Returns a NON-NEGATIVE value — use with a
    NEGATIVE weight. Normal rise / stepping speeds sit under v_high (m/s) and are NOT
    penalised (dead zone below the threshold); only a launch is hit. Gated from
    `ramp_from` (default _POST_GATE_RAMP_FROM = 0.45 m) upward so it stays active
    THROUGHOUT the rise — where the jump-to-stand exploit lives — not just once
    standing. `cap` bounds the per-step penalty so a fall/blow-up transient cannot
    explode the term.

    `ramp_from` is exposed (rather than hardcoded to _POST_GATE_RAMP_FROM) so a caller
    can extend the gate further down into the supine-to-crouch phase (h < 0.45m),
    which otherwise has ZERO velocity shaping — this is the only term that penalizes
    speed during the rise at all, so a lower ramp_from is the intended lever for
    "policy launches too fast in the first phase" (see env_cfgs.py override / plan).
    """
    asset: Entity = env.scene[asset_cfg.name]
    speed = asset.data.root_link_lin_vel_b[:, :3].norm(dim=1)
    excess = torch.clamp(speed - v_high, min=0.0, max=cap)
    return excess.pow(2) * _post_gate(
        asset, height_threshold, ramp_from=ramp_from
    )


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
    # ramp_from: pull the trunk upright from the crouch upward, not just once standing.
    return torch.exp(-scale * theta_xy.pow(2).sum(dim=1)) * _post_gate(
        asset, height_threshold, ramp_from=_POST_GATE_RAMP_FROM
    )


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
    # ramp_from: the key climbing reward — pull the pelvis toward target_height from
    # the crouch upward instead of only rewarding it once already standing.
    return torch.exp(-scale * (h - target_height).pow(2)) * _post_gate(
        asset, height_threshold, ramp_from=_POST_GATE_RAMP_FROM
    )


def post_upper_body_posture(
    env: ManagerBasedRlEnv,
    target_joint_pos: dict[str, float],
    joint_weights: dict[str, float],
    joint_bands: dict[str, float] | None = None,
    scale: float = 1.0,
    k: float = 3.0,
    max_metric: float = 1.0,
    height_threshold: float = H_STAGE2,
    ramp_from: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task upper-body posture PENALTY: scale·(exp(k·metric) − 1) · gate [B,].

    ``ramp_from`` selects the gate's height-ramp mode (see _post_gate): None keeps
    the default symmetric band [0.57, 0.73]; a low anchor (e.g. H_STAGE1=0.45)
    fades the penalty in from Stage 2 entry so arm posture is shaped DURING the
    rise instead of only once standing — the arms-behind-the-back habit forms in
    the rise phase, where the band gate is fully closed and cannot correct it.

    `metric` is the weighted MEAN per-joint squared error from the HOME pose
    (normalized by total joint weight), clamped to `max_metric`. Returns a
    NON-NEGATIVE value — use with a NEGATIVE weight. Best possible is 0, exactly
    at HOME.

    Shape history (each form sim-verified, then evaluated over a standardized
    1000-it resume from the same checkpoint):
    - exp(-scale·Σ err²) REWARD: gradient-dead (unnormalized 17-joint sum
      saturated at any realistic error).
    - linear positive clamp(1−0.5·metric, 0), weight 6: live but weak
      (0.81→0.67 rad RMS, plateau).
    - linear PENALTY metric·weight, -10 then -20: better (→0.60 rad RMS) but
      diminishing returns per weight doubling, plateauing while the residual
      error was still visually wrong.
    This form makes the penalty grow SUPRALINEARLY with the error instead of
    scaling the same linear ramp harder: at the observed plateau (metric≈0.36)
    and especially at the resume start (metric≈0.62) the gradient k·exp(k·m)
    exceeds the outgoing linear form's constant slope, while near HOME it eases
    off (the residual polish there is post_standing_posture's job). Same
    bounded-exponential pattern as the hand-contact penalty (exp(excess/scale)−1
    with a clamp): `max_metric` caps the worst per-step penalty (at defaults
    scale·(e³−1) ≈ 19·scale·|weight|) so a transient flail while the gate is
    open cannot swamp the post_task budget or teach stand-avoidance.

    ``joint_weights`` weights the upper-body joints; ``target_joint_pos`` is the
    HOME pose. Both are {regex: value} dicts resolved once (cached), set per-robot.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cache = getattr(env, _POST_UPPER_CACHE_ATTR, None)
    if cache is None:
        names = asset.joint_names
        tgt = torch.tensor([resolve_expr(target_joint_pos or {}, names, 0.0)], device=env.device)
        w = torch.tensor([resolve_expr(joint_weights or {}, names, 0.0)], device=env.device)
        band = torch.tensor([resolve_expr(joint_bands or {}, names, 0.0)], device=env.device)
        cache = {"tgt": tgt, "w": w, "band": band, "n_w": w.sum().clamp(min=1.0)}
        setattr(env, _POST_UPPER_CACHE_ATTR, cache)
    err = asset.data.joint_pos - cache["tgt"]  # [B, J]
    metric = torch.sum(cache["w"] * _banded_sq_err(err, cache["band"]), dim=1) / cache["n_w"]  # [B]
    metric = torch.clamp(metric, max=max_metric)
    return scale * (torch.exp(k * metric) - 1.0) * _post_gate(
        asset, height_threshold, ramp_from=ramp_from
    )


# Left/right mirror-sign convention, derived from the MJCF joint axes and the
# sagittal (left-right) mirror plane: a rotation about the axis PERPENDICULAR
# to the mirror plane keeps its sense under mirroring (symmetric, q_left ==
# q_right); a rotation about an axis LYING IN the mirror plane has its sense
# REVERSED (anti-symmetric, q_left == -q_right). In this MJCF, pitch joints use
# axis (0,1,0) (perpendicular to the sagittal plane) and roll/yaw joints use
# axis (1,0,0) / (0,0,1) (both in the sagittal plane) — confirmed empirically
# against the two pairs with asymmetric (non-zero-centered) joint ranges:
# shoulder_roll [-1.5882,2.2515]/[-2.2515,1.5882] and hip_roll
# [-0.5236,2.9671]/[-2.9671,0.5236] both negate-and-swap between sides, exactly
# what the anti-symmetric rule predicts. +1 = symmetric, -1 = anti-symmetric.
_MIRROR_SIGN: dict[str, float] = {
    "hip_pitch": 1.0, "hip_roll": -1.0, "hip_yaw": -1.0,
    "knee": 1.0,
    "ankle_pitch": 1.0, "ankle_roll": -1.0,
    "shoulder_pitch": 1.0, "shoulder_roll": -1.0, "shoulder_yaw": -1.0,
    "elbow": 1.0,
    "wrist_roll": -1.0, "wrist_pitch": 1.0, "wrist_yaw": -1.0,
}
_BILATERAL_SYMMETRY_CACHE_ATTR = "_bilateral_symmetry_cache"


def post_bilateral_symmetry(
    env: ManagerBasedRlEnv,
    joint_weights: dict[str, float],
    target_joint_pos: dict[str, float] | None = None,
    joint_bands: dict[str, float] | None = None,
    scale: float = 1.0,
    kp: float = 4.0,
    height_threshold: float = H_STAGE2,
    ramp_from: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the WORSE side's HOME error: exp(-kp·max(metric_L, metric_R))·gate [B,].

    v21 REWRITE (HOME-anchored worst-side). The v16-v20 form compared left and right
    joints to EACH OTHER (`q_left - sign·q_right`): its gradient pulled the GOOD limb
    toward the BAD limb exactly as hard as the reverse — it had no idea which side was
    correct, so a policy could satisfy it in a symmetric-but-wrong basin (both arms
    behind the back scores perfectly). Training data confirmed the harm: the final
    pose was slightly better before v16 added the term and worse after v19 raised its
    leg weights, while in the v20 run it paid only 1.12/8.0 (metric≈0.49 — failing to
    fix the asymmetry it was added for, yet still reshaping the landscape).

    This form keeps the original ANTI-MASKING intent — a MEAN over many joints lets
    one converged limb hide one lagging limb — but re-anchors both sides on the HOME
    target: metric_side = weighted MEAN squared error of that side's joints from
    HOME, and the reward sees only max(metric_L, metric_R). The max ignores the good
    side entirely, so (a) the gradient lands 100% on the worst-deviating side and
    always points AT HOME (never at the other limb), and (b) a good side can never
    mask a bad one. It is maximal (=scale) only when BOTH sides are at HOME.

    ``joint_weights`` keys are joint FAMILY names (e.g. "shoulder_pitch", "elbow",
    "hip_roll" — no left_/right_ prefix, no regex); the value weights that joint on
    BOTH sides. Families are validated against `_MIRROR_SIGN` (same vocabulary as the
    old form, so existing wiring keys stay valid). ``target_joint_pos`` is the HOME
    pose as a {regex: value} dict (same object passed to post_upper_body_posture /
    post_standing_posture), resolved per full joint name; missing joints target 0.0.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cache = getattr(env, _BILATERAL_SYMMETRY_CACHE_ATTR, None)
    if cache is None:
        names = asset.joint_names
        name_to_id = {n: i for i, n in enumerate(names)}
        tgt_full = resolve_expr(target_joint_pos or {}, names, 0.0)
        left_ids, right_ids, tgt_left, tgt_right, weights, bands = [], [], [], [], [], []
        for family, w in (joint_weights or {}).items():
            if family not in _MIRROR_SIGN:
                raise ValueError(
                    f"post_bilateral_symmetry: unknown joint family '{family}' — "
                    f"must be one of {sorted(_MIRROR_SIGN)}"
                )
            left_name, right_name = f"left_{family}_joint", f"right_{family}_joint"
            if left_name not in name_to_id or right_name not in name_to_id:
                raise ValueError(
                    f"post_bilateral_symmetry: joint pair '{left_name}'/'{right_name}' "
                    "not found on the asset"
                )
            left_ids.append(name_to_id[left_name])
            right_ids.append(name_to_id[right_name])
            tgt_left.append(tgt_full[name_to_id[left_name]])
            tgt_right.append(tgt_full[name_to_id[right_name]])
            weights.append(w)
            bands.append(float((joint_bands or {}).get(family, 0.0)))
        w_t = torch.tensor(weights, device=env.device)
        cache = {
            "left_ids": torch.tensor(left_ids, dtype=torch.long, device=env.device),
            "right_ids": torch.tensor(right_ids, dtype=torch.long, device=env.device),
            "tgt_left": torch.tensor(tgt_left, device=env.device),
            "tgt_right": torch.tensor(tgt_right, device=env.device),
            "w": w_t,
            "band": torch.tensor(bands, device=env.device),
            "n_w": w_t.sum().clamp(min=1e-6),
        }
        setattr(env, _BILATERAL_SYMMETRY_CACHE_ATTR, cache)
    q = asset.data.joint_pos
    err_l = q[:, cache["left_ids"]] - cache["tgt_left"]  # [B, P]
    err_r = q[:, cache["right_ids"]] - cache["tgt_right"]  # [B, P]
    metric_l = torch.sum(cache["w"] * _banded_sq_err(err_l, cache["band"]), dim=1) / cache["n_w"]  # [B]
    metric_r = torch.sum(cache["w"] * _banded_sq_err(err_r, cache["band"]), dim=1) / cache["n_w"]  # [B]
    metric = torch.maximum(metric_l, metric_r)  # worst side only
    return scale * torch.exp(-kp * metric) * _post_gate(
        asset, height_threshold, ramp_from=ramp_from
    )


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
    ramp_from: float | None = None,
    joint_names: tuple[str, ...] = ("left_hip_yaw_joint", "right_hip_yaw_joint"),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task feet yaw alignment: exp(-scale·(q_yawL² + q_yawR²))·gate [B,].

    Both feet point straight forward when hip_yaw=0. Decays when the robot toes
    out or in after standing. Peak reward = 1.0 when both hip_yaw = 0.

    ``ramp_from``: None keeps the hard standing band; a low anchor (H_STAGE1=0.45)
    fades the reward in through the rise so hip-yaw alignment is shaped while the
    legs-crossed habit forms, not only once already standing (the v8 run logged
    this term DECLINING 0.40->0.096 over training with the band gate).
    """
    asset: Entity = env.scene[asset_cfg.name]
    ids = _cached_joint_ids(env, asset, joint_names, "_post_feet_yaw_joint_ids")
    q = asset.data.joint_pos[:, ids]  # [B, 2]
    return torch.exp(-scale * q.pow(2).sum(dim=1)) * _post_gate(
        asset, height_threshold, ramp_from=ramp_from
    )


# ---------------------------------------------------------------------------
# v23: FUNCTIONAL final-pose terms. These specify the OBJECTIVE (arms in front,
# whole-body stillness, stay-standing, left/right symmetry) with generous
# tolerance and let the policy discover its own stable joint configuration —
# replacing the exact-HOME-pose imitation terms (post_home_pose_l2 /
# post_standing_posture / post_upper_body_posture, now weight 0) that piled up in
# v16-v22 and created conflicting gradients around a pose the body may not hold.
# ---------------------------------------------------------------------------


def arms_in_front(
    env: ManagerBasedRlEnv,
    fwd_lo: float = 0.10,
    fwd_hi: float = 0.35,
    margin: float = 0.15,
    value_at_margin: float = 0.1,
    torso_body: str = "torso_link",
    height_threshold: float = _HOME_HEIGHT,
    ramp_from: float | None = _POST_GATE_RAMP_FROM,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the palms being IN FRONT of the torso, in a comfortable band [B,].

    Each palm position is expressed in the TORSO frame (yaw-invariant, via
    quat_apply_inverse) and its FORWARD (+X) component relative to the torso is
    scored by f_tol against the band ``[fwd_lo, fwd_hi]`` metres: full reward when
    the hand is far enough forward not to be pinned against the body (>= fwd_lo) but
    not stretched out (<= fwd_hi), Gaussian falloff outside. A hand BEHIND the torso
    (arm-behind-the-back) has forward < 0 and scores ~0.

    Non-prescriptive: it constrains WHERE the hands are, not the joint angles, so the
    policy is free to reach that region however is stable. ``ramp_from`` fades it in
    during the rise so the arms-behind-the-back habit (which forms while rising) is
    shaped, not only corrected once standing. asset_cfg carries the palm site names.
    """
    asset: Entity = env.scene[asset_cfg.name]
    torso_ids = _cached_body_ids(env, asset, (torso_body,), "_arms_torso_id")  # [1]
    torso_pos = asset.data.body_link_pos_w[:, torso_ids, :]   # [B, 1, 3]
    torso_quat = asset.data.body_link_quat_w[:, torso_ids, :]  # [B, 1, 4]
    palm_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]   # [B, P, 3]
    offset_w = palm_w - torso_pos                             # [B, P, 3]
    quat = torso_quat.expand(-1, palm_w.shape[1], -1)         # [B, P, 4]
    offset_b = quat_apply_inverse(quat, offset_w)             # [B, P, 3]
    fwd = offset_b[..., 0]                                    # [B, P] forward comp
    r = f_tol(fwd, fwd_lo, fwd_hi, margin, value_at_margin)   # [B, P]
    return r.mean(dim=1) * _post_gate(asset, height_threshold, ramp_from=ramp_from)


def post_joint_stillness(
    env: ManagerBasedRlEnv,
    v_free: float = 0.5,
    margin: float = 1.5,
    value_at_margin: float = 0.1,
    height_threshold: float = _HOME_HEIGHT,
    band: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task WHOLE-BODY joint stillness, dead-zoned and HOME-gated [B,].

    f_tol(rms(q̇_actuated), [0, v_free], margin) · HOME_gate: full reward while the
    RMS per-joint velocity stays under ``v_free`` (dead zone for small settling
    adjustments), decaying above. Mirrors post_base_lin_vel / post_base_ang_vel but on
    JOINT velocity, so the policy is rewarded for holding a quiet pose once standing —
    not only a quiet base. Gated to _HOME_HEIGHT so it never penalises the push-off
    motion during the rise. Deliberately SECONDARY to the stability terms (being stably
    up matters more than being motionless); wire it at a modest weight.

    NOTE (v24 fix): uses RMS (dof-count-invariant "typical per-joint speed"), NOT the
    raw L2 norm over all 29 joints. The v23 norm form made ‖q̇‖ blow past v_free=0.5
    on any small residual motion (a sum over 29 dofs), so the term logged ~0.0004 —
    it never fired. RMS makes v_free a per-joint threshold matching the design intent.
    """
    asset: Entity = env.scene[asset_cfg.name]
    ctrl_ids = asset.indexing.ctrl_ids
    speed = asset.data.joint_vel[:, ctrl_ids].pow(2).mean(dim=-1).sqrt()  # RMS per joint
    return f_tol(speed, 0.0, v_free, margin, value_at_margin) * _post_gate(
        asset, height_threshold, band=band
    )


def standing_alive_bonus(
    env: ManagerBasedRlEnv,
    height_threshold: float = _HOME_HEIGHT,
    band: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Small continuous 'alive' bonus for BEING upright at standing height [B,].

    Returns the height×upright gate itself (in [0, 1]): a smooth per-step reward for
    every step spent standing, so the policy is encouraged to STAY up (duration), not
    only to reach the height once. Kept at LOW weight — post_base_height already rewards
    being at the target height, so this only adds a gentle survival incentive and must
    not dominate the stability shaping.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return _post_gate(asset, height_threshold, band=band)


# ---------------------------------------------------------------------------
# Terminal "quiet standing" region (task-space, DeepMimic-style). Instead of
# imposing an exact HOME joint configuration, these terms define WHAT a settled
# stand IS — trunk vertical, both feet loaded, CoM over the support, and the
# feet correctly LEFT/RIGHT ordered — with generous tolerance, and combine the
# factors MULTIPLICATIVELY (geometric mean / product), so no single factor can
# be farmed while another is ~0. leg_width_ordering is the direct crossed-legs
# fix (signed lateral foot separation in the pelvis yaw-frame); post_terminal_core
# is the geometric-mean core; terminal_gate is the continuous g4 gate (diagnostic).
# ---------------------------------------------------------------------------


def _pelvis_yaw_frame_xy(asset: Entity, pos_w: torch.Tensor) -> torch.Tensor:
    """XY offset of world positions from the pelvis, in the pelvis YAW-ONLY frame.

    ``pos_w`` : [B, N, 3] world positions (e.g. the two foot sites). Returns
    [B, N, 2] = (x=forward, y=left) after subtracting the pelvis origin and
    rotating by the INVERSE of the pelvis yaw-only orientation. Yaw-only (via
    ``yaw_quat``) makes the result invariant to the trunk's roll/pitch — which
    tilt heavily while getting up — and to the robot's heading, leaving a clean
    body-lateral (y) coordinate for the left/right foot-ordering check. Mirrors
    the torso-frame projection in ``arms_in_front`` but with a yaw-only rotation.
    """
    root_pos = asset.data.root_link_pos_w[:, :3].unsqueeze(1)      # [B, 1, 3]
    yaw_q = yaw_quat(asset.data.root_link_quat_w)                  # [B, 4]
    yaw_q = yaw_q.unsqueeze(1).expand(-1, pos_w.shape[1], -1)      # [B, N, 4]
    offset_b = quat_apply_inverse(yaw_q, pos_w - root_pos)         # [B, N, 3]
    return offset_b[..., :2]                                       # [B, N, 2]


def leg_width_ordering(
    env: ManagerBasedRlEnv,
    dmin: float = 0.10,
    dmax: float = 0.30,
    margin: float = 0.10,
    value_at_margin: float = 0.1,
    height_threshold: float = H_STAGE2,
    ramp_from: float | None = _POST_GATE_RAMP_FROM,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the feet being LEFT-of-RIGHT with a comfortable lateral gap [B,].

    The single most direct fix for the crossed-legs failure. The two foot sites
    (``asset_cfg.site_ids`` = left, right) are expressed in the pelvis YAW-ONLY
    frame and the SIGNED lateral separation ``y_L - y_R`` (body +Y = left, so a
    correctly-ordered stance has y_L > y_R) is scored by ``f_tol`` against the band
    [dmin, dmax] metres:
      - crossed legs (y_L - y_R < 0)          -> ~0 (well below dmin)
      - too narrow  (0 <= y_L - y_R < dmin)   -> Gaussian falloff toward 0
      - good stance (dmin <= . <= dmax)       -> full reward 1.0
      - too wide    (y_L - y_R > dmax)        -> Gaussian falloff
    Signed (not absolute), so ORDER matters — exactly what a plain foot-distance
    term (which uses the absolute gap) cannot see. Gated near-standing via
    ``_post_gate``, faded in from ``ramp_from`` so the leg-crossing habit is shaped
    through the rise, not only corrected once already standing. asset_cfg carries
    the two foot site names.
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]       # [B, 2, 3] (L, R)
    xy = _pelvis_yaw_frame_xy(asset, foot_w)                       # [B, 2, 2]
    y_sep = xy[:, 0, 1] - xy[:, 1, 1]                              # y_L - y_R [B]
    r = f_tol(y_sep, dmin, dmax, margin, value_at_margin)         # [B]
    return r * _post_gate(asset, height_threshold, ramp_from=ramp_from)


def shank_worst_leg(
    env: ManagerBasedRlEnv,
    knee_names: tuple[str, ...] = ("left_knee_link", "right_knee_link"),
    foot_names: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    min_cos: float = 0.9,
    height_threshold: float = H_STAGE2,
    ramp_from: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Post-task WORST-LEG shank-verticality penalty [B,] (use a NEGATIVE weight).

    Per-leg cos = (z_knee - z_foot)/‖knee - foot‖ (1 = shank vertical). Penalty =
    clamp(min_cos - min_over_legs(cos), 0)² — evaluated on the WORST leg (min cos)
    so a good leg can never hide a splayed one. This is the per-leg ``min`` form the
    plan asks for, mirroring ``thigh_orientation`` (rewards.py); the existing
    ``shank_orientation`` instead SUMS both legs, letting one vertical shank offset
    a bad one. A vertical shank under each hip is what makes the final stand stack
    stably instead of A-framing. Gated to the near-standing HOLD window via
    ``_post_gate`` (ramp_from=None → standing-band only, so it never fights the
    legitimately-bent-shank crouch during the rise, unlike the rise-phase
    ``shank_orientation``).
    """
    asset: Entity = env.scene[asset_cfg.name]
    knee_ids = _cached_body_ids(env, asset, knee_names, "_shank_worst_knee_ids")
    foot_ids = _cached_body_ids(env, asset, foot_names, "_shank_worst_foot_ids")
    vec = (
        asset.data.body_link_pos_w[:, knee_ids, :]
        - asset.data.body_link_pos_w[:, foot_ids, :]
    )  # [B, 2, 3]
    cos = vec[..., 2] / (torch.norm(vec, dim=-1) + 1e-6)  # [B, 2]
    worst = cos.min(dim=1).values                          # [B]
    pen = torch.clamp(min_cos - worst, min=0.0).pow(2)     # [B]
    return pen * _post_gate(asset, height_threshold, ramp_from=ramp_from)


def post_terminal_core(
    env: ManagerBasedRlEnv,
    foot_sensor_name: str,
    dmin: float = 0.10,
    dmax: float = 0.30,
    width_margin: float = 0.10,
    upright_k: float = 4.0,
    support_dist_scale: float = 8.0,
    foot_height_threshold: float = 0.1,
    height_threshold: float = H_STAGE2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Task-space terminal-region CORE reward: GEOMETRIC MEAN of 4 factors [B,].

    ``r_core = (r_upright · r_double_support · r_support_margin · r_leg_ordering)^0.25``,
    each factor in [0, 1], then gated to near-standing by ``_post_gate``. The
    geometric mean (NOT a weighted sum) means no factor can be farmed while another
    sits at ~0: a settled terminal stand requires ALL of trunk-vertical, BOTH feet
    planted, CoM over the support midpoint, and correctly-ordered feet. This is the
    DeepMimic-style "terminal region in task space" reward — it specifies WHAT a
    stable stand is, not which joint angles to hit. Factors:
      - r_upright        = exp(-upright_k·(1 + proj_grav_z)); 1 upright, ~0 flat
                           (same form as ``body_up_exp``).
      - r_double_support = mean over feet of (in_contact AND low) ∈ {0, 0.5, 1};
                           graded so a momentary single-foot lift does not hard-zero
                           the core (the harder both-feet pressure lives in the
                           separate ``post_stand_on_feet`` term).
      - r_support_margin = exp(-support_dist_scale·‖CoM_xy - feet_mid_xy‖²) using the
                           whole-body CoM (subtree_com) — same signal as
                           ``com_over_support``.
      - r_leg_ordering   = f_tol(y_L - y_R, [dmin, dmax], width_margin); the pelvis
                           yaw-frame signed lateral foot separation (crossed-legs
                           factor, same as ``leg_width_ordering``).
    asset_cfg carries the two foot site names; foot_sensor_name is the feet ground
    contact sensor. Use a POSITIVE weight.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # r_upright: torso vertical (body_up_exp form) — 1 upright, ~0 flat/inverted.
    pg_z = asset.data.projected_gravity_b[:, 2]
    r_upright = torch.exp(-upright_k * (1.0 + pg_z))

    # r_double_support: graded both-feet-planted (contact AND low height).
    contact_sensor: ContactSensor = env.scene[foot_sensor_name]
    assert contact_sensor.data.found is not None
    in_contact = contact_sensor.data.found > 0                    # [B, N_feet]
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]      # [B, N_feet]
    planted = (in_contact & (foot_z < foot_height_threshold)).float()
    r_double_support = planted.mean(dim=1)                        # [B] in {0, .5, 1}

    # r_support_margin: whole-body CoM over the feet midpoint (com_over_support form).
    root_id = asset.data.indexing.root_body_id
    com_xy = asset.data.data.subtree_com[:, root_id, :2]          # [B, 2]
    feet_mid_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2].mean(dim=1)  # [B, 2]
    dist = torch.norm(com_xy - feet_mid_xy, dim=-1)              # [B]
    r_support_margin = torch.exp(-support_dist_scale * dist.pow(2))

    # r_leg_ordering: signed pelvis-yaw-frame lateral foot separation (f_tol band).
    foot_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]      # [B, 2, 3]
    xy = _pelvis_yaw_frame_xy(asset, foot_w)                      # [B, 2, 2]
    y_sep = xy[:, 0, 1] - xy[:, 1, 1]                             # y_L - y_R [B]
    r_leg_ordering = f_tol(y_sep, dmin, dmax, width_margin, 0.1)  # [B]

    core = (
        r_upright * r_double_support * r_support_margin * r_leg_ordering
    ).clamp(min=0.0)
    return core.pow(0.25) * _post_gate(asset, height_threshold)


def terminal_gate(
    env: ManagerBasedRlEnv,
    foot_sensor_name: str,
    height_threshold: float = _HOME_HEIGHT,
    f_min: float = 50.0,
    s_f: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Continuous terminal-stability gate g4 in [0, 1] [B,] (diagnostic).

    ``g4 = g_height · g_upright · g_feet``, where ``g_height·g_upright`` is the
    existing ``_post_gate`` (smooth height ramp × uprightness) and ``g_feet =
    sigmoid(s_f·(Fz_L - f_min)) · sigmoid(s_f·(Fz_R - f_min))`` uses the per-foot
    VERTICAL contact force from the ``feet_ground_contact`` netforce sensor (global
    frame → column 2 is Fz). g4 is ~1 only when the robot is at HOME height, upright,
    AND actively loading BOTH feet — the task-space definition of a settled quiet
    stand. Exposed as a metric (``success/terminal_gate``) so the terminal region is
    visible in TensorBoard; the trained core (``post_terminal_core``) computes its
    own internal gate rather than depending on this value.
    """
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[foot_sensor_name]
    force = sensor.data.force
    if force is None:
        g_feet = torch.zeros(env.num_envs, device=env.device)
    else:
        # Vertical contact-force MAGNITUDE per foot. The feet_ground_contact netforce
        # sensor reports Fz with a negative sign (ground-reaction convention: a loaded
        # foot reads ~-160 N, i.e. half body weight), so take abs() — we care about how
        # hard each foot is loaded, not the sign.
        fz = force[:, 2:3] if force.ndim == 2 else force[..., 2]  # [B,1] or [B,N_feet]
        g = torch.sigmoid(s_f * (fz.abs() - f_min))               # [B, .]
        g_feet = g.prod(dim=1)                                    # [B]
    return _post_gate(asset, height_threshold) * g_feet


# ---------------------------------------------------------------------------
# v27: FUNCTIONAL task-space ARM/HAND terminal-pose terms. Decompose the single
# coarse `arms_in_front` (forward-X only, now weight 0) into unambiguous,
# individually-logged factors — hand PLACEMENT (3D torso-frame box, worst hand),
# ELBOW flexion band, shoulder-hand OVEREXTENSION, and self-collision CLEARANCE —
# mirroring the v26 leg terminal-region family. Bands calibrated to the measured
# HOME arm pose in the torso frame: palm x≈0.157, |y|≈0.227, z≈-0.124,
# shoulder→palm≈0.422, radial≈0.276. All gated to the near-standing HOLD window
# (_HOME_HEIGHT, ramp_from=None) so the two PENALTIES never fire during floor
# push-off, when the hands are legitimately near the body/ground. Symmetry is a
# DEFERRED later pass (region first) — see joint_symmetry_mirror below.
# ---------------------------------------------------------------------------

_ARM_TORSO_ID_CACHE = "_arm_torso_id"
_ARM_OVEREXT_SHOULDER_IDS = "_arm_overext_shoulder_ids"
_ARM_ELBOW_JOINT_IDS = "_arm_elbow_joint_ids"


def _palm_offset_torso_frame(
    env: "ManagerBasedRlEnv",
    asset: Entity,
    torso_body: str,
    site_ids: torch.Tensor,
) -> torch.Tensor:
    """Palm site positions in the TORSO frame [B, P, 3] (x=fwd, y=left, z=up).

    Same projection as ``arms_in_front``: the FULL torso quaternion (yaw+roll+pitch),
    so forward/lateral/up are the trunk's own axes as it tilts. Shared by
    ``arm_hand_box`` (placement) and ``arm_clearance`` (radial proxy).
    """
    torso_ids = _cached_body_ids(env, asset, (torso_body,), _ARM_TORSO_ID_CACHE)  # [1]
    torso_pos = asset.data.body_link_pos_w[:, torso_ids, :]     # [B, 1, 3]
    torso_quat = asset.data.body_link_quat_w[:, torso_ids, :]   # [B, 1, 4]
    palm_w = asset.data.site_pos_w[:, site_ids, :]              # [B, P, 3]
    quat = torso_quat.expand(-1, palm_w.shape[1], -1)           # [B, P, 4]
    return quat_apply_inverse(quat, palm_w - torso_pos)         # [B, P, 3]


def arm_hand_box(
    env: ManagerBasedRlEnv,
    x_lo: float = 0.12,
    x_hi: float = 0.30,
    y_lo: float = 0.12,
    y_hi: float = 0.32,
    z_lo: float = -0.30,
    z_hi: float = 0.00,
    margin: float = 0.10,
    value_at_margin: float = 0.1,
    torso_body: str = "torso_link",
    height_threshold: float = _HOME_HEIGHT,
    ramp_from: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward each palm sitting in a comfortable 3D BOX in the torso frame [B,].

    Worst-hand geometric mean of three tolerance bands per palm:
      - x (forward) in [x_lo, x_hi]: in front of the trunk, not pinned, not extended.
      - y_out       in [y_lo, y_hi]: OUTWARD lateral (y_out = +y for the LEFT palm,
        −y for the RIGHT): enforces left-hand-left / right-hand-right; a hand crossed
        to the wrong side has y_out < 0 → ~0 (same signed trick as leg_width_ordering).
      - z (up)      in [z_lo, z_hi]: hand height between pelvis and lower torso.
    ``r_hand = (f_tol(x)·f_tol(y_out)·f_tol(z))^(1/3)``; return ``min`` over the two
    hands × gate, so one badly-placed hand cannot hide behind a well-placed one
    (worst-of, like shank_worst_leg). Replaces the forward-only ``arms_in_front``.
    Bands calibrated to HOME (x≈0.157, |y|≈0.227, z≈−0.124). asset_cfg carries the two
    palm sites in ORDER (left, right).
    """
    asset: Entity = env.scene[asset_cfg.name]
    off_b = _palm_offset_torso_frame(env, asset, torso_body, asset_cfg.site_ids)  # [B,P,3]
    assert off_b.shape[1] == 2, "arm_hand_box expects exactly 2 palm sites (L, R)"
    x = off_b[..., 0]                                              # [B, 2]
    z = off_b[..., 2]                                              # [B, 2]
    # OUTWARD lateral: +y for the left palm (site 0), −y for the right (site 1).
    sign = torch.tensor([1.0, -1.0], device=env.device).view(1, -1)  # [1, 2]
    y_out = off_b[..., 1] * sign                                   # [B, 2]
    r_x = f_tol(x, x_lo, x_hi, margin, value_at_margin)
    r_y = f_tol(y_out, y_lo, y_hi, margin, value_at_margin)
    r_z = f_tol(z, z_lo, z_hi, margin, value_at_margin)
    r_hand = (r_x * r_y * r_z).clamp(min=0.0).pow(1.0 / 3.0)       # [B, 2]
    worst = r_hand.min(dim=1).values                              # [B]
    return worst * _post_gate(asset, height_threshold, ramp_from=ramp_from)


def arm_elbow_flexion(
    env: ManagerBasedRlEnv,
    e_lo: float = 0.35,
    e_hi: float = 0.95,
    margin: float = 0.15,
    value_at_margin: float = 0.1,
    elbow_joints: tuple[str, ...] = ("left_elbow_joint", "right_elbow_joint"),
    height_threshold: float = _HOME_HEIGHT,
    ramp_from: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward both elbows sitting in a comfortable FLEXION band [B,].

    ``f_tol(q_elbow, [e_lo, e_hi])`` averaged over the two elbows × gate. Rejects a
    fully-straight (locked, q≈0) arm and a hyper-flexed one; centered on HOME 0.60.
    Replaces prescribing 'arms outstretched' — a bent elbow is what a natural relaxed
    stand shows. Both elbows share sign, so one band serves both.
    """
    asset: Entity = env.scene[asset_cfg.name]
    ids = _cached_joint_ids(env, asset, elbow_joints, _ARM_ELBOW_JOINT_IDS)
    q = asset.data.joint_pos[:, ids]                              # [B, 2]
    r = f_tol(q, e_lo, e_hi, margin, value_at_margin).mean(dim=1)  # [B]
    return r * _post_gate(asset, height_threshold, ramp_from=ramp_from)


def arm_overextension(
    env: ManagerBasedRlEnv,
    l_max: float = 0.50,
    shoulder_bodies: tuple[str, ...] = (
        "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    ),
    height_threshold: float = _HOME_HEIGHT,
    ramp_from: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize a hand reaching past a max arm length [B,] (use NEGATIVE weight).

    ``pen = mean_hands ReLU(‖palm − shoulder‖ − l_max)² × gate`` — a squared hinge
    (dead-zone below l_max, quadratic beyond), the same robust form as shank_worst_leg
    / hand_contact_penalty. Discourages the straight-locked over-reach a plain
    'arms in front' term can encourage. l_max sits just above the HOME bent-elbow reach
    (≈0.42) so HOME is free and only genuine over-extension is charged. asset_cfg
    carries the palm sites (order left, right); shoulder_bodies the matching proximal
    links.
    """
    asset: Entity = env.scene[asset_cfg.name]
    sh_ids = _cached_body_ids(env, asset, shoulder_bodies, _ARM_OVEREXT_SHOULDER_IDS)
    palm_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]      # [B, 2, 3]
    sh_w = asset.data.body_link_pos_w[:, sh_ids, :]              # [B, 2, 3]
    d = torch.norm(palm_w - sh_w, dim=-1)                        # [B, 2]
    pen = torch.clamp(d - l_max, min=0.0).pow(2).mean(dim=1)     # [B]
    return pen * _post_gate(asset, height_threshold, ramp_from=ramp_from)


def arm_clearance(
    env: ManagerBasedRlEnv,
    d_safe: float = 0.15,
    torso_body: str = "torso_link",
    height_threshold: float = _HOME_HEIGHT,
    ramp_from: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize a palm pinned against the trunk, BEFORE contact [B,] (NEGATIVE weight).

    Torso-frame radial proxy: ``d_radial = sqrt(x_b² + y_b²)`` is the palm's horizontal
    distance from the trunk's central (vertical) axis. ``pen = mean_hands
    ReLU(d_safe − d_radial)² × gate`` — a squared hinge that grows as a hand comes
    within d_safe of the torso axis (jammed against chest/back), a PREDICTIVE margin the
    force-based hand_contact_penalty only sees on actual contact. d_safe < HOME radial
    (≈0.276) so a normal stand is free. asset_cfg carries the palm sites.
    """
    asset: Entity = env.scene[asset_cfg.name]
    off_b = _palm_offset_torso_frame(env, asset, torso_body, asset_cfg.site_ids)  # [B,P,3]
    d_radial = torch.sqrt(off_b[..., 0].pow(2) + off_b[..., 1].pow(2))  # [B, P]
    pen = torch.clamp(d_safe - d_radial, min=0.0).pow(2).mean(dim=1)    # [B]
    return pen * _post_gate(asset, height_threshold, ramp_from=ramp_from)


_SYMMETRY_MIRROR_CACHE_ATTR = "_symmetry_mirror_cache"


def joint_symmetry_mirror(
    env: ManagerBasedRlEnv,
    joint_weights: dict[str, float],
    scale: float = 1.0,
    kp: float = 4.0,
    height_threshold: float = H_STAGE2,
    ramp_from: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward LEFT/RIGHT joints MIRRORING each other — no HOME target [B,].

    metric = weighted mean over joint FAMILIES of (q_left − sign·q_right)², where
    ``sign`` is the sagittal mirror sign (_MIRROR_SIGN: +1 symmetric, −1 anti-symmetric).
    reward = scale · exp(−kp · metric) · gate, maximal (=scale) only when the two sides
    perfectly mirror. Purely RELATIONAL: it asks the sides to match each OTHER, never to
    match any absolute pose, so it does not re-introduce HOME prescription. The degenerate
    'both limbs identically wrong' basin (e.g. both arms behind the back) is ruled out by
    arms_in_front. ``joint_weights`` keys are joint FAMILY names (no left_/right_ prefix).
    """
    asset: Entity = env.scene[asset_cfg.name]
    cache = getattr(env, _SYMMETRY_MIRROR_CACHE_ATTR, None)
    if cache is None:
        names = asset.joint_names
        name_to_id = {n: i for i, n in enumerate(names)}
        left_ids, right_ids, signs, weights = [], [], [], []
        for family, w in (joint_weights or {}).items():
            if family not in _MIRROR_SIGN:
                raise ValueError(
                    f"joint_symmetry_mirror: unknown joint family '{family}' — "
                    f"must be one of {sorted(_MIRROR_SIGN)}"
                )
            left_name, right_name = f"left_{family}_joint", f"right_{family}_joint"
            if left_name not in name_to_id or right_name not in name_to_id:
                raise ValueError(
                    f"joint_symmetry_mirror: joint pair '{left_name}'/'{right_name}' "
                    "not found on the asset"
                )
            left_ids.append(name_to_id[left_name])
            right_ids.append(name_to_id[right_name])
            signs.append(_MIRROR_SIGN[family])
            weights.append(w)
        w_t = torch.tensor(weights, device=env.device)
        cache = {
            "left_ids": torch.tensor(left_ids, dtype=torch.long, device=env.device),
            "right_ids": torch.tensor(right_ids, dtype=torch.long, device=env.device),
            "sign": torch.tensor(signs, device=env.device),
            "w": w_t,
            "n_w": w_t.sum().clamp(min=1e-6),
        }
        setattr(env, _SYMMETRY_MIRROR_CACHE_ATTR, cache)
    q = asset.data.joint_pos
    err = q[:, cache["left_ids"]] - cache["sign"] * q[:, cache["right_ids"]]  # [B, P]
    metric = torch.sum(cache["w"] * err.pow(2), dim=1) / cache["n_w"]  # [B]
    return scale * torch.exp(-kp * metric) * _post_gate(
        asset, height_threshold, ramp_from=ramp_from
    )


_JOINT_VEL_LIMIT_CACHE = "_joint_vel_limit_cache"
_ARM_VEL_SOFT_LIMIT_CACHE = "_arm_vel_soft_limit_cache"


def action_acc_l2_deadzone(
    env: ManagerBasedRlEnv,
    deadzone: float = 1.0,
) -> torch.Tensor:
    """Jerk penalty with dead zone: Σ_j [|a_t - 2a_{t-1} + a_{t-2}| - deadzone]₊² [B,].

    Only penalizes jerk that exceeds the dead zone. Small and moderate acceleration
    during standup are free; only violent jerk is penalized.

    With actions in [-1,1], max second difference = 4. At deadzone=1.0:
      per-dim max = (4-1)² = 9  →  29 action dims × 9 = 261  →  at weight -0.01 → max ≈ -2.61
    """
    action_acc = (
        env.action_manager.action
        - 2 * env.action_manager.prev_action
        + env.action_manager.prev_prev_action
    )  # [B, A]
    excess = torch.clamp(action_acc.abs() - deadzone, min=0.0)
    return torch.sum(excess.pow(2), dim=1)


def arm_vel_soft_limit(
    env: ManagerBasedRlEnv,
    vel_limits: dict[str, float] | None = None,
    max_excess: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Soft arm velocity threshold: Σ_i clamp(|q̇_i| - limit_i, 0) [B,].

    Same logic as joint_vel_limits but uses a separate cache so both terms can
    coexist on the same env. Unmatched joints get limit 1e6 (never penalized).
    Use a negative weight; the function returns a NON-NEGATIVE violation amount.

    EXPLOSION GUARD: `max_excess` (rad/s) caps the per-joint over-limit amount so a
    single velocity spike (sim glitch / fall transient) cannot produce a huge one-step
    penalty. None = unbounded (original behaviour). With max_excess set, each joint
    contributes at most `max_excess`, so the per-step penalty is bounded by
    n_arm_joints * max_excess * |weight|.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cache = getattr(env, _ARM_VEL_SOFT_LIMIT_CACHE, None)
    if cache is None:
        cache = torch.tensor(
            [resolve_expr(vel_limits or {}, asset.joint_names, 1e6)], device=env.device
        )  # [1, J]
        setattr(env, _ARM_VEL_SOFT_LIMIT_CACHE, cache)
    over = torch.clamp(asset.data.joint_vel.abs() - cache, min=0.0)
    if max_excess is not None:
        over = over.clamp(max=max_excess)
    return torch.sum(over, dim=1)


def joint_vel_limits(
    env: ManagerBasedRlEnv,
    vel_limits: dict[str, float] | None = None,
    soft_ratio: float = 1.0,
    normalize: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Joint velocity limit penalty, optionally normalized and threshold-gated [B,].

    Default (soft_ratio=1.0, normalize=False): original HoST behavior —
      Σ clamp(|q̇| - limit, 0), unbounded, per-joint in rad/s.

    With soft_ratio=0.85, normalize=True (recommended):
      e_j = clamp((|q̇_j| - 0.85*lim_j) / (0.15*lim_j), 0, 1)
      return Σ e_j  →  each joint contributes at most 1; total bounded by num_joints.
      No penalty below 85% of limit; smooth ramp from 85% to 100%; hard cap at 1.
      At weight -0.15: max penalty = -0.15 * num_joints ≈ -6.45 for 43 joints.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cache = getattr(env, _JOINT_VEL_LIMIT_CACHE, None)
    if cache is None:
        raw = torch.tensor(
            [resolve_expr(vel_limits or {}, asset.joint_names, 1e6)], device=env.device
        )  # [1, J]
        lo = raw * soft_ratio
        span = (raw * (1.0 - soft_ratio)).clamp(min=1e-6)
        cache = {"lo": lo, "span": span, "normalize": normalize}
        setattr(env, _JOINT_VEL_LIMIT_CACHE, cache)
    over = torch.clamp(asset.data.joint_vel.abs() - cache["lo"], min=0.0)
    if cache["normalize"]:
        over = (over / cache["span"]).clamp(max=1.0)
    return torch.sum(over, dim=1)
