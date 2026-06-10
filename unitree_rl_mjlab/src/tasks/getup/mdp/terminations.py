"""Get-up task termination functions.

The robot STARTS on the ground in contact, so most failure terminations (bad
orientation, ground contact, self-collision) would fire at reset and must be
excluded. Only terminations that cannot trigger on a healthy fresh episode are
used here.

Current terminations:
  - time_out, nan_detection       (mjlab built-ins, wired in cfg)
  - root_height_below_minimum     (mjlab built-in, deep penetration safety net)
  - joint_velocity_explosion      (finite blow-up guard, here)
  - base_velocity_out             (base-speed blow-up guard, HoST-style, here)
  - feet_too_high                 (legs waved in the air, HoST-style, here)
  - standing_fall_timeout         (fell after standing for M steps, here)
  - no_progress_timeout           (stalled below a height record for N steps, here)

The velocity / feet guards take a `grace_steps` window so the spawn-drop landing
transient during the unactuated/settling phase never triggers them (HoST gates its
dof_vel / base_vel checks on real_episode_length_buf > unactuated_time the same way).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .events import get_episode_state

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _contact_force_norm(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
) -> torch.Tensor:
    """Return contact-force norm [B,] for a named sensor."""
    sensor = env.scene[sensor_name]
    force = sensor.data.force
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    if force.ndim == 2:
        return torch.norm(force, dim=-1)
    return torch.norm(force, dim=-1).max(dim=1).values


def joint_velocity_explosion(
    env: ManagerBasedRlEnv,
    max_velocity: float = 50.0,
    grace_steps: int = 0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if any joint velocity exceeds a physically impossible threshold [B].

    Catches simulation blow-ups one step before they become NaN. The threshold
    is well above any achievable G1 joint speed (~20-37 rad/s), so it never
    fires on a healthy episode or at reset.

    `grace_steps` (HoST-style): the check is suppressed for the first `grace_steps`
    env-steps so the spawn-drop landing transient during the unactuated/settling
    window never triggers it.
    """
    asset: Entity = env.scene[asset_cfg.name]
    exploded = torch.any(asset.data.joint_vel.abs() > max_velocity, dim=1)
    if grace_steps > 0:
        exploded = exploded & (env.episode_length_buf > grace_steps)
    return exploded


def base_velocity_out(
    env: ManagerBasedRlEnv,
    max_velocity: float = 8.0,
    grace_steps: int = 0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if the base linear speed blows up (HoST `base_vel_out`) [B].

    Safety net against launches / simulation blow-ups of the floating base: a
    healthy get-up moves the pelvis at ~1-2 m/s peak, so a threshold well above
    that (default 8 m/s) only fires on a degenerate launch. Like the joint-velocity
    guard it is suppressed for the first `grace_steps` steps so the spawn-drop
    landing transient during the unactuated/settling window never triggers it.
    """
    asset: Entity = env.scene[asset_cfg.name]
    speed = torch.norm(asset.data.root_link_lin_vel_b, dim=-1)
    out = speed > max_velocity
    if grace_steps > 0:
        out = out & (env.episode_length_buf > grace_steps)
    return out


def feet_too_high(
    env: ManagerBasedRlEnv,
    max_height: float = 0.35,
    grace_steps: int = 0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if the feet are lifted too high (HoST platform/wall reset) [B].

    Catches the degenerate strategy of waving the legs up in the air instead of
    planting them to push the body up: a get-up keeps the feet near the ground
    (they come UNDER the body), so a mean foot height above `max_height` (default
    0.35 m) is non-physical for a real rise. Uses the mean of the foot site
    heights; `asset_cfg` must specify the foot site_names. Suppressed for the
    first `grace_steps` steps for symmetry with the velocity guards.
    """
    asset: Entity = env.scene[asset_cfg.name]
    feet_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N_feet]
    too_high = feet_z.mean(dim=1) > max_height
    if grace_steps > 0:
        too_high = too_high & (env.episode_length_buf > grace_steps)
    return too_high


def standing_fall_timeout(
    env: ManagerBasedRlEnv,
    height_threshold: float = 0.65,
    n_fall_steps: int = 30,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if the robot falls after having stood up [B].

    Spec: "after-standing fall termination — una volta in piedi, se ricade
    davvero, l'episodio deve finire." Fires only after `ever_stood` is True
    (set by the stable_success_hold reward), and only after the robot has been
    continuously below `height_threshold` for `n_fall_steps` consecutive steps
    (~0.3 s at 100 Hz). This prevents terminating on a brief dip during the
    transition to standing.

    Note: terminations run BEFORE rewards in mjlab's step loop, so `ever_stood`
    seen here was written by the previous step's stable_success_hold call
    (1-step lag). This is acceptable.
    """
    asset: Entity = env.scene[asset_cfg.name]
    state = get_episode_state(env, asset)
    h = asset.data.root_link_pos_w[:, 2]
    # Only count as "fallen" if the robot was previously standing this episode.
    fallen = state["ever_stood"] & (h < height_threshold)
    state["fall_counter"] = torch.where(
        fallen,
        state["fall_counter"] + 1,
        torch.zeros_like(state["fall_counter"]),
    )
    return state["fall_counter"] >= n_fall_steps


def no_progress_timeout(
    env: ManagerBasedRlEnv,
    min_progress: float = 0.02,
    n_stall_steps: int = 200,
    grace_steps: int = 50,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if the robot stops gaining height for too long [B].

    Without this, an episode runs the full 1000 steps even if the policy just
    rolls into a half-propped pose and parks there, farming the dense static
    height rewards (base_height_exp / body_up_exp) forever. That makes "lie
    still until timeout" the highest-return strategy and inflates returns so much
    that the early advantages floor the adaptive learning rate.

    Logic: track the best pelvis height reached this episode. Whenever the robot
    beats that record by at least `min_progress` (m), bump the record and reset
    the stall counter; otherwise increment the counter. Fire once the counter
    reaches `n_stall_steps` consecutive steps without a new record. Updating the
    record ONLY on a real jump means a slow-but-genuine ascent keeps resetting the
    counter and is never cut short — only a true height plateau terminates. A
    `grace_steps` window at episode start is exempt so the settle/landing phase
    and the first attempt are never cut short.

    Registered WITHOUT time_out=True, so it counts as a failure termination and
    activates the is_terminated penalty — stalling is punished, not free.

    Reads/writes only its own episode state (best_height, stall_counter), so the
    terminations-before-rewards ordering introduces no lag here.
    """
    asset: Entity = env.scene[asset_cfg.name]
    state = get_episode_state(env, asset)
    h = asset.data.root_link_pos_w[:, 2]
    improved = h > state["best_height"] + min_progress
    state["best_height"] = torch.where(improved, h, state["best_height"])
    state["stall_counter"] = torch.where(
        improved,
        torch.zeros_like(state["stall_counter"]),
        state["stall_counter"] + 1,
    )
    stalled = state["stall_counter"] >= n_stall_steps
    past_grace = env.episode_length_buf > grace_steps
    return stalled & past_grace


def head_impact_termination(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1200.0,
) -> torch.Tensor:
    """Terminate only on violent head impacts, not on light contact [B]."""
    force_norm = _contact_force_norm(env, sensor_name)
    return force_norm > force_threshold
