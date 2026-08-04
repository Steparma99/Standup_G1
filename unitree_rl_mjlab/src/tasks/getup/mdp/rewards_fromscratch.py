"""Simplified post-task reward set for the from-scratch curriculum V2 (spec section 13.4).

Eight terms (A-H) replace the ~20 overlapping post-task terms of the legacy config. ALL
share ONE gate — the hysteretic ``standing_latched`` from the shared standing evaluator
(mdp/standing.compute_standing_status) — instead of each re-deriving its own height gate.
This is the same latch that drives the phase-dependent critic weights and the recovery-
reward gating, so "post-task" means exactly one thing everywhere.

Legacy reward functions are left intact in rewards.py (weight 0 in the new config) for
ablation; nothing here duplicates a still-active legacy term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from .standing import compute_standing_status, get_curriculum_cfg
from .rewards import _resolve_pose_cache, _grouped_pose_value

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "fs_upright", "fs_double_support", "fs_dynamic_stillness", "fs_banded_pose",
    "fs_progressive_hold", "fs_success_bonus", "fs_anti_jump",
    "fs_height_progress", "fs_supine_rising_prep", "fs_prone_recovery",
    "fs_pelvis_height_bridge",
]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _g_post(env: "ManagerBasedRlEnv", asset) -> torch.Tensor:
    """Shared post-task gate = hysteretic standing latch (no flicker) [B]."""
    cfg = get_curriculum_cfg(env)
    return compute_standing_status(env, asset, cfg.stability)["standing_latched"].float()


# --- A. Upright orientation (weight +6) --------------------------------------------
def fs_upright(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    pg_xy = asset.data.projected_gravity_b[:, :2]
    return torch.exp(-8.0 * pg_xy.pow(2).sum(dim=1)) * _g_post(env, asset)


# --- B. Double-foot support (weight +4): min(c_L,c_R), NO half credit for one foot --
def fs_double_support(
    env: "ManagerBasedRlEnv", sensor_name: str = "feet_ground_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cfg = get_curriculum_cfg(env).stability
    try:
        force = env.scene[sensor_name].data.force
    except (KeyError, AttributeError):
        force = None
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    fz = force[..., 2].abs()  # [B,2]
    span = max(cfg.foot_plant_on_n - cfg.foot_plant_off_n, 1e-6)
    c = torch.clamp((fz - cfg.foot_plant_off_n) / span, 0.0, 1.0)  # smooth planted [B,2]
    return c.min(dim=1).values * _g_post(env, asset)


# --- D. Dynamic stillness (weight +5): blended v/omega/qdot exp terms ---------------
def fs_dynamic_stillness(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    from .standing import _body_joint_mask
    asset = env.scene[asset_cfg.name]
    v = asset.data.root_link_lin_vel_b[:, :3].norm(dim=1)
    w = asset.data.root_link_ang_vel_b[:, :3].norm(dim=1)
    qd = asset.data.joint_vel[:, _body_joint_mask(env, asset)]
    qd_rms = qd.pow(2).mean(dim=1).sqrt()
    r_v = torch.exp(-8.0 * v.pow(2))
    r_w = torch.exp(-4.0 * w.pow(2))
    r_qd = torch.exp(-2.0 * qd_rms.pow(2))
    return (0.35 * r_v + 0.35 * r_w + 0.30 * r_qd) * _g_post(env, asset)


# --- E. Banded nominal-pose reward (weight +3): one banded pose cost, exp(-L_q) ------
_FS_POSE_CACHE = "_fs_pose_cache"


def fs_banded_pose(
    env: "ManagerBasedRlEnv",
    target_joint_pos: dict[str, float],
    joint_weights: dict[str, float],
    joint_bands: dict[str, float] | None = None,
    scale: float = 1.0, k: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cache = _resolve_pose_cache(env, asset, _FS_POSE_CACHE, target_joint_pos, joint_weights, joint_bands)
    return _grouped_pose_value(asset, cache, scale, k) * _g_post(env, asset)


# --- F. Progressive stable-hold (weight +5): min(n_stable/N_hold, 1) ----------------
def fs_progressive_hold(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cfg = get_curriculum_cfg(env)
    st = compute_standing_status(env, asset, cfg.stability)
    frac = (st["stable_counter"].float() / max(cfg.stability.stable_hold_steps, 1)).clamp(0.0, 1.0)
    return frac  # inherently 0 unless stable_now; latch-gating would be redundant


# --- G. Stable-success bonus (weight +10): ONE-TIME when hold first reached ----------
def fs_success_bonus(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cfg = get_curriculum_cfg(env)
    st = compute_standing_status(env, asset, cfg.stability)
    just_reached = st["first_stable_step"] == int(env.common_step_counter)
    return just_reached.float()  # +1 exactly on the first-hold step; weight applies the +10


# --- H. Anti-jump safety penalty (weight -2): -ReLU(|v|-1.5)^2 -----------------------
def fs_anti_jump(
    env: "ManagerBasedRlEnv", v_thresh: float = 1.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    v = asset.data.root_link_lin_vel_b[:, :3].norm(dim=1)
    return torch.clamp(v - v_thresh, min=0.0).pow(2) * _g_post(env, asset)


# ======================================================================================
# Section 12: recovery-reward gating. These wrap the legacy recovery rewards and force
# them to ZERO once standing_latched (one shared mask = 1 - latch), so the policy is not
# still paid to "recover" after the recovery is complete. Same params as the base funcs.
# ======================================================================================
def _recovery_gate(env: "ManagerBasedRlEnv", asset) -> torch.Tensor:
    return 1.0 - _g_post(env, asset)  # 1 during ground/rise, 0 after standing latched


def fs_height_progress(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kw):
    from . import rewards as _R
    return _R.height_progress(env, asset_cfg=asset_cfg, **kw) * _recovery_gate(env, env.scene[asset_cfg.name])


def fs_supine_rising_prep(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kw):
    from . import rewards as _R
    return _R.supine_rising_prep(env, asset_cfg=asset_cfg, **kw) * _recovery_gate(env, env.scene[asset_cfg.name])


def fs_prone_recovery(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kw):
    from . import rewards as _R
    return _R.prone_recovery(env, asset_cfg=asset_cfg, **kw) * _recovery_gate(env, env.scene[asset_cfg.name])


def fs_pelvis_height_bridge(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kw):
    from . import rewards as _R
    return _R.pelvis_height_bridge(env, asset_cfg=asset_cfg, **kw) * _recovery_gate(env, env.scene[asset_cfg.name])
