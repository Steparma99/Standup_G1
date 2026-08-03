"""Shared standing-status evaluator — ONE definition of standing for the whole task.

`compute_standing_status(env, asset, cfg)` is the single source of truth used by:
  * post-task rewards (upright/support/stillness/hold gating),
  * success metrics,
  * the assistance-force curriculum,
  * the beta curriculum,
  * the pose-randomization + pose-expansion curricula,
  * the phase-dependent multi-critic advantage mask.

No subsystem re-derives its own height/upright/stability thresholds — they all read the
StabilityCfg carried on the env. The evaluator MUTATES per-episode counters (in the shared
episode_state), so it is memoised per policy step (keyed by env.common_step_counter): the
first caller in a step computes + advances the counters, every later caller in the same
step gets the identical cached result. Counters are reset per-episode by the
`reset_episode_state` event (mdp/events.py).

Returned dict (all tensors [B], bool or long/float):
    standing_candidate   bool  candidate condition met THIS step (pre-latch)
    standing_latched     bool  hysteretic standing latch (the phase boundary)
    stable_now           bool  strict stable-standing condition met THIS step
    stable_counter       long  consecutive stable_now steps
    stable_hold_seconds  float stable_counter / policy_hz
    stable_hold_reached  bool  stable_counter has reached N_hold this episode
    episode_success      bool  hold reached AND not fall_after_stand (valid at episode end)
    fall_after_stand     bool  fell/lost upright after the latch
    first_stable_step    long  step index of first hold (-1 if never)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.string import resolve_expr
from .events import get_episode_state

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from ..curriculum_cfg import StabilityCfg

_STATUS_ATTR = "_standing_status"
_STATUS_STEP_ATTR = "_standing_status_step"
_BODY_JOINT_MASK_ATTR = "_standing_body_joint_mask"
# 29 controlled body joints (exclude the 14 PD-held finger joints) for RMS(qdot).
_BODY_JOINT_PATTERNS = (
    ".*_hip_.*", ".*_knee_joint", ".*_ankle_.*", "waist_.*",
    ".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*",
)


def _body_joint_mask(env: "ManagerBasedRlEnv", asset: "Entity") -> torch.Tensor:
    """Boolean [J] mask selecting the 29 controlled body joints (cached on the env)."""
    m = getattr(env, _BODY_JOINT_MASK_ATTR, None)
    if m is None:
        names = asset.joint_names
        sel = resolve_expr(
            {p: 1.0 for p in _BODY_JOINT_PATTERNS}, names, 0.0
        )
        m = torch.tensor(sel, device=env.device) > 0.5
        setattr(env, _BODY_JOINT_MASK_ATTR, m)
    return m


def _capture_point_margin(env: "ManagerBasedRlEnv", asset: "Entity") -> torch.Tensor:
    """Signed margin (m) of the capture point into the double-foot support polygon [B].

    Reuses the SAME support-polygon + capture-point construction as the com_over_support
    reward (rewards._box_signed_margin / _pelvis_yaw_frame_xy). > 0 inside, < 0 outside.
    """
    # Local import avoids a circular import at module-load (rewards imports standing later).
    from .rewards import _box_signed_margin, _pelvis_yaw_frame_xy

    root_id = asset.data.indexing.root_body_id
    com_w = asset.data.data.subtree_com[:, root_id, :]        # [B,3]
    v_com_w = asset.data.data.subtree_linvel[:, root_id, :]   # [B,3]
    h_com = com_w[:, 2].clamp(min=0.35)
    omega0 = torch.sqrt(torch.tensor(9.81, device=env.device) / h_com)
    xi_xy = com_w[:, :2] + v_com_w[:, :2] / omega0.unsqueeze(1)
    xi_w = torch.cat([xi_xy, com_w[:, 2:3]], dim=1)

    foot_w = asset.data.site_pos_w[:, _foot_site_ids(env, asset), :]  # [B,2,3]
    stacked = torch.cat([foot_w, xi_w.unsqueeze(1)], dim=1)           # [B,3,3]
    proj = _pelvis_yaw_frame_xy(asset, stacked)                       # [B,3,2]
    feet_p = proj[:, :2, :]
    xi_p = proj[:, 2, :]
    fx, fy = feet_p[..., 0], feet_p[..., 1]
    x_min = fx.min(dim=1).values - 0.094
    x_max = fx.max(dim=1).values + 0.092
    y_min = fy.min(dim=1).values - 0.026
    y_max = fy.max(dim=1).values + 0.026
    return _box_signed_margin(xi_p, x_min, x_max, y_min, y_max)       # [B]


_FOOT_SITE_ATTR = "_standing_foot_site_ids"


def _foot_site_ids(env: "ManagerBasedRlEnv", asset: "Entity") -> list[int]:
    ids = getattr(env, _FOOT_SITE_ATTR, None)
    if ids is None:
        names = list(asset.data.site_names) if hasattr(asset.data, "site_names") else []
        want = ["left_foot", "right_foot"]
        ids = [names.index(w) for w in want] if all(w in names for w in want) else [0, 1]
        setattr(env, _FOOT_SITE_ATTR, ids)
    return ids


def _feet_planted(env: "ManagerBasedRlEnv", cfg: "StabilityCfg") -> torch.Tensor:
    """Both feet planted via the hysteretic contact latch (shared with com_over_support)."""
    state = get_episode_state(env, env.scene["robot"])
    latch = state["foot_contact_latch"]                       # [B,2] bool
    try:
        sensor = env.scene["feet_ground_contact"]
        force = sensor.data.force
    except (KeyError, AttributeError):
        force = None
    if force is not None:
        fz = force[..., 2].abs()                              # [B,2]
        latch = torch.where(fz > cfg.foot_plant_on_n, torch.ones_like(latch),
                            torch.where(fz < cfg.foot_plant_off_n, torch.zeros_like(latch), latch))
        state["foot_contact_latch"] = latch
    return latch[:, 0] & latch[:, 1]                          # [B]


def compute_standing_status(
    env: "ManagerBasedRlEnv", asset: "Entity", cfg: "StabilityCfg"
) -> dict[str, torch.Tensor]:
    """Compute (memoised per step) the shared standing status. See module docstring."""
    step = int(env.common_step_counter)
    if getattr(env, _STATUS_STEP_ATTR, None) == step:
        return getattr(env, _STATUS_ATTR)

    state = get_episode_state(env, asset)
    h = asset.data.root_link_pos_w[:, 2]
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    v_base = asset.data.root_link_lin_vel_b[:, :3].norm(dim=1)
    w_base = asset.data.root_link_ang_vel_b[:, :3].norm(dim=1)
    mask = _body_joint_mask(env, asset)
    qd = asset.data.joint_vel[:, mask]
    qd_rms = qd.pow(2).mean(dim=1).sqrt()
    both_feet = _feet_planted(env, cfg)

    # --- 2.1 candidate + hysteretic latch ---------------------------------------
    cand = (h > cfg.candidate_height) & (upright > cfg.candidate_upright) & both_feet
    cand_ctr = torch.where(cand, state["cand_counter"] + 1,
                           torch.zeros_like(state["cand_counter"]))
    state["cand_counter"] = cand_ctr
    latched = state["standing_latched"]
    newly = (~latched) & (cand_ctr >= cfg.candidate_consec_steps)
    latched = latched | newly

    exit_cond = (h < cfg.latch_exit_height) | (upright < cfg.latch_exit_upright)
    exit_ctr = torch.where(exit_cond, state["latch_exit_counter"] + 1,
                           torch.zeros_like(state["latch_exit_counter"]))
    state["latch_exit_counter"] = exit_ctr
    unlatch = latched & (exit_ctr >= cfg.latch_exit_consec_steps)
    latched = latched & ~unlatch
    state["standing_latched"] = latched
    # ever_latched: sticky for the rest of the episode (fall detection keys off this,
    # NOT the hysteretic latch, which would itself exit before the 20-step fall window).
    state["ever_latched"] = state["ever_latched"] | newly

    # --- 2.2 strict stable_now --------------------------------------------------
    cp_margin = _capture_point_margin(env, asset)             # >0 inside polygon
    cp_ok = cp_margin >= -cfg.stable_capture_margin
    cp_ok = cp_ok & torch.isfinite(cp_margin)                 # invalid -> not stable
    stable_now = (
        (h > cfg.stable_height) & (upright > cfg.stable_upright)
        & (v_base < cfg.stable_lin_vel) & (w_base < cfg.stable_ang_vel)
        & (qd_rms < cfg.stable_joint_vel_rms) & both_feet & cp_ok
    )

    # --- 2.3 stable hold counter ------------------------------------------------
    stable_ctr = torch.where(stable_now, state["stable_counter"] + 1,
                             torch.zeros_like(state["stable_counter"]))
    state["stable_counter"] = stable_ctr
    hold_reached = stable_ctr >= cfg.stable_hold_steps
    first_hit = hold_reached & (state["first_stable_step"] < 0)
    state["first_stable_step"] = torch.where(
        first_hit, torch.full_like(state["first_stable_step"], step),
        state["first_stable_step"])
    state["stable_hold_reached"] = state["stable_hold_reached"] | hold_reached
    # accumulate stable steps after the first hold (for median stable-duration metric)
    after_first = state["first_stable_step"] >= 0
    state["stable_step_accum"] = state["stable_step_accum"] + (stable_now & after_first).long()

    # --- 2.4 fall-after-stand (keys off ever_latched, not the hysteretic latch) --
    post_bad = state["ever_latched"] & ((h < cfg.fall_after_stand_height) | (upright < cfg.latch_exit_upright))
    bad_ctr = torch.where(post_bad, state["post_latch_bad_counter"] + 1,
                          torch.zeros_like(state["post_latch_bad_counter"]))
    state["post_latch_bad_counter"] = bad_ctr
    fell = state["fall_after_stand"] | (bad_ctr >= cfg.fall_after_stand_consec_steps)
    state["fall_after_stand"] = fell

    # episode_success is only meaningful at episode end; expose the running predicate.
    episode_success = state["stable_hold_reached"] & (~fell)

    status = {
        "standing_candidate": cand,
        "standing_latched": latched,
        "stable_now": stable_now,
        "stable_counter": stable_ctr,
        "stable_hold_seconds": stable_ctr.float() / cfg.policy_hz,
        "stable_hold_reached": state["stable_hold_reached"].clone(),
        "episode_success": episode_success,
        "fall_after_stand": fell,
        "first_stable_step": state["first_stable_step"].clone(),
    }
    setattr(env, _STATUS_ATTR, status)
    setattr(env, _STATUS_STEP_ATTR, step)
    return status
