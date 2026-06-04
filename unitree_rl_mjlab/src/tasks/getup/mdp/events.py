"""Custom reset events for the get-up task.

`reset_to_random_keyframe` is the multi-pose, domain-randomized reset:
  1. pick a reference pose at random from a set (e.g. SUPINE, PRONE),
  2. perturb the joints around that pose (per-group ranges), CLAMPED to the
     joint limits so the simulator never gets an out-of-range configuration,
  3. perturb the root pose (height + roll/pitch/yaw) and give random initial
     root velocity (linear + angular).

All ranges are configurable and can be set to 0 to recover a fixed pose — so
you can validate a single fixed pose first, then ramp up randomization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform
from mjlab.utils.string import resolve_expr

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_CACHE_ATTR = "_getup_keyframe_cache"
_EULER_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _range_tensor(range_dict, device) -> torch.Tensor:
    """Turn a {x,y,z,roll,pitch,yaw: (lo,hi)} dict into a (6,2) tensor."""
    rd = range_dict or {}
    rows = [rd.get(k, (0.0, 0.0)) for k in _EULER_KEYS]
    return torch.tensor(rows, device=device, dtype=torch.float32)


def _build_cache(env, asset: Entity, keyframes, joint_pos_range) -> dict:
    """Precompute, once, the nominal root pose + joint vector of every reference
    pose, and the per-joint perturbation range. Expansion of the regex joint
    dicts uses mjlab's own resolve_expr, so it matches the normal reset exactly.
    """
    names = asset.joint_names
    num_joints = len(names)
    device = env.device

    roots, joints = [], []
    for kf in keyframes:
        pos = tuple(getattr(kf, "pos", None) or (0.0, 0.0, 0.0))
        rot = tuple(getattr(kf, "rot", None) or (1.0, 0.0, 0.0, 0.0))  # (w,x,y,z)
        roots.append(list(pos) + list(rot))
        joint_vec = resolve_expr(getattr(kf, "joint_pos", {}) or {}, names, 0.0)
        joints.append(list(joint_vec))

    # Per-joint perturbation range (num_joints, 2); 0 for joints not in any group.
    pert = torch.zeros((num_joints, 2), device=device, dtype=torch.float32)
    for pattern, (lo, hi) in (joint_pos_range or {}).items():
        ids, _ = asset.find_joints(pattern)
        for i in ids:
            pert[i, 0], pert[i, 1] = float(lo), float(hi)

    return {
        "root": torch.tensor(roots, device=device, dtype=torch.float32),     # (P,7)
        "joints": torch.tensor(joints, device=device, dtype=torch.float32),  # (P,J)
        "pert": pert,                                                         # (J,2)
        "num_joints": num_joints,
    }


def reset_to_random_keyframe(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    keyframes: Sequence,
    joint_pos_range: dict[str, tuple[float, float]] | None = None,
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset each env to a randomly chosen reference pose + perturbation.

    Args:
      keyframes: sequence of EntityCfg.InitialStateCfg (the reference poses).
      joint_pos_range: {joint_name_regex: (lo, hi)} added to nominal joint pos.
      pose_range: {x,y,z,roll,pitch,yaw: (lo, hi)} added to the root pose.
      velocity_range: {x,y,z,roll,pitch,yaw: (lo, hi)} root velocity (nominal 0).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    asset: Entity = env.scene[asset_cfg.name]

    cache = getattr(env, _CACHE_ATTR, None)
    if cache is None:
        if len(keyframes) == 0:
            raise ValueError(
                "reset_to_random_keyframe: 'keyframes' is empty. Set it in the "
                "robot-specific config (e.g. (SUPINE_KEYFRAME, PRONE_KEYFRAME))."
            )
        cache = _build_cache(env, asset, keyframes, joint_pos_range)
        setattr(env, _CACHE_ATTR, cache)

    n = len(env_ids)
    num_poses = cache["root"].shape[0]
    num_joints = cache["num_joints"]
    device = env.device

    # 1) Pick one reference pose per environment.
    sel = torch.randint(0, num_poses, (n,), device=device)

    # 2) Root pose = reference + perturbation + per-env grid origin.
    root = cache["root"][sel]  # (n,7)
    psamp = sample_uniform(*_range_tensor(pose_range, device).unbind(dim=1), (n, 6), device)
    positions = root[:, 0:3] + psamp[:, 0:3] + env.scene.env_origins[env_ids]
    quat_delta = quat_from_euler_xyz(psamp[:, 3], psamp[:, 4], psamp[:, 5])
    orientations = quat_mul(root[:, 3:7], quat_delta)
    asset.write_root_link_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )

    # 3) Root velocity = perturbation around 0 (so the robot doesn't always start at rest).
    vsamp = sample_uniform(*_range_tensor(velocity_range, device).unbind(dim=1), (n, 6), device)
    asset.write_root_link_velocity_to_sim(vsamp, env_ids=env_ids)

    # 4) Joints = reference + per-joint perturbation, CLAMPED to soft limits.
    jp = cache["joints"][sel].clone()  # (n,J)
    pert = cache["pert"]
    jp += sample_uniform(pert[:, 0], pert[:, 1], (n, num_joints), device)
    limits = asset.data.soft_joint_pos_limits[env_ids]  # (n,J,2)
    jp = jp.clamp(limits[..., 0], limits[..., 1])
    jv = torch.zeros((n, num_joints), device=device)
    joint_ids = torch.arange(num_joints, device=device)
    asset.write_joint_state_to_sim(jp, jv, env_ids=env_ids, joint_ids=joint_ids)


__all__ = ["reset_to_random_keyframe", "reset_episode_state"]

# ---------------------------------------------------------------------------
# Per-env episode progress state
#
# Shared by:  rewards.height_progress, rewards.stable_success_hold
#             terminations.standing_fall_timeout
#
# Lifecycle:
#   - Lazily initialised on first reward/termination call via get_episode_state().
#   - Reset for terminated envs by reset_episode_state (mode="reset" event).
#
# Step order in mjlab:  terminations → rewards → reset-events → observations.
# So terminations read state written by the *previous* step's rewards (1-step
# lag on ever_stood), which is intentional and harmless.
# ---------------------------------------------------------------------------

_EPISODE_STATE_ATTR = "_episode_progress"


def get_episode_state(env: "ManagerBasedRlEnv", asset: "Entity") -> dict:
    """Return the shared per-env progress state, initialising it if needed."""
    if hasattr(env, _EPISODE_STATE_ATTR):
        return getattr(env, _EPISODE_STATE_ATTR)
    n, device = env.num_envs, env.device
    state = {
        # Height at the previous control step (for height_progress reward).
        "prev_height": asset.data.root_link_pos_w[:, 2].clone(),
        # Consecutive steps with pelvis above the standing threshold.
        "standing_counter": torch.zeros(n, device=device, dtype=torch.long),
        # True once the robot has been above the standing threshold in this episode.
        "ever_stood": torch.zeros(n, device=device, dtype=torch.bool),
        # Consecutive steps below the threshold after ever_stood became True.
        "fall_counter": torch.zeros(n, device=device, dtype=torch.long),
    }
    setattr(env, _EPISODE_STATE_ATTR, state)
    return state


def reset_episode_state(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset per-env episode progress state (mode='reset' event).

    Called automatically by the event manager for every environment that starts
    a new episode, so counters and prev_height are always episode-local.
    """
    if not hasattr(env, _EPISODE_STATE_ATTR):
        return  # not yet initialised; get_episode_state() handles first call
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset: Entity = env.scene[asset_cfg.name]
    state = getattr(env, _EPISODE_STATE_ATTR)
    state["prev_height"][env_ids] = asset.data.root_link_pos_w[env_ids, 2]
    state["standing_counter"][env_ids] = 0
    state["ever_stood"][env_ids] = False
    state["fall_counter"][env_ids] = 0
