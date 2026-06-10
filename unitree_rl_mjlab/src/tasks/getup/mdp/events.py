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
    fall_height: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset each env to a randomly chosen reference pose + perturbation.

    Args:
      keyframes: sequence of EntityCfg.InitialStateCfg (the reference poses).
      joint_pos_range: {joint_name_regex: (lo, hi)} added to nominal joint pos.
      pose_range: {x,y,z,roll,pitch,yaw: (lo, hi)} added to the root pose.
      velocity_range: {x,y,z,roll,pitch,yaw: (lo, hi)} root velocity (nominal 0).
      fall_height: deterministic height (m) added to the spawn z of EVERY env, so
        the robot is dropped from above its resting pose and falls to the ground
        each reset (HoST-style: HoST spawns the G1 at pelvis z=0.5 m — ~0.4 m above
        its settled height — at rest, and lets it fall). Set to 0 to spawn already
        settled. The fall provides reset diversity; HoST adds NO random root
        velocity (the tumble comes from the drop), so keep velocity_range at 0.
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
    # HoST-style drop: lift every spawn by fall_height so the robot falls to the
    # ground each reset (the tumble gives reset diversity without random velocity).
    if fall_height:
        positions[:, 2] += fall_height
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


def settle_zero_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
    settle_steps: int = 0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Zero residual root + joint velocities at the END of the settling window.

    Registered as a ``mode="step"`` event. The settling phase (see
    LowPassJointPositionAction.settle_steps) holds the policy for the first
    ``settle_steps`` env-steps while the robot drops onto the floor; this removes the
    leftover landing velocity (bounce) on the single step each env exits the window
    (``episode_length_buf == settle_steps``), so the policy takes over from a clean,
    at-rest state. No-op when ``settle_steps <= 0``.
    """
    del env_ids  # step events act on all envs; we select by the per-env counter.
    if settle_steps <= 0:
        return
    at_end = env.episode_length_buf == settle_steps  # [B]
    ids = at_end.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
        return
    asset: Entity = env.scene[asset_cfg.name]
    asset.write_root_link_velocity_to_sim(
        torch.zeros(len(ids), 6, device=env.device), env_ids=ids
    )
    q = asset.data.joint_pos[ids]
    asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)


__all__ = [
    "reset_to_random_keyframe",
    "reset_episode_state",
    "settle_zero_velocity",
    "AssistanceCurriculum",
]


# ---------------------------------------------------------------------------
# Assistance curriculum (HoST-style decaying support force)
# ---------------------------------------------------------------------------

_ASSIST_FORCE_ATTR = "_assistance_force"  # exposed on env for the metrics logger


class AssistanceCurriculum:
    """Decaying upward support force on the torso that bootstraps get-up learning.

    Early in training the robot almost never stands by chance, so the policy
    gradient is extremely sparse. We apply an UPWARD, world-frame force to the
    torso — like a hand under the chest — that makes rising easier. Each env
    carries its OWN force magnitude. When an env succeeds in an episode (its
    pelvis reached `success_height`), the force is decayed at that env's next
    reset; once it hits `force_min` the env is unassisted. The help therefore
    anneals away per-env, exactly as fast as each env learns to stand on its own
    (this is the HoST mechanism, adapted to mjlab/MuJoCo external wrenches).

    The force is suppressed for the first `unactuated_steps` of every episode so
    the robot first settles onto the ground from its (slightly floating) init
    pose, avoiding a launch at t=0.

    Registered with ``mode="step"``: ``__call__`` runs every control step
    (apply force + track the episode's peak pelvis height) and ``reset`` is
    called by the event manager for the envs starting a new episode (decay the
    force for the ones that succeeded, clear state + residual wrench).

    Wrench is written via ``write_external_wrench_to_sim`` (sets ``xfrc_applied``
    in WORLD frame), so an upward assist is simply ``+Z`` — no rotation needed.
    """

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        p = cfg.params
        self._asset: Entity = env.scene[p["asset_cfg"].name]
        body_name: str = p["body_name"]
        ids, _ = self._asset.find_bodies(body_name)
        assert len(ids) == 1, (
            f"AssistanceCurriculum: body '{body_name}' matched {len(ids)} bodies; "
            "expected exactly one (e.g. 'torso_link')."
        )
        self._body_id = int(ids[0])
        self._num_envs = env.num_envs
        self._device = env.device

        self._initial_force = float(p["initial_force_n"])
        self._decay = float(p["force_decay_per_success"])
        self._force_min = float(p.get("force_min", 0.0))
        self._success_height = float(p["success_height"])
        self._unactuated_steps = int(p.get("unactuated_steps", 0))

        # Per-env state.
        self._force = torch.full(
            (self._num_envs,), self._initial_force, device=self._device
        )
        self._peak_height = torch.zeros(self._num_envs, device=self._device)
        # Reusable zero wrench buffer (N, 1 body, 3).
        self._zeros = torch.zeros(self._num_envs, 1, 3, device=self._device)

        # Expose the per-env force tensor for the metrics logger (read-only use).
        setattr(env, _ASSIST_FORCE_ATTR, self._force)

    def __call__(self, env: "ManagerBasedRlEnv", env_ids=None, **kwargs) -> None:
        del env_ids, kwargs  # step events always act on all envs
        # Track the episode's peak pelvis height (drives the decay at reset).
        h = self._asset.data.root_link_pos_w[:, 2]
        torch.maximum(self._peak_height, h, out=self._peak_height)

        # Upward world-frame force; suppressed during the initial settle phase.
        active = env.episode_length_buf >= self._unactuated_steps
        forces = self._zeros.clone()
        forces[:, 0, 2] = self._force * active.to(self._force.dtype)
        self._asset.write_external_wrench_to_sim(
            forces, self._zeros, body_ids=[self._body_id]
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        # Decay the force for envs that reached standing height this episode.
        succeeded = self._peak_height[env_ids] >= self._success_height
        if succeeded.any():
            won = env_ids[succeeded]
            self._force[won] = torch.clamp(
                self._force[won] - self._decay, min=self._force_min
            )
        # Clear episode peak + any residual applied wrench for the reset envs.
        self._peak_height[env_ids] = 0.0
        n = len(env_ids) if not isinstance(env_ids, slice) else self._num_envs
        zeros = torch.zeros(n, 1, 3, device=self._device)
        self._asset.write_external_wrench_to_sim(
            zeros, zeros, env_ids=env_ids, body_ids=[self._body_id]
        )

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
        # Projected-gravity x at the previous step (for prone_recovery progress:
        # pg_x = +1 prone / -1 supine, so a DECREASE means rolling off the face).
        "prev_pg_x": asset.data.projected_gravity_b[:, 0].clone(),
        # Consecutive steps with pelvis above the standing threshold.
        "standing_counter": torch.zeros(n, device=device, dtype=torch.long),
        # True once the robot has been above the standing threshold in this episode.
        "ever_stood": torch.zeros(n, device=device, dtype=torch.bool),
        # Consecutive steps below the threshold after ever_stood became True.
        "fall_counter": torch.zeros(n, device=device, dtype=torch.long),
        # Best pelvis height reached this episode (for the no-progress stall timeout).
        "best_height": asset.data.root_link_pos_w[:, 2].clone(),
        # Consecutive steps since the last new height record (no-progress stall timeout).
        "stall_counter": torch.zeros(n, device=device, dtype=torch.long),
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
    state["prev_pg_x"][env_ids] = asset.data.projected_gravity_b[env_ids, 0]
    state["standing_counter"][env_ids] = 0
    state["ever_stood"][env_ids] = False
    state["fall_counter"][env_ids] = 0
    state["best_height"][env_ids] = asset.data.root_link_pos_w[env_ids, 2]
    state["stall_counter"][env_ids] = 0
