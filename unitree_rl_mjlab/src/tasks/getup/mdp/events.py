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


def gate_settling_rewards(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
    settle_steps: int = 0,
) -> None:
    """Zero ALL reward signals for envs still inside the settling window.

    Registered as a ``mode="step"`` event. During the first ``settle_steps`` env-steps
    of each episode the policy does NOT control the robot: the action term PD-holds the
    spawn pose while MuJoCo resolves the landing contacts (see
    LowPassJointPositionAction.settle_steps / settle_zero_velocity). Any reward accrued
    in that window is uncontrollable by the policy, so letting it enter the training
    signal (or the logged returns) teaches the value function to expect "free" reward it
    can never influence — corrupting the advantages of the steps the policy DOES control.

    This is the single global gate (rather than per-term masking): it runs AFTER
    ``RewardManager.compute()`` (env.step ~line 388) but BEFORE the RL runner reads the
    reward tensors, so one zeroing propagates to every consumer at once:
      * ``_step_reward``  — per-term rate consumed by MultiCriticPPO._grouped_rewards and
                            the reward_group/* logging metrics,
      * ``_reward_buf``   — the scalar reward returned to the runner (bookkeeping/logging),
      * ``_episode_sums`` — the Episode_Reward/* per-term accumulators (contribution undone).

    Boundary (see actions.py): the action term holds while ``episode_length_buf <
    settle_steps`` at process-action time (episode_length_buf pre-increment), so the last
    held step computes its reward at ``episode_length_buf == settle_steps``. At step-event
    time episode_length_buf is post-increment, hence the settling window is exactly
    ``episode_length_buf <= settle_steps``; rewards flow from ``settle_steps + 1`` on (the
    first policy-controlled step).

    ``reset_buf`` guard: ``_reset_idx`` runs BEFORE step events, so an env that terminated
    THIS step already has ``episode_length_buf == 0`` here, yet its just-computed reward
    belongs to the previous episode's final (controlled) step and must be kept. Excluding
    ``reset_buf`` envs protects that terminating-step reward from being wrongly zeroed.
    No-op when ``settle_steps <= 0``.
    """
    del env_ids  # step events act on all envs; we select by the per-env counter.
    if settle_steps <= 0:
        return
    rm = env.reward_manager
    settling = env.episode_length_buf <= settle_steps  # [B]
    # Keep the terminating-step reward of any env that reset THIS step (buf just set to 0).
    settling = settling & (~env.reset_buf.bool())
    if not bool(settling.any()):
        return
    scale = env.step_dt if getattr(rm, "_scale_by_dt", True) else 1.0
    # Undo this step's contribution to the per-term episode sums (they were accumulated
    # inside RewardManager.compute using the dt-scaled value == _step_reward * scale).
    for idx, name in enumerate(rm._term_names):
        rm._episode_sums[name][settling] -= rm._step_reward[settling, idx] * scale
    # Zero the training/logging reward tensors for the settling envs.
    rm._step_reward[settling, :] = 0.0
    rm._reward_buf[settling] = 0.0


__all__ = [
    "reset_to_random_keyframe",
    "reset_episode_state",
    "settle_zero_velocity",
    "gate_settling_rewards",
    "AssistanceCurriculum",
    "BetaRescalerCurriculum",
]


# ---------------------------------------------------------------------------
# Feet-planted qualifier — shared by the assistance-force and beta curricula
#
# Both curricula must only award "success" (and therefore withdraw their help)
# when the robot genuinely stood on its legs, NOT when a torso-lean / arch-back
# transiently spiked the pelvis/head height without the feet bearing weight.
# These two helpers resolve the feet contact sensor + foot sites (once, in
# __init__) and compute the per-step "both feet planted" mask (mirror of
# mdp.rewards.stand_on_feet's contact+height logic), so the two curricula share
# identical qualifier logic and cannot drift apart.
# ---------------------------------------------------------------------------


def _resolve_feet_planted_cfg(asset: "Entity", params: dict, device):
    """Resolve (feet_sensor_name, foot_site_ids, foot_height_threshold) from cfg.

    ``foot_site_names`` are resolved to ids IN ORDER so ``site_pos_w[:, ids, 2]``
    lines up with the sensor's per-foot ``found`` columns.
    """
    feet_sensor_name = str(params["feet_sensor_name"])
    foot_site_names = tuple(params["foot_site_names"])
    site_ids, _ = asset.find_sites(foot_site_names, preserve_order=True)
    foot_site_ids = torch.as_tensor(site_ids, device=device, dtype=torch.long)
    foot_height_threshold = float(params.get("foot_height_threshold", 0.1))
    return feet_sensor_name, foot_site_ids, foot_height_threshold


def _both_feet_planted(
    env: "ManagerBasedRlEnv",
    asset: "Entity",
    feet_sensor_name: str,
    foot_site_ids: torch.Tensor,
    foot_height_threshold: float,
    num_envs: int,
    device,
) -> torch.Tensor:
    """[B] bool: both feet in ground contact AND not elevated (foot z < threshold).

    Returns all-False before the contact sensor is populated (found is None).
    """
    contact_sensor = env.scene[feet_sensor_name]
    found = contact_sensor.data.found
    if found is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    in_contact = found > 0  # [B, N_feet]
    foot_heights = asset.data.site_pos_w[:, foot_site_ids, 2]  # [B, N_feet]
    foot_low = foot_heights < foot_height_threshold
    return (in_contact & foot_low).all(dim=1)  # [B]


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
        # Fix 1 — stable-hold decay gate. The assist only anneals when an env
        # achieves a GENUINE stand: pelvis above success_height WHILE both feet
        # are planted, held for `hold_steps` consecutive control steps (matches
        # the stable_hold success metric, n_hold=50). This replaces the old
        # partial-credit-on-peak-height gate that retired the support on one-frame
        # height touches that immediately toppled — the false positive that let the
        # previous run anneal the lift to ~0 while stable_hold never left ~0.
        self._hold_steps = int(p.get("hold_steps", 50))
        # Fix 2 — reversible crutch. An env that fails to hold a stand for
        # `ramp_up_after` consecutive episodes has its support bumped back up by
        # `ramp_up_step` N (clamped to the initial force), so a regressed env
        # recovers instead of collapsing to the zero floor forever. Turns the old
        # monotonic ratchet into a servo that tracks each env's true competence.
        self._ramp_up_after = int(p.get("ramp_up_after_failures", 20))
        self._ramp_up_step = float(p.get("ramp_up_step_n", 5.0))
        # Rate limiter: after an env decays its assist, it must wait this many
        # episodes before another decay is allowed. This breaks the positive
        # feedback loop where higher success density speeds up global annealing.
        self._decay_cooldown = int(p.get("decay_cooldown_episodes", 0))

        # Feet-planted qualifier for the decay credit. Only pelvis height reached
        # while BOTH feet are genuinely planted (in ground contact AND not
        # elevated) counts toward peak height. Without this, a torso-lean /
        # arch-back that transiently spikes pelvis height WITHOUT the legs bearing
        # weight retires the assist force — the failure mode that let the old
        # policy anneal the 120 N lift away while never learning to stand on its
        # legs. See _both_feet_planted (shared with BetaRescalerCurriculum).
        (
            self._feet_sensor_name,
            self._foot_site_ids,
            self._foot_height_threshold,
        ) = _resolve_feet_planted_cfg(self._asset, p, self._device)

        # Per-env state.
        self._force = torch.full(
            (self._num_envs,), self._initial_force, device=self._device
        )
        # Consecutive planted steps above success_height this episode (Fix 1).
        self._hold_counter = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        # Latched True once this episode reached a genuine hold_steps-long stand.
        self._stable_held = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.bool
        )
        # Consecutive FAILED episodes (no stable hold) since the last hold / bump
        # (Fix 2: drives the support re-ramp).
        self._fail_streak = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        # Per-env remaining cooldown episodes before assist is allowed to decay again.
        # This persists across resets by design; reset() decrements it once per episode.
        self._decay_cooldown_counter = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        # Reusable zero wrench buffer (N, 1 body, 3).
        self._zeros = torch.zeros(self._num_envs, 1, 3, device=self._device)

        # Expose the per-env force tensor for the metrics logger (read-only use).
        setattr(env, _ASSIST_FORCE_ATTR, self._force)

    def __call__(self, env: "ManagerBasedRlEnv", env_ids=None, **kwargs) -> None:
        del env_ids, kwargs  # step events always act on all envs
        # Track a GENUINE stable hold (drives the decay at reset, Fix 1): the
        # pelvis must be above success_height WHILE both feet are planted, and
        # stay there for hold_steps consecutive steps. A torso-lean that spikes
        # the pelvis without loading the legs never satisfies the planted check,
        # and a one-frame touch that topples resets the counter — so only a real,
        # held stand latches _stable_held and retires the assist.
        h = self._asset.data.root_link_pos_w[:, 2]
        both_planted = _both_feet_planted(
            env, self._asset, self._feet_sensor_name, self._foot_site_ids,
            self._foot_height_threshold, self._num_envs, self._device,
        )
        holding = (h >= self._success_height) & both_planted
        self._hold_counter = torch.where(
            holding, self._hold_counter + 1, torch.zeros_like(self._hold_counter)
        )
        self._stable_held |= self._hold_counter >= self._hold_steps

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
        held = self._stable_held[env_ids]
        cooldown = self._decay_cooldown_counter[env_ids]
        can_decay = cooldown == 0
        decayed = held & can_decay
        # Fix 1 — decay only on a genuine held stand (binary credit). Support is
        # never retired on a one-frame height touch that immediately topples, so
        # the crutch can no longer outrun real competence (the failure that
        # cratered the previous run's force to ~0 while stable_hold stayed ~0).
        self._force[env_ids] = torch.where(
            decayed, self._force[env_ids] - self._decay, self._force[env_ids]
        )
        cooldown = torch.where(
            decayed,
            torch.full_like(cooldown, self._decay_cooldown),
            torch.clamp(cooldown - 1, min=0),
        )
        self._decay_cooldown_counter[env_ids] = cooldown
        # Fix 2 — reversible crutch. Track consecutive failed episodes; after
        # ramp_up_after of them, bump the support back up so a regressed env
        # recovers instead of sitting at the zero floor forever. The streak
        # resets on any held stand or on a bump, giving a servo that tracks each
        # env's competence rather than a one-way ratchet.
        streak = self._fail_streak[env_ids]
        streak = torch.where(held, torch.zeros_like(streak), streak + 1)
        ramp = streak >= self._ramp_up_after
        self._force[env_ids] = torch.where(
            ramp, self._force[env_ids] + self._ramp_up_step, self._force[env_ids]
        )
        self._fail_streak[env_ids] = torch.where(
            ramp, torch.zeros_like(streak), streak
        )
        # Clamp to [force_min, initial_force] (re-ramp never exceeds the start).
        self._force[env_ids] = torch.clamp(
            self._force[env_ids], min=self._force_min, max=self._initial_force
        )
        # Clear episode hold state + any residual applied wrench for reset envs.
        self._hold_counter[env_ids] = 0
        self._stable_held[env_ids] = False
        n = len(env_ids) if not isinstance(env_ids, slice) else self._num_envs
        zeros = torch.zeros(n, 1, 3, device=self._device)
        self._asset.write_external_wrench_to_sim(
            zeros, zeros, env_ids=env_ids, body_ids=[self._body_id]
        )

# ---------------------------------------------------------------------------
# Action-rescaler (beta) curriculum — HoST-style per-env action-scale annealing
# ---------------------------------------------------------------------------

_BETA_RESCALER_ATTR = "_beta_rescaler"  # per-env beta, read by the action term,
                                        # the beta observation and the beta metric.
_BETA_CURRICULUM_ATTR = "_beta_curriculum"  # the curriculum instance itself, read
                                            # by the breaker metrics (success EMA
                                            # + paused flag).


class BetaRescalerCurriculum:
    """Per-env action-rescaler beta that anneals 1.0 -> 0.25 on success (HoST).

    The PD target is  p^d = q_default + beta * a  (the action term reads this
    per-env beta). beta starts at `initial_beta` = 1.0, giving the policy large
    action authority while it is still flailing on the floor, and is decremented
    by `decrement` (0.02) every time an env's episode ends having achieved a
    GENUINE stand — the standing head height (`success_head_height`) reached
    WHILE both feet are planted — down to a floor of `beta_min` (0.25 — exactly
    the fixed value HoST's no-curriculum ablation uses). The policy OBSERVES its
    env's beta, so it adapts as its action scale shrinks.

    Feet-planted qualifier (matches AssistanceCurriculum): the success flag only
    latches when the head-height threshold is met AT THE SAME STEP as both feet
    being planted. Without it, a torso-lean / arch-back that transiently lifts the
    head past the threshold — without the legs bearing weight — would permanently
    shrink the policy's action authority before it has actually learned to stand.
    This is what let the previous run's beta collapse (1.0 -> 0.76) while
    success/stable_hold stayed ~0: authority was withdrawn on false positives.

    This is the SAME per-env success-driven mechanism as AssistanceCurriculum
    (the vertical support force): the help/authority fades per env, exactly as
    fast as that env learns to stand. It is NOT a global iteration ramp, so it is
    invariant to the number of parallel envs and the episode length.

    Registered with ``mode="step"``: ``__call__`` runs every control step (latch
    whether this episode has genuinely stood) and ``reset`` is called for envs
    starting a new episode (decrement beta for the ones that succeeded, clear the
    per-episode success flag).
    """

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        p = cfg.params
        self._asset: Entity = env.scene[p["asset_cfg"].name]
        body_name: str = p["body_name"]
        ids, _ = self._asset.find_bodies(body_name)
        assert len(ids) == 1, (
            f"BetaRescalerCurriculum: body '{body_name}' matched {len(ids)} bodies; "
            "expected exactly one (e.g. 'torso_link')."
        )
        self._body_id = int(ids[0])
        self._num_envs = env.num_envs
        self._device = env.device

        self._initial_beta = float(p["initial_beta"])
        self._decrement = float(p["decrement"])
        self._beta_min = float(p["beta_min"])
        self._success_head_height = float(p["success_head_height"])
        # Fix 1 — stable-hold decay gate (mirrors AssistanceCurriculum): beta only
        # decays after a GENUINE held stand (head height above threshold WHILE both
        # feet planted, for hold_steps consecutive steps), not a one-frame spike.
        self._hold_steps = int(p.get("hold_steps", 50))
        # Fix 2 — reversible crutch: restore action authority (bump beta up) after
        # ramp_up_after consecutive failed episodes, so a regressed env is not
        # locked at the low-authority floor forever.
        self._ramp_up_after = int(p.get("ramp_up_after_failures", 20))
        self._ramp_up_step = float(p.get("ramp_up_step", 0.02))
        # Fix 3 — decay-cooldown rate limiter (mirrors AssistanceCurriculum): after
        # an env decrements beta, it must wait this many episodes before it may
        # decrement again. Without it, at large num_envs a synchronized wave of
        # successes (e.g. ~350 of 4096 envs holding a stand per iteration once the
        # policy is good) drops the POPULATION-mean beta far faster than gradient
        # descent can re-adapt the policy to the shrinking action window — a runaway
        # positive-feedback loop that cratered the v10_beta_home_anchor run once
        # beta fell below ~0.87. The per-env ramp-up (Fix 2) is far too slow to
        # counteract a mass simultaneous decrement. Same fix, same value as the
        # assist-force curriculum's decay_cooldown_episodes.
        self._decay_cooldown = int(p.get("decay_cooldown_episodes", 0))
        # Fix 4 — population circuit breaker. All per-env safeguards above are
        # LOCAL: the cooldown paces one env's decay, the re-ramp only fires after
        # one env personally fails ramp_up_after episodes in a row. But all envs
        # share one policy, so successes are correlated — a synchronized wave can
        # drop the population-mean beta faster than PPO re-adapts, while no single
        # env's fail-streak is long enough to trigger its re-ramp (the slow bleed
        # seen in v11_beta_cooldown: stable_hold 0.80 -> 0.49 as beta 0.90 -> 0.63).
        # This breaker closes the missing population-level feedback loop: track an
        # EMA of the held-stand fraction over resetting envs; when it drops below
        # pause_below, HALT all decay globally (and nudge beta back up a little
        # each episode) until the EMA recovers above resume_above. The gap between
        # the two thresholds (hysteresis) prevents rapid on/off flapping.
        self._ema_alpha = float(p.get("success_ema_alpha", 0.5))
        self._pause_below = float(p.get("pause_below", 0.65))
        self._resume_above = float(p.get("resume_above", 0.75))
        self._recovery_step = float(p.get("recovery_step", 0.0025))
        self._success_ema = 1.0
        self._paused = False

        # Feet-planted qualifier (see _both_feet_planted, shared with
        # AssistanceCurriculum): beta only decays on a GENUINE stand, not a
        # torso-lean head-height spike.
        (
            self._feet_sensor_name,
            self._foot_site_ids,
            self._foot_height_threshold,
        ) = _resolve_feet_planted_cfg(self._asset, p, self._device)

        # Per-env state.
        self._beta = torch.full(
            (self._num_envs,), self._initial_beta, device=self._device
        )
        # Consecutive planted steps above the success head height this episode.
        self._hold_counter = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        # Latched True once this episode reached a genuine hold_steps-long stand.
        self._stable_held = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.bool
        )
        # Consecutive FAILED episodes since the last hold / bump (Fix 2 re-ramp).
        self._fail_streak = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        # Per-env remaining cooldown episodes before beta is allowed to decay again
        # (Fix 3). Persists across resets by design; reset() decrements it once per
        # episode. Matches AssistanceCurriculum._decay_cooldown_counter.
        self._decay_cooldown_counter = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )

        # Expose the per-env beta tensor for the action term / obs / metric, and
        # the instance itself for the circuit-breaker metrics (EMA + paused flag).
        setattr(env, _BETA_RESCALER_ATTR, self._beta)
        setattr(env, _BETA_CURRICULUM_ATTR, self)

    def __call__(self, env: "ManagerBasedRlEnv", env_ids=None, **kwargs) -> None:
        del env_ids, kwargs  # step events always act on all envs
        # Genuine HELD stand = head-height proxy (torso_link z) at/above threshold
        # WHILE both feet planted, sustained for hold_steps consecutive steps. A
        # one-frame spike that topples resets the counter, so beta only decays on a
        # real hold — not the transient touch that shrank authority prematurely.
        h_head = self._asset.data.body_link_pos_w[:, self._body_id, 2]
        both_planted = _both_feet_planted(
            env, self._asset, self._feet_sensor_name, self._foot_site_ids,
            self._foot_height_threshold, self._num_envs, self._device,
        )
        holding = (h_head >= self._success_head_height) & both_planted
        self._hold_counter = torch.where(
            holding, self._hold_counter + 1, torch.zeros_like(self._hold_counter)
        )
        self._stable_held |= self._hold_counter >= self._hold_steps

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        held = self._stable_held[env_ids]
        # Fix 4 — population circuit breaker. Fold this reset batch's held
        # fraction into the population EMA. The blend factor is scaled by the
        # batch's share of the population so the EMA's effective time constant
        # (~1/ema_alpha population-generations of episodes) does not depend on
        # how many envs happen to reset in one call.
        n_batch = int(held.numel())
        if n_batch > 0:
            k = min(1.0, self._ema_alpha * n_batch / self._num_envs)
            self._success_ema = (
                (1.0 - k) * self._success_ema + k * float(held.float().mean())
            )
        if self._paused:
            if self._success_ema > self._resume_above:
                self._paused = False
        elif self._success_ema < self._pause_below:
            self._paused = True
        cooldown = self._decay_cooldown_counter[env_ids]
        can_decay = cooldown == 0
        decayed = held & can_decay
        if self._paused:
            # Breaker engaged: the population is not keeping up with the current
            # beta — halt ALL decay (every env, regardless of its own cooldown)
            # and actively walk beta back up a little each episode until the
            # population success EMA recovers above resume_above.
            decayed = torch.zeros_like(decayed)
            self._beta[env_ids] = self._beta[env_ids] + self._recovery_step
        # Fix 1 — decrement beta only after a genuine held stand (binary credit),
        # not a one-frame head-height touch, so action authority is never withdrawn
        # on a false positive. Fix 3 — and only if this env's decay cooldown has
        # elapsed, rate-limiting how fast the population-mean beta can fall.
        self._beta[env_ids] = torch.where(
            decayed, self._beta[env_ids] - self._decrement, self._beta[env_ids]
        )
        cooldown = torch.where(
            decayed,
            torch.full_like(cooldown, self._decay_cooldown),
            torch.clamp(cooldown - 1, min=0),
        )
        self._decay_cooldown_counter[env_ids] = cooldown
        # Fix 2 — reversible crutch: after ramp_up_after consecutive failed
        # episodes, bump beta back up (restore authority); streak resets on any
        # held stand or bump. Symmetric with the assist-force re-ramp.
        streak = self._fail_streak[env_ids]
        streak = torch.where(held, torch.zeros_like(streak), streak + 1)
        ramp = streak >= self._ramp_up_after
        self._beta[env_ids] = torch.where(
            ramp, self._beta[env_ids] + self._ramp_up_step, self._beta[env_ids]
        )
        self._fail_streak[env_ids] = torch.where(
            ramp, torch.zeros_like(streak), streak
        )
        # Clamp to [beta_min, initial_beta] (re-ramp never exceeds the start).
        self._beta[env_ids] = torch.clamp(
            self._beta[env_ids], min=self._beta_min, max=self._initial_beta
        )
        # Clear per-episode hold state for the reset envs.
        self._hold_counter[env_ids] = 0
        self._stable_held[env_ids] = False


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
        # Diagnostic flags written each step by the stable_success_hold reward and
        # read by metrics.hold_stillness_fraction (no training effect): whether the
        # robot is currently above standing height, and whether it is ALSO holding
        # still (height + low base/joint velocity). Their ratio shows how much gross
        # wobble is costing the stillness-gated hold. Initialised False so the metric
        # is safe on step 0 before the reward has run.
        "stand_height_ok": torch.zeros(n, device=device, dtype=torch.bool),
        "stand_still_ok": torch.zeros(n, device=device, dtype=torch.bool),
        # True once the pelvis has reached the HOME height (~0.72 m) in this episode.
        # Latches the hand-contact penalty OFF for the rest of the episode so a later
        # fall does not dump a huge contact penalty on an already-successful stand.
        "ever_reached_home": torch.zeros(n, device=device, dtype=torch.bool),
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
    state["stand_height_ok"][env_ids] = False
    state["stand_still_ok"][env_ids] = False
    state["ever_reached_home"][env_ids] = False
    state["fall_counter"][env_ids] = 0
    state["best_height"][env_ids] = asset.data.root_link_pos_w[env_ids, 2]
    state["stall_counter"][env_ids] = 0
