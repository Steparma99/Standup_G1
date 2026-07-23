"""Custom action term for the get-up task: low-pass filtered joint position target.

Same residual-on-default scheme as mjlab's JointPositionAction (target = default
pose + scale * action, fed to the PD actuators), with an extra exponential
low-pass (EMA) filter on the commanded target:

    q_cmd_t = alpha * (default + scale * a_t) + (1 - alpha) * q_cmd_{t-1}

The EMA acts as a physical barrier on the command (spec: "action low pass"),
discouraging the violent, non-deployable motions RL tends to discover. The filter
state is reset to the default pose at every episode reset. alpha in (0, 1];
alpha = 1 disables the filter (recovers plain JointPositionAction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import (
    JointPositionAction,
    JointPositionActionCfg,
)
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.utils.string import resolve_expr

from .events import _BETA_RESCALER_ATTR

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "HandHoldAction",
    "HandHoldActionCfg",
    "LowPassJointPositionAction",
    "LowPassJointPositionActionCfg",
]


class LowPassJointPositionAction(JointPositionAction):
    """Joint position action with an EMA low-pass filter on the commanded target."""

    cfg: "LowPassJointPositionActionCfg"

    def __init__(self, cfg: "LowPassJointPositionActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self._env = env
        self._alpha = float(cfg.alpha)
        self._settle_steps = int(cfg.settle_steps)
        # Slew-rate limit: max absolute change in the commanded target per control
        # step = max_rate [rad/s] * step_dt [s]. None disables it.
        self._max_target_delta = (
            None if cfg.max_rate is None else float(cfg.max_rate) * float(env.step_dt)
        )
        # Liftoff phase: reduced alpha / slew right after the settle window, held
        # for liftoff_steps then linearly blended back to the nominal values over
        # liftoff_ramp_steps. liftoff_steps = 0 disables the phase entirely.
        self._liftoff_steps = int(cfg.liftoff_steps)
        self._liftoff_ramp_steps = max(int(cfg.liftoff_ramp_steps), 1)
        self._liftoff_alpha = (
            self._alpha if cfg.liftoff_alpha is None else float(cfg.liftoff_alpha)
        )
        if cfg.liftoff_max_rate is None:
            self._liftoff_max_target_delta = self._max_target_delta
        else:
            assert self._max_target_delta is not None, (
                "LowPassJointPositionAction: liftoff_max_rate requires max_rate to "
                "be set (there is no nominal rate to ramp back to)."
            )
            self._liftoff_max_target_delta = float(cfg.liftoff_max_rate) * float(
                env.step_dt
            )
        if cfg.default_pos_override is not None:
            # Anchor the residual/beta scheme on an explicit pose instead of the
            # entity's spawn pose. mjlab derives default_joint_pos from
            # init_state.joint_pos, which for the get-up task is the SUPINE
            # (lying) keyframe — with the beta rescaler, p^d = anchor + beta*a,
            # so a decaying beta would squeeze the reachable targets toward
            # lying on the floor instead of toward standing. The override lets
            # the task anchor on the standing HOME pose (the paper's intent).
            vals = resolve_expr(
                cfg.default_pos_override, tuple(self._target_names), 0.0
            )
            default = (
                torch.tensor(vals, dtype=torch.float, device=self.device)
                .unsqueeze(0)
                .repeat(self.num_envs, 1)
            )
            self._offset = default.clone()
        else:
            default = self._entity.data.default_joint_pos[:, self._target_ids].clone()
        self._default_target = default
        # Filter state tracks the realizable, clamped command to avoid windup.
        self._filtered_target = default.clone()
        # Store the post-EMA / pre-clamp target for penalties and diagnostics.
        self._filtered_target_unclamped = default.clone()
        # Preserve raw policy output before and after defensive clipping.
        self._raw_policy_actions = torch.zeros_like(default)
        self._clipped_raw_actions = torch.zeros_like(default)
        self._needs_history_backfill = torch.ones(
            env.num_envs, device=env.device, dtype=torch.bool
        )

    @property
    def applied_target(self) -> torch.Tensor:
        """The (filtered) joint position target actually commanded to the PD [B, A]."""
        return self._processed_actions

    @property
    def filtered_target_unclamped(self) -> torch.Tensor:
        """The EMA target before joint-limit clamp [B, A]."""
        return self._filtered_target_unclamped

    @property
    def raw_policy_actions(self) -> torch.Tensor:
        """Raw policy output before defensive clamp [B, A]."""
        return self._raw_policy_actions

    @property
    def clipped_raw_actions(self) -> torch.Tensor:
        """Raw policy output after defensive clamp to [-1, 1] [B, A]."""
        return self._clipped_raw_actions

    def _clamp_to_joint_limits(self, target: torch.Tensor) -> torch.Tensor:
        limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
        return torch.clamp(target, limits[..., 0], limits[..., 1])

    def _read_beta(self) -> torch.Tensor | float:
        """Per-env action-rescaler beta as [B, 1] (broadcasts over joints).

        Published by BetaRescalerCurriculum (anneals 1.0 -> 0.25 on success).
        Falls back to a scalar 1.0 when the curriculum is disabled / not yet
        initialised, recovering the plain residual-on-default scheme.
        """
        beta = getattr(self._env, _BETA_RESCALER_ATTR, None)
        if beta is None:
            return 1.0
        return beta.unsqueeze(-1)  # [B, 1]

    def _set_if_present(
        self,
        obj: object,
        attr_name: str,
        value: torch.Tensor,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        if not hasattr(obj, attr_name):
            return
        attr = getattr(obj, attr_name)
        if not isinstance(attr, torch.Tensor):
            return
        if attr.shape != value.shape:
            return
        if env_ids is None:
            attr[:] = value
        else:
            attr[env_ids] = value[env_ids]

    def _backfill_history(self) -> None:
        if not bool(self._needs_history_backfill.any()):
            return
        mask = self._needs_history_backfill.unsqueeze(-1)
        current = self._clipped_raw_actions
        manager = self._env.action_manager

        # On the first post-reset step, make first/second-order differences zero.
        # The manager exposes action/prev_action/prev_prev_action as READ-ONLY
        # properties (backed by _action etc.), so mutate the returned tensor
        # in-place — same pattern as _set_if_present — instead of setattr().
        for attr_name in ("action", "prev_action", "prev_prev_action"):
            if not hasattr(manager, attr_name):
                continue
            attr = getattr(manager, attr_name)
            if isinstance(attr, torch.Tensor) and attr.shape == current.shape:
                attr[:] = torch.where(mask, current, attr)
        self._needs_history_backfill &= ~mask.squeeze(-1)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_policy_actions = actions.clone()
        self._clipped_raw_actions = torch.clamp(actions, -1.0, 1.0)

        # super() sets _processed_actions = scale * action + default (the desired target).
        # With cfg.scale = 1.0 (set per-robot for HoST) the residual is just the
        # clipped action a, so applying the per-env rescaler beta gives the HoST
        # PD target  p^d = q_default + beta * a  (beta anneals 1.0 -> 0.25). When the
        # beta curriculum is disabled, _read_beta() returns 1.0 and this reduces to
        # the plain residual-on-default target.
        super().process_actions(self._clipped_raw_actions)
        beta = self._read_beta()
        desired_target = self._default_target + beta * (
            self._processed_actions - self._default_target
        )
        # Liftoff phase: for the first liftoff_steps after the settle window use
        # the reduced liftoff_alpha / liftoff_max_rate (a deliberately slower
        # initial rise), then blend LINEARLY back to the nominal values over
        # liftoff_ramp_steps — a hard step back to the fast values would itself
        # be a jerk seam mid-rise. ramp_frac is per-env [B, 1]: 0 = full liftoff
        # values, 1 = nominal values. During settle steps the ramp value is
        # irrelevant (the settle override below wins).
        alpha: torch.Tensor | float = self._alpha
        max_target_delta: torch.Tensor | float | None = self._max_target_delta
        if self._liftoff_steps > 0:
            steps_since_settle = (
                self._env.episode_length_buf - self._settle_steps
            ).clamp(min=0)
            ramp_frac = (
                (steps_since_settle - self._liftoff_steps).float()
                / self._liftoff_ramp_steps
            ).clamp(0.0, 1.0).unsqueeze(-1)  # [B, 1]
            alpha = self._liftoff_alpha + ramp_frac * (
                self._alpha - self._liftoff_alpha
            )
            if self._max_target_delta is not None:
                max_target_delta = self._liftoff_max_target_delta + ramp_frac * (
                    self._max_target_delta - self._liftoff_max_target_delta
                )
        ema_target = (
            alpha * desired_target
            + (1.0 - alpha) * self._filtered_target
        )
        # Slew-rate limit (optional): bound the per-step change of the commanded
        # target to ±max_target_delta rad/joint relative to the previous realizable
        # command (self._filtered_target still holds the prior step's value here).
        # Applied after the EMA so it caps ABSOLUTE speed — which the EMA, being a
        # fraction-of-distance filter, cannot — and before the joint-limit clamp so
        # the anti-windup filter state stays on the realizable command.
        if max_target_delta is not None:
            delta = torch.clamp(
                ema_target - self._filtered_target,
                -max_target_delta,
                max_target_delta,
            )
            ema_target = self._filtered_target + delta
        self._filtered_target_unclamped = ema_target
        self._processed_actions = self._clamp_to_joint_limits(ema_target)
        # Anti-windup: keep the filter state on the realizable command.
        self._filtered_target = self._processed_actions.clone()

        # --- Settling phase: for the first `settle_steps` env-steps of each episode
        # the policy does NOT control. The PD target is held at the robot's CURRENT
        # measured joint pose, so the actuators only damp residual joint motion while
        # MuJoCo's integrator resolves the spawn contacts and the robot accommodates
        # onto the floor. The policy's action for these steps is ignored (kept short
        # so the on-policy bias is negligible; the impact penalties are also masked
        # over this window, see rewards._reset_ramp). episode_length_buf is the
        # per-env step count (already incremented for the current step at this point
        # only inside step(); during reset it is 0, but process_actions is not called
        # at reset). ---
        if self._settle_steps > 0:
            in_settle = self._env.episode_length_buf < self._settle_steps  # [B]
            if bool(in_settle.any()):
                hold = self._entity.data.joint_pos[:, self._target_ids]
                mask = in_settle.unsqueeze(-1)
                self._processed_actions = torch.where(mask, hold, self._processed_actions)
                # Keep the filter state on the held pose so there is no jump when the
                # policy takes over at the end of the settling window.
                self._filtered_target = torch.where(mask, hold, self._filtered_target)
                self._filtered_target_unclamped = torch.where(
                    mask, hold, self._filtered_target_unclamped
                )
        self._backfill_history()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)  # zeros raw actions for env_ids
        default = self._default_target
        zeros = torch.zeros_like(default)
        # Clone to de-alias from _processed_actions before resetting the filter state.
        self._filtered_target = self._filtered_target.clone()
        self._filtered_target_unclamped = self._filtered_target_unclamped.clone()
        self._raw_policy_actions = self._raw_policy_actions.clone()
        self._clipped_raw_actions = self._clipped_raw_actions.clone()
        if env_ids is None:
            self._filtered_target[:] = default
            self._filtered_target_unclamped[:] = default
            self._processed_actions[:] = default
            self._raw_policy_actions[:] = zeros
            self._clipped_raw_actions[:] = zeros
            self._needs_history_backfill[:] = True
        else:
            self._filtered_target[env_ids] = default[env_ids]
            self._filtered_target_unclamped[env_ids] = default[env_ids]
            self._processed_actions[env_ids] = default[env_ids]
            self._raw_policy_actions[env_ids] = zeros[env_ids]
            self._clipped_raw_actions[env_ids] = zeros[env_ids]
            self._needs_history_backfill[env_ids] = True

        manager = self._env.action_manager
        # Explicitly clear action-history buffers when the backend exposes them.
        self._set_if_present(manager, "action", zeros, env_ids)
        self._set_if_present(manager, "prev_action", zeros, env_ids)
        self._set_if_present(manager, "prev_prev_action", zeros, env_ids)
        self._set_if_present(self, "_raw_actions", zeros, env_ids)
        self._set_if_present(self, "_processed_actions", default, env_ids)


class HandHoldAction(ActionTerm):
    """Zero-action-dim term that PD-holds a joint group at a fixed target pose.

    Removes the matched joints from the policy's action space (contributes 0 to
    total_action_dim) while still actively commanding them: apply_actions() —
    called every physics substep, same hook as the main joint_pos term — writes
    a constant joint position target that the entity's builtin PD actuators
    servo to. This makes the "fingers are held, not policy-driven" behavior an
    explicit, configurable part of the action config instead of relying on
    mjlab's reset-time zeroing of joint_pos_target (which only coincidentally
    equals a sensible finger pose while every keyframe leaves fingers at 0).

    The target is NOT default_joint_pos: several G1 finger joints have q=0 as a
    range boundary, so the hold pose is given explicitly via `target_joint_pos`
    (regex -> radians; unmatched joints hold at 0.0).
    """

    cfg: "HandHoldActionCfg"

    def __init__(self, cfg: "HandHoldActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        target_ids, target_names = self._entity.find_joints_by_actuator_names(
            cfg.actuator_names
        )
        assert len(target_ids) > 0, (
            f"HandHoldAction: actuator_names={cfg.actuator_names!r} matched no "
            "actuated joints."
        )
        self._target_ids = torch.tensor(
            target_ids, device=self.device, dtype=torch.long
        )
        target = torch.zeros(len(target_ids), device=self.device)
        if cfg.target_joint_pos:
            idx, _, vals = resolve_matching_names_values(
                cfg.target_joint_pos, target_names
            )
            target[idx] = torch.tensor(vals, device=self.device, dtype=target.dtype)
        # [B, N] fixed hold target; [B, 0] raw-action placeholder.
        self._target = target.unsqueeze(0).expand(self.num_envs, -1).contiguous()
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        del actions  # [B, 0] slice — nothing to consume.

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(
            self._target, joint_ids=self._target_ids
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        # Entity.reset() zeroes joint_pos_target for ALL joints; re-assert the
        # hold target so the fingers never see a 0-rad (range-boundary) command,
        # not even for the first post-reset substep. The target is identical for
        # every env, so writing all envs (ignoring env_ids) is both correct and
        # sidesteps set_joint_position_target's elementwise (non-outer)
        # tensor×tensor indexing when env_ids and joint_ids are both tensors.
        del env_ids
        self._entity.set_joint_position_target(
            self._target, joint_ids=self._target_ids
        )


@dataclass(kw_only=True)
class HandHoldActionCfg(ActionTermCfg):
    """Configuration for HandHoldAction."""

    actuator_names: tuple[str, ...] | list[str]
    """Regex patterns (matched against actuated joint names, same semantics as
    the main joint_pos term) selecting the joints to hold."""

    target_joint_pos: dict[str, float] | None = None
    """Hold pose: regex -> target angle (rad). Unmatched joints hold at 0.0."""

    def build(self, env: "ManagerBasedRlEnv") -> "HandHoldAction":
        return HandHoldAction(self, env)


@dataclass(kw_only=True)
class LowPassJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for LowPassJointPositionAction."""

    alpha: float = 0.5
    """EMA coefficient in (0, 1]. Lower = more smoothing. 1.0 disables the filter."""

    max_rate: float | None = None
    """Optional per-joint slew-rate limit on the commanded PD target, in rad/s.
    Applied AFTER the EMA blend (and before the joint-limit clamp): the change in
    the filtered target from one control step to the next is clamped to
    ±(max_rate * step_dt) rad per joint. Unlike `alpha` — an EMA that moves a fixed
    FRACTION of the remaining distance-to-target each step, so its absolute step
    size is large when the target is far (early in the rise) — this bounds the
    ABSOLUTE speed of the commanded motion regardless of distance. None disables it
    (identical to the EMA-only behavior). Use to slow the fast early-rise phase that
    a low alpha cannot reach."""

    settle_steps: int = 0
    """Number of env-steps at the start of each episode during which the policy does
    NOT control: the PD target is held at the current measured pose so the robot
    settles onto the floor (the integrator resolves spawn contacts). 0 disables it."""

    liftoff_steps: int = 0
    """Number of env-steps right AFTER the settle window during which the reduced
    `liftoff_alpha` / `liftoff_max_rate` values apply (a deliberately slower,
    smoother initial rise). After this hold, the values blend linearly back to the
    nominal `alpha` / `max_rate` over `liftoff_ramp_steps`. 0 disables the phase."""

    liftoff_alpha: float | None = None
    """EMA coefficient during the liftoff phase (lower = slower initial motion).
    None = no override (use `alpha` throughout)."""

    liftoff_max_rate: float | None = None
    """Slew-rate limit (rad/s) during the liftoff phase. Requires `max_rate` to be
    set (there must be a nominal rate to ramp back to). None = no override."""

    liftoff_ramp_steps: int = 20
    """Length (env-steps) of the linear blend from the liftoff values back to the
    nominal `alpha` / `max_rate` once `liftoff_steps` have elapsed. A short ramp
    avoids the jerk seam a hard step-change would introduce mid-rise."""

    default_pos_override: dict[str, float] | None = None
    """Optional anchor pose for the residual/beta scheme: regex -> angle (rad),
    same format as InitialStateCfg.joint_pos (unmatched joints anchor at 0.0).
    When set, p^d = anchor + beta * a uses THIS pose instead of the entity's
    spawn default_joint_pos — needed when the spawn pose (e.g. lying supine) is
    not the pose the beta curriculum should shrink toward (standing). None keeps
    the current spawn-pose behavior."""

    def build(self, env: "ManagerBasedRlEnv") -> LowPassJointPositionAction:
        return LowPassJointPositionAction(self, env)
