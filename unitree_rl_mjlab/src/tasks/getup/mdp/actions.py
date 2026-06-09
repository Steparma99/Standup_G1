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

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["LowPassJointPositionAction", "LowPassJointPositionActionCfg"]


class LowPassJointPositionAction(JointPositionAction):
    """Joint position action with an EMA low-pass filter on the commanded target."""

    cfg: "LowPassJointPositionActionCfg"

    def __init__(self, cfg: "LowPassJointPositionActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self._env = env
        self._alpha = float(cfg.alpha)
        self._settle_steps = int(cfg.settle_steps)
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
        super().process_actions(self._clipped_raw_actions)
        ema_target = (
            self._alpha * self._processed_actions
            + (1.0 - self._alpha) * self._filtered_target
        )
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


@dataclass(kw_only=True)
class LowPassJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for LowPassJointPositionAction."""

    alpha: float = 0.5
    """EMA coefficient in (0, 1]. Lower = more smoothing. 1.0 disables the filter."""

    settle_steps: int = 0
    """Number of env-steps at the start of each episode during which the policy does
    NOT control: the PD target is held at the current measured pose so the robot
    settles onto the floor (the integrator resolves spawn contacts). 0 disables it."""

    def build(self, env: "ManagerBasedRlEnv") -> LowPassJointPositionAction:
        return LowPassJointPositionAction(self, env)
