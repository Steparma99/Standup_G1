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
        self._alpha = float(cfg.alpha)
        # Filter state starts at the default pose (the neutral command).
        self._filtered_target = (
            self._entity.data.default_joint_pos[:, self._target_ids].clone()
        )

    @property
    def applied_target(self) -> torch.Tensor:
        """The (filtered) joint position target actually commanded to the PD [B, A]."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        # super() sets _processed_actions = scale * action + default (the desired target).
        super().process_actions(actions)
        self._filtered_target = (
            self._alpha * self._processed_actions
            + (1.0 - self._alpha) * self._filtered_target
        )
        self._processed_actions = self._filtered_target

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)  # zeros raw actions for env_ids
        default = self._entity.data.default_joint_pos[:, self._target_ids]
        # Clone to de-alias from _processed_actions before resetting the filter state.
        self._filtered_target = self._filtered_target.clone()
        if env_ids is None:
            self._filtered_target[:] = default
        else:
            self._filtered_target[env_ids] = default[env_ids]


@dataclass(kw_only=True)
class LowPassJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for LowPassJointPositionAction."""

    alpha: float = 0.5
    """EMA coefficient in (0, 1]. Lower = more smoothing. 1.0 disables the filter."""

    def build(self, env: "ManagerBasedRlEnv") -> LowPassJointPositionAction:
        return LowPassJointPositionAction(self, env)
