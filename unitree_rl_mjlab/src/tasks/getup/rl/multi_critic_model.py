"""Multi-critic value model: an ensemble of independent critic MLPs.

Wraps ``num_critics`` stock ``MLPModel`` critics (one per reward group) in a single
module whose forward returns a ``[N, num_critics]`` value tensor. Reusing ``MLPModel``
gives us obs-group selection, observation normalization, and activation handling for
free, identical to the single-critic path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState


class MultiCriticModel(nn.Module):
    """Ensemble of ``num_critics`` independent scalar-value MLP critics."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        num_critics: int,
        **critic_cfg,
    ) -> None:
        super().__init__()
        self.num_critics = num_critics
        # Each critic is a deterministic scalar-output MLPModel (no distribution_cfg).
        # We deep-copy the cfg per critic so MLPModel is free to pop/mutate its own copy.
        self.critics = nn.ModuleList(
            [
                MLPModel(obs, obs_groups, obs_set, 1, **dict(critic_cfg))
                for _ in range(num_critics)
            ]
        )

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return value estimates of shape ``[N, num_critics]``."""
        values = [critic(obs, masks=masks, hidden_state=hidden_state) for critic in self.critics]
        return torch.cat(values, dim=-1)

    # --- Delegation to the underlying MLPModel critics (non-recurrent no-ops) -------
    def update_normalization(self, obs: TensorDict) -> None:
        for critic in self.critics:
            critic.update_normalization(obs)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        for critic in self.critics:
            critic.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass
