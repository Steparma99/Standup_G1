"""Rollout storage for the multi-critic PPO.

Extends the stock ``RolloutStorage`` so that ``rewards``, ``values`` and ``returns``
carry one column per reward group (``[T, N, G]``), while ``advantages`` remains a single
combined scalar (``[T, N, 1]``) so the PPO surrogate loss is unchanged.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage


class MultiCriticRolloutStorage(RolloutStorage):
    """RolloutStorage with per-group reward/value/return buffers."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        num_groups: int,
        device: str = "cpu",
    ) -> None:
        if training_type != "rl":
            raise ValueError("MultiCriticRolloutStorage only supports training_type='rl'.")
        self.num_groups = num_groups
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)

        # Re-allocate the per-group buffers ([T, N, G]). advantages stays [T, N, 1]
        # (the combined, group-weighted scalar consumed by the surrogate loss).
        T, N = num_transitions_per_env, num_envs
        self.rewards = torch.zeros(T, N, num_groups, device=self.device)
        self.values = torch.zeros(T, N, num_groups, device=self.device)
        self.returns = torch.zeros(T, N, num_groups, device=self.device)
        # self.advantages keeps the base [T, N, 1] allocation.

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Add one transition; rewards/values are stored per group ([N, G])."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)  # type: ignore
        self.rewards[self.step].copy_(transition.rewards.view(-1, self.num_groups))  # type: ignore
        self.dones[self.step].copy_(transition.dones.view(-1, 1))  # type: ignore

        # Reinforcement learning
        self.values[self.step].copy_(transition.values)  # type: ignore  # [N, G]
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))  # type: ignore
        if self.distribution_params is None:
            self.distribution_params = tuple(
                torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
                for p in transition.distribution_params  # type: ignore
            )
        for i, p in enumerate(transition.distribution_params):  # type: ignore
            self.distribution_params[i][self.step].copy_(p)

        # Recurrent hidden states (no-op for feedforward)
        self._save_hidden_states(transition.hidden_states)

        self.step += 1
