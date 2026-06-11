"""Rollout storage for the multi-critic PPO.

Extends the stock ``RolloutStorage`` so that ``rewards``, ``values`` and ``returns``
carry one column per reward group (``[T, N, G]``), while ``advantages`` remains a single
combined scalar (``[T, N, 1]``) so the PPO surrogate loss is unchanged.

It also stores ``next_observations`` (the state s_{t+1} paired with each transition) so
the L2C2 smoothness regularizer can interpolate between consecutive states. The feed
forward mini-batch generator additionally yields ``next_observations`` and ``dones``.
"""

from __future__ import annotations

from collections.abc import Generator

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

        # Next observation s_{t+1} for each transition (L2C2 smoothness). Filled by
        # fill_next_observations() once the rollout is complete.
        self.next_observations = TensorDict(
            {key: torch.zeros_like(value) for key, value in self.observations.items()},
            batch_size=[T, N],
            device=self.device,
        )

    def fill_next_observations(self, final_obs: TensorDict) -> None:
        """Populate next_observations: shift observations by one step, append final_obs.

        next_observations[t] = observations[t+1] for t < T-1, and
        next_observations[T-1] = final_obs (the obs after the last stored action).
        Uses copy_ so the buffers stay regular (autograd-usable) tensors even though
        this is called inside the runner's inference_mode block.
        """
        for key in self.observations.keys():
            self.next_observations[key][:-1].copy_(self.observations[key][1:])
            self.next_observations[key][-1].copy_(final_obs[key])

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

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[RolloutStorage.Batch, None, None]:
        """Feed-forward generator that also yields next_observations and dones (L2C2)."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        dones = self.dones.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                batch = RolloutStorage.Batch(
                    observations=observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                    dones=dones[batch_idx],
                )
                # Attached for the L2C2 smoothness term (not a stock Batch field).
                batch.next_observations = next_observations[batch_idx]
                yield batch
