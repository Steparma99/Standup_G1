"""HoST-style multi-critic PPO.

Subclasses the stock rsl-rl :class:`PPO`. Instead of one critic on the summed reward,
there are ``G`` independent critics (one per reward group). Per-group GAE produces
per-group advantages, which are normalized and combined with ``reward_group_weights``
into a single scalar advantage used by the (unchanged) PPO surrogate loss. The value
loss is the sum of the per-group clipped value losses.

NOTE: the advantage-combining rule and value-loss form below are a sensible DEFAULT.
The exact critic loss / advantage formula / PPO hyperparameters will be provided by the
user and are expected to replace this default combiner.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from .multi_critic_model import MultiCriticModel
from .multi_critic_storage import MultiCriticRolloutStorage
from .reward_groups import GROUP_ORDER, build_group_onehot


class MultiCriticPPO(PPO):
    """PPO with one critic per reward group."""

    def __init__(
        self,
        actor,
        critic: MultiCriticModel,
        storage: MultiCriticRolloutStorage,
        *,
        env: VecEnv,
        reward_group_weights,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.env = env
        self.num_groups = len(reward_group_weights)
        self.reward_group_weights = torch.as_tensor(
            reward_group_weights, dtype=torch.float32, device=self.device
        )
        # Lazily built on the first env step from RewardManager.active_terms.
        self._group_onehot: torch.Tensor | None = None

    # ------------------------------------------------------------------ rewards
    def _grouped_rewards(self) -> torch.Tensor:
        """Aggregate the per-term reward breakdown into per-group sums ([N, G]).

        Reads ``RewardManager._step_reward`` ([N, num_terms], per-term unscaled rate)
        and scales by ``step_dt`` to match the scalar ``reward_buf`` convention.
        """
        reward_manager = self.env.unwrapped.reward_manager
        step_reward = reward_manager._step_reward  # [N, num_terms]
        if self._group_onehot is None:
            self._group_onehot = build_group_onehot(
                list(reward_manager.active_terms), device=step_reward.device
            )
            if self._group_onehot.shape[1] != self.num_groups:
                raise ValueError(
                    f"reward_group_weights has {self.num_groups} entries but the group "
                    f"map defines {self._group_onehot.shape[1]} groups."
                )
        # Match reward_buf's convention: terms are scaled by dt iff the manager does.
        scale = self.env.unwrapped.step_dt if getattr(reward_manager, "_scale_by_dt", True) else 1.0
        grouped = (step_reward @ self._group_onehot) * scale  # [N, G]
        return grouped.to(self.device)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one step using per-group rewards instead of the scalar reward."""
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)

        # Per-group rewards ([N, G]); the scalar `rewards` arg is ignored (used only for
        # the runner's bookkeeping/logging).
        self.transition.rewards = self._grouped_rewards()
        self.transition.dones = dones

        # Bootstrap on time outs (per group; transition.values is [N, G]).
        if "time_outs" in extras:
            self.transition.rewards = self.transition.rewards + self.gamma * (
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device)
            )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    # ----------------------------------------------------------------- returns
    def compute_returns(self, obs: TensorDict) -> None:
        """Per-group GAE -> per-group advantage normalization -> weighted scalar advantage."""
        st = self.storage
        last_values = self.critic(obs).detach()  # [N, G]

        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()  # [N, 1]
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]

        # Per-group advantages, normalized independently over (T, N).
        adv = st.returns - st.values  # [T, N, G]
        mean = adv.mean(dim=(0, 1), keepdim=True)
        std = adv.std(dim=(0, 1), keepdim=True)
        adv_norm = (adv - mean) / (std + 1e-8)

        # Combine into a single weighted scalar advantage [T, N, 1] for the surrogate.
        w = self.reward_group_weights.view(1, 1, -1)
        st.advantages = (adv_norm * w).sum(dim=-1, keepdim=True)

    # ------------------------------------------------------------------ update
    def update(self) -> dict[str, float]:
        """PPO update. Surrogate is unchanged; value loss is summed over the G critics."""
        if self.rnd is not None or self.symmetry is not None:
            raise NotImplementedError("MultiCriticPPO does not support RND or symmetry.")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_group_value_losses = torch.zeros(self.num_groups, device=self.device)

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            # Recompute log-prob / entropy / values under current parameters.
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])  # [B, G]
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Adaptive learning rate from KL divergence.
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss (combined scalar advantage).
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            advantages = torch.squeeze(batch.advantages)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss summed over the G critics.
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                v_losses = (values - batch.returns).pow(2)
                v_losses_clipped = (value_clipped - batch.returns).pow(2)
                per_group = torch.max(v_losses, v_losses_clipped)  # [B, G]
            else:
                per_group = (batch.returns - values).pow(2)
            group_value_losses = per_group.mean(dim=0)  # [G]
            value_loss = group_value_losses.sum()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_group_value_losses += group_value_losses.detach()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_group_value_losses /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        for i, name in enumerate(GROUP_ORDER):
            loss_dict[f"value_{name}"] = mean_group_value_losses[i].item()
        return loss_dict

    # ------------------------------------------------------------- construction
    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "MultiCriticPPO":
        """Build the multi-critic algorithm (mirrors PPO.construct_algorithm)."""
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        # The critic is always our MultiCriticModel; ignore any critic class_name.
        cfg["critic"].pop("class_name", None)

        # Resolve obs groups; ensure rnd_cfg / symmetry_cfg keys exist (set to None) so
        # the runner's learn() loop can read cfg["algorithm"]["rnd_cfg"] without KeyError.
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
        # CNN-encoder sharing is not supported by the multi-critic ensemble.
        cfg["algorithm"].pop("share_cnn_encoders", None)

        reward_group_weights = cfg["algorithm"]["reward_group_weights"]
        num_groups = len(reward_group_weights)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        critic = MultiCriticModel(obs, cfg["obs_groups"], "critic", num_groups, **cfg["critic"]).to(device)
        print(f"Critic Model ({num_groups} critics): {critic}")

        storage = MultiCriticRolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], num_groups, device
        )

        alg = alg_class(
            actor,
            critic,
            storage,
            env=env,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        return alg
