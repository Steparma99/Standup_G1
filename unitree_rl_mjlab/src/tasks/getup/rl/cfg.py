"""Config dataclasses for the multi-critic algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlPpoAlgorithmCfg

from .reward_groups import GROUP_WEIGHTS


@dataclass
class MultiCriticPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm config extended with per-group critic weights.

    ``class_name`` points rsl-rl's ``resolve_callable`` at our custom algorithm, which is
    plugged in without editing the installed rsl-rl package.
    """

    reward_group_weights: tuple[float, ...] = field(default_factory=lambda: GROUP_WEIGHTS)
    """Per-group advantage weights, ordered like ``reward_groups.GROUP_ORDER``."""

    l2c2_actor_coef: float = 1.0
    """L2C2 smoothness weight on the actor (lambda_actor)."""

    l2c2_critic_coef: float = 0.1
    """L2C2 smoothness weight applied to EACH group critic (lambda_critic)."""

    class_name: str = "src.tasks.getup.rl.multi_critic_ppo:MultiCriticPPO"
    """Resolved by rsl-rl to the multi-critic PPO implementation."""
