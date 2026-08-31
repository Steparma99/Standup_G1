"""Custom distribution modules for the get-up task."""

import math

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules.distribution import Distribution


class _IdentityModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ClampedGaussianDistribution(Distribution):
    """Gaussian policy with a smoothly bounded, state-independent std.

        sigma = sigma_min + (sigma_max - sigma_min) * sigmoid(p)

    `p` is an unconstrained nn.Parameter (state-independent: one entry per action
    dim, shared across the batch — only the MEAN comes from the actor network).

    This replaces the previous hard `clamp(std_param, min_std, max_std)`. The
    clamp has ZERO gradient w.r.t. the parameter exactly at the bounds, which is
    where a saturating policy ends up sitting: once std_param hit max_std the
    entropy term could no longer move it back, and once it hit min_std it was
    frozen. The sigmoid squash (Andrychowicz et al., "What Matters in On-Policy
    RL") keeps the gradient non-zero everywhere while still guaranteeing
    sigma in (sigma_min, sigma_max) by construction, and makes the
    initialisation exactly controllable: p is set to the logit that reproduces
    `init_std`,

        p0 = log((sigma_0 - sigma_min) / (sigma_max - sigma_0)).

    NOTE (checkpoints): the learnable parameter is renamed `std_logit_param` and
    its VALUE semantics changed (it is now a logit, not sigma itself). Loading a
    pre-change checkpoint therefore fails loudly on the missing key rather than
    silently reinterpreting the old sigma as a logit.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.3,
        std_type: str = "scalar",
        min_std: float = 0.05,
        max_std: float = 1.0,
    ) -> None:
        super().__init__(output_dim)
        if std_type != "scalar":
            raise ValueError(
                f"ClampedGaussianDistribution: std_type={std_type!r} is no longer "
                "supported — the sigmoid parametrisation makes the scalar/log "
                "distinction meaningless (the parameter is an unconstrained logit "
                "in both cases). Use std_type='scalar'."
            )
        if not min_std < init_std < max_std:
            raise ValueError(
                f"ClampedGaussianDistribution: init_std={init_std} must lie strictly "
                f"inside (min_std, max_std) = ({min_std}, {max_std}); the logit "
                "initialisation is undefined at or outside the bounds."
            )
        self.std_type = std_type
        self._min_std = min_std
        self._max_std = max_std

        # p0 = log((sigma_0 - sigma_min) / (sigma_max - sigma_0))
        p0 = math.log((init_std - min_std) / (max_std - init_std))
        self.std_logit_param = nn.Parameter(torch.full((output_dim,), p0))

        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        mean = mlp_output
        std = self._min_std + (self._max_std - self._min_std) * torch.sigmoid(
            self.std_logit_param
        )
        self._distribution = Normal(mean, std.expand_as(mean))

    def sample(self) -> torch.Tensor:
        return self._distribution.sample()  # type: ignore

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output

    def as_deterministic_output_module(self) -> nn.Module:
        return _IdentityModule()

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        return self._distribution.mean  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        return self._distribution.stddev  # type: ignore

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=-1)  # type: ignore

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self.mean, self.std)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(outputs).sum(dim=-1)  # type: ignore

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(
            Normal(old_mean, old_std), Normal(new_mean, new_std)
        ).sum(dim=-1)
