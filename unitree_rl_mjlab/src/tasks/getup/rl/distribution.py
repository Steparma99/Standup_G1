"""Custom distribution modules for the get-up task."""

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules.distribution import Distribution


class _IdentityModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ClampedGaussianDistribution(Distribution):
    """Gaussian distribution with a hard clamp on the learned std.

    Inherits directly from Distribution (not GaussianDistribution) to avoid
    rsl_rl version-compatibility issues with std_range handling. Functionally
    identical to GaussianDistribution but with std clamped to [min_std, max_std].
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_type: str = "scalar",
        min_std: float = 0.05,
        max_std: float = 2.0,
    ) -> None:
        super().__init__(output_dim)
        self.std_type = std_type
        self._min_std = min_std
        self._max_std = max_std

        if std_type == "scalar":
            self.std_param = nn.Parameter(init_std * torch.ones(output_dim))
        elif std_type == "log":
            self.log_std_param = nn.Parameter(torch.log(init_std * torch.ones(output_dim)))
        else:
            raise ValueError(f"Unknown std_type: {std_type}")

        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        mean = mlp_output
        if self.std_type == "scalar":
            std = torch.clamp(self.std_param, self._min_std, self._max_std).expand_as(mean)
        else:
            std = torch.clamp(torch.exp(self.log_std_param), self._min_std, self._max_std).expand_as(mean)
        self._distribution = Normal(mean, std)

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
