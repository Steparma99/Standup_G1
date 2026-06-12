"""Custom distribution modules for the get-up task."""

import torch
from torch.distributions import Normal
from rsl_rl.modules.distribution import GaussianDistribution


class ClampedGaussianDistribution(GaussianDistribution):
    """GaussianDistribution with a hard clamp on the learned std parameter.

    Prevents the action std from exploding during early training when task
    reward is sparse and the entropy gradient becomes the dominant signal.
    The clamp is applied at sample time (not gradient time), so gradients
    still flow through std_param — they just can't push std above max_std.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_type: str = "scalar",
        min_std: float = 0.05,
        max_std: float = 2.0,
    ) -> None:
        super().__init__(output_dim, init_std, std_type)
        self._min_std = min_std
        self._max_std = max_std

    def update(self, mlp_output: torch.Tensor) -> None:
        mean = mlp_output
        if self.std_type == "scalar":
            std = torch.clamp(self.std_param, self._min_std, self._max_std).expand_as(mean)
        elif self.std_type == "log":
            std = torch.clamp(torch.exp(self.log_std_param), self._min_std, self._max_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown std_type: {self.std_type}")
        self._distribution = Normal(mean, std)
