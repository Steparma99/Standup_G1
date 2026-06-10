"""RL configuration for Unitree G1 get-up task."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def unitree_g1_getup_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Unitree G1 get-up task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(768, 384, 192),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(768, 384, 192),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            # Raised 0.008→0.015: entropy collapsed to ~0.3 within ~1k iters last run
            # (premature convergence to the lie-still local optimum). Keep exploration
            # alive longer so the policy actually tries to rise.
            entropy_coef=0.015,
            num_learning_epochs=5,
            num_mini_batches=8,
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            # Raised 0.01→0.02: last run the adaptive-KL schedule floored the LR to ~0
            # within ~1k iters (huge early advantages from 1000-step static-reward
            # farming). More KL tolerance keeps the LR from permanently collapsing;
            # shorter episodes from no_progress_timeout also shrink those advantages.
            desired_kl=0.02,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_getup",
        # W&B online for real-time dashboards: every Episode_Reward/<term> and
        # metric is auto-plotted. Requires `wandb login` once (or set WANDB_API_KEY);
        # do NOT export WANDB_MODE=offline if you want live streaming.
        logger="wandb",
        wandb_project="g1_getup",
        save_interval=100,
        num_steps_per_env=64,
        max_iterations=12000,
    )
