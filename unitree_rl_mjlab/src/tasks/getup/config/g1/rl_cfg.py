"""RL configuration for Unitree G1 get-up task."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
)

from src.tasks.getup.rl.cfg import MultiCriticPpoAlgorithmCfg


def unitree_g1_getup_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Unitree G1 get-up task."""
    return RslRlOnPolicyRunnerCfg(
        # HoST-style actor: 3-layer MLP [512, 256, 128], ELU, stacked obs history (n=5,
        # set on the actor obs group in getup_env_cfg.py). Gaussian policy with a learned
        # scalar std at train time; deterministic mean at deploy.
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "src.tasks.getup.rl.distribution:ClampedGaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
                "min_std": 0.05,
                "max_std": 2.0,
            },
        ),
        # HoST-style critic: this config is applied to EACH of the 4 group critics
        # (Task / Style / Regularization / Post-task). Each is a 2-layer MLP [512, 256]
        # outputting a scalar value for its reward group. Critics see the privileged
        # critic obs and are free at deployment (only the actor is exported to ONNX).
        critic=RslRlModelCfg(
            hidden_dims=(512, 256),
            activation="elu",
            obs_normalization=True,
        ),
        # Multi-critic PPO: per-group advantages combined with reward_group_weights
        # (Task 2.5 / Style 1.9 / Regularization 0.1 / Post-task 1.0).
        algorithm=MultiCriticPpoAlgorithmCfg(
            reward_group_weights=(2.5, 1.9, 0.1, 1.0),
            # L2C2 smoothness: Lipschitz-continuity penalty on the actor and each critic,
            # added to the loss every update (lambda_actor=1.0, lambda_critic=0.1 per critic).
            l2c2_actor_coef=1.0,
            l2c2_critic_coef=0.1,
            value_loss_coef=1.0,
            # Plain per-group MSE value loss (no value clipping): the multi-critic
            # objective sums L(phi_i) = ||return_i - V_i||^2 over the 4 group critics.
            use_clipped_value_loss=False,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
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
        num_steps_per_env=50,
        max_iterations=12000,
    )
