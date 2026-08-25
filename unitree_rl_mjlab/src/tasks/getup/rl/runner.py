import os
import time

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.utils import check_nan

from .deterministic_eval import DeterministicEvaluator

# Env attributes published by the from-scratch env config / curriculum event.
_FS_CFG_ATTR = "_fromscratch_cfg"
_FS_CURRICULUM_ATTR = "_fs_assist_beta_curriculum"


class GetUpOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = "policy.onnx"
        self.export_policy_to_onnx(policy_path, filename)
        run_name: str = (
            wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
        )  # type: ignore[assignment]
        onnx_path = os.path.join(policy_path, filename)
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        attach_metadata_to_onnx(onnx_path, metadata)
        if self.logger.logger_type in ["wandb"]:
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

    # ------------------------------------------------------------------ evaluation
    def _build_evaluator(self) -> DeterministicEvaluator | None:
        """Build the deterministic evaluator, or None if this task doesn't use it.

        Both the from-scratch task and the legacy get-up task share this runner; the
        legacy env publishes neither the curriculum cfg nor the coupled curriculum
        event, so it silently keeps the stock training loop.
        """
        unwrapped = self.env.unwrapped
        fs_cfg = getattr(unwrapped, _FS_CFG_ATTR, None)
        curriculum = getattr(unwrapped, _FS_CURRICULUM_ATTR, None)
        if fs_cfg is None or curriculum is None:
            return None
        eval_cfg = getattr(fs_cfg, "deterministic_eval", None)
        if eval_cfg is None or not eval_cfg.enabled:
            return None
        return DeterministicEvaluator(
            self.env, self.alg, eval_cfg, curriculum, self.device
        )

    def _log_eval(self, it: int, stats: dict) -> None:
        decision = stats.pop("decision", "hold")
        print(
            f"[eval] it={it} phase={self._evaluator.phase()} "  # type: ignore[union-attr]
            f"p_hat={stats['p_hat']:.4f} (+-{stats['se']:.4f}, N={int(stats['n_episodes'])}) "
            f"-> {decision.upper()}  lambda_F={stats['lambda_F']:.1f} beta={stats['beta']:.3f}"
        )
        writer = self.logger.writer
        if writer is None:
            return
        for key, value in stats.items():
            writer.add_scalar(f"Eval/{key}", value, it)
        writer.add_scalar("Eval/advanced", 1.0 if decision == "advance" else 0.0, it)
        writer.add_scalar("Eval/rolled_back", 1.0 if decision == "rollback" else 0.0, it)

    def _run_evaluation(self, it: int) -> None:
        """Evaluate, apply the curriculum decision, and restore the training rollout."""
        assert self._evaluator is not None
        t0 = time.time()
        stats = self._evaluator.step()
        stats["eval_time_s"] = time.time() - t0
        self._log_eval(it, stats)

        # The evaluation left every env freshly reset and perfectly in phase. Left
        # alone, all envs would then time out on the SAME step for the rest of the run,
        # making episode-end statistics bursty and the value bootstrap correlated.
        # Re-stagger exactly as `init_at_random_ep_len` does at startup.
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf, high=int(self.env.max_episode_length)
        )

    # ------------------------------------------------------------------ train loop
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """rsl_rl 5.0.1 OnPolicyRunner.learn() + the deterministic-evaluation hook.

        Copied rather than wrapped: rsl_rl exposes no per-iteration callback, and
        calling learn() in chunks would re-initialise the W&B writer and write a
        "final" checkpoint per chunk. Kept deliberately close to the original so
        upstream diffs stay readable — the additions are the two marked blocks.
        """
        # ---- ADDED: deterministic evaluation ----
        self._evaluator = self._build_evaluator()
        next_eval_it = None
        if self._evaluator is not None:
            next_eval_it = self.current_learning_iteration + self._evaluator.interval()
            print(
                f"[eval] deterministic evaluation ENABLED: N={self._evaluator.n_eval} "
                f"episodes, phase {self._evaluator.phase()} interval "
                f"{self._evaluator.interval()} its, first eval at it={next_eval_it}. "
                "The rolling K-window trigger is demoted to a diagnostic."
            )
        # ---- END ADDED ----

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    # Move to device
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards if RND is used (only for logging)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    )
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

            # ---- ADDED: deterministic evaluation + coupled curriculum advancement ----
            if self._evaluator is not None and next_eval_it is not None and it >= next_eval_it:
                self._run_evaluation(it)
                # The interval is phase-dependent, so it is re-read AFTER the decision.
                next_eval_it = it + self._evaluator.interval()
                # The evaluation replaced the env state; the cached obs is stale.
                obs = self.env.get_observations().to(self.device)
            # ---- END ADDED ----

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
