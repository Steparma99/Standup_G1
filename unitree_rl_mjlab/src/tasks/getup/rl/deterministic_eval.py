"""Deterministic evaluation — the curriculum's advancement authority.

Training rolls out a STOCHASTIC policy, a_t ~ N(mu_theta(o_t), sigma). Any success rate
measured on that rollout therefore measures the *exploration process*: it moves when the
entropy coefficient, the learned std, or the KL schedule move, and it is systematically
pessimistic about what the policy actually knows. Advancing a curriculum on it couples
difficulty to exploration noise.

This module instead measures COMPETENCE: every `interval` policy iterations the runner
pauses learning and rolls out

    a_t = mu_theta(o_t)                       (the distribution mean, no sampling)

for one full episode horizon on every parallel env, and estimates

    p_hat = (1/N) sum_i 1[ stable_counter_i >= N_hold at episode end ]

using the SAME success predicate as training (the shared chi_t stable-hold product, see
mdp/standing.py). The coupled (lambda_F, beta) level then steps down at most once per
evaluation, per `EvalAdvancementPolicy` below.

Hygiene the rollout maintains (all of it matters):
  * the actor is called directly, NOT through `alg.act()` — nothing is written to the
    PPO rollout storage, and `process_env_step` (which updates the observation
    normalizer's running mean/var) is never called, so evaluation cannot shift the
    normalizer;
  * `alg.eval_mode()` for the duration, restored to `train_mode()` afterwards;
  * the curriculum event is frozen: episode ends are routed to this module's sink
    instead of the rolling K-window buffer, so the training diagnostic
    (curriculum/fs_success_rate) never sees evaluation episodes, and no level can move
    mid-evaluation;
  * the assist wrench KEEPS being applied at the current lambda_F — it is part of the
    difficulty level under measurement, not a training-only aid;
  * observation noise stays ON: it is the deployment condition and, with all pose
    randomization disabled in this phase, the only thing decorrelating the N episodes;
  * the torch RNG is saved, re-seeded with a constant, and restored, so consecutive
    evaluations are PAIRED (same N noise sequences) and their difference reflects the
    policy rather than the draw. Training RNG is untouched;
  * under torchrun, the per-rank success counts are all-reduced before the decision, so
    every rank advances to the identical level (per-rank p_hat would otherwise let the
    global scalars drift apart between ranks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..curriculum_cfg import DeterministicEvalCfg


# ======================================================================================
# Advancement rule (pure python — no torch, no env; unit-testable in isolation)
# ======================================================================================
class EvalAdvancementPolicy:
    """Decides advance / hold / rollback from the sequence of deterministic p_hat.

    Base rule:      advance iff p_hat >= tau_s.
    Anti-fluke:     if the PREVIOUS evaluation was far below threshold
                    (p_prev < tau_s - guard_delta), a jump straight to >= tau_s is more
                    likely a lucky draw than mastery, so a second consecutive
                    evaluation >= tau_s is required to confirm. guard_delta = 0.08 is
                    ~5 sigma at N=512 and ~10 sigma at N=2048, so the guard reacts to
                    real regressions, never to sampling noise.
    Rollback:       `rollback_consecutive` evaluations below `rollback_success_rate`
                    step the level back UP once, after which `readvance_consecutive`
                    consecutive evaluations >= tau_s are needed to advance again.
    """

    ADVANCE = "advance"
    HOLD = "hold"
    ROLLBACK = "rollback"

    def __init__(self, cfg: "DeterministicEvalCfg") -> None:
        self.tau = float(cfg.success_rate_threshold)
        self.guard_delta = float(cfg.guard_delta)
        self.rollback_enabled = bool(cfg.rollback_enabled)
        self.rollback_rate = float(cfg.rollback_success_rate)
        self.rollback_consecutive = int(cfg.rollback_consecutive)
        self.readvance_consecutive = int(cfg.readvance_consecutive)

        self.history: list[float] = []
        self._high_streak = 0     # consecutive evaluations >= tau_s
        self._low_streak = 0      # consecutive evaluations < rollback_rate
        self._rollback_armed = False  # a rollback happened; re-advance needs confirming

    @property
    def required_consecutive(self) -> int:
        """How many consecutive p_hat >= tau_s the CURRENT state demands to advance."""
        if self._rollback_armed:
            return self.readvance_consecutive
        prev = self.history[-2] if len(self.history) >= 2 else None
        if prev is not None and prev < self.tau - self.guard_delta:
            return 2
        return 1

    def decide(self, p_hat: float, can_rollback: bool = True) -> str:
        """Record p_hat and return ADVANCE / HOLD / ROLLBACK."""
        p_hat = float(p_hat)
        self.history.append(p_hat)

        # --- rollback check first: a collapsed level must not be advanced off of ---
        if self.rollback_enabled:
            self._low_streak = self._low_streak + 1 if p_hat < self.rollback_rate else 0
            if self._low_streak >= self.rollback_consecutive and can_rollback:
                self._low_streak = 0
                self._high_streak = 0
                self._rollback_armed = True
                return self.ROLLBACK

        # --- advancement ---
        if p_hat < self.tau:
            self._high_streak = 0
            return self.HOLD

        self._high_streak += 1
        if self._high_streak >= self.required_consecutive:
            self._high_streak = 0
            self._rollback_armed = False
            self._low_streak = 0
            return self.ADVANCE
        return self.HOLD


# ======================================================================================
# Deterministic rollout
# ======================================================================================
class DeterministicEvaluator:
    """Runs the deterministic rollout and applies the advancement decision."""

    def __init__(self, env, alg, cfg: "DeterministicEvalCfg", curriculum, device) -> None:
        self._env = env                  # RslRlVecEnvWrapper
        self._alg = alg
        self._cfg = cfg
        self._curr = curriculum          # FromScratchAssistBetaCurriculum
        self._device = device
        self._policy = EvalAdvancementPolicy(cfg)

        self._num_envs = int(env.num_envs)
        n = int(cfg.n_eval) if cfg.n_eval else self._num_envs
        self.n_eval = min(n, self._num_envs)
        self._success_window = int(curriculum.success_window_steps)
        self.eval_count = 0

        # The evaluator, not the K-window, now owns advancement.
        curriculum.set_autonomous(False)

        # Per-episode records, filled by the sink at each env's FIRST termination.
        # Records live on the ENV's device (where the curriculum's cached tensors are),
        # which may differ from the learning device.
        dev = getattr(env, "unwrapped", env).device
        self._done_once = torch.zeros(self._num_envs, dtype=torch.bool, device=dev)
        self._success = torch.zeros(self._num_envs, dtype=torch.bool, device=dev)
        self._fell = torch.zeros(self._num_envs, dtype=torch.bool, device=dev)
        self._hold = torch.zeros(self._num_envs, dtype=torch.long, device=dev)
        self._ttl_hold = torch.full((self._num_envs,), -1, dtype=torch.long, device=dev)
        self._ep_len = torch.zeros(self._num_envs, dtype=torch.long, device=dev)

    # -- interval schedule ------------------------------------------------------
    def phase(self) -> str:
        """A / B / C — see DeterministicEvalCfg."""
        if self._curr.lambda_F > self._curr.lambda_F_min:
            return "A"
        if self._curr.beta > self._curr.beta_min:
            return "B"
        return "C"

    def interval(self) -> int:
        return {
            "A": self._cfg.interval_phase_a,
            "B": self._cfg.interval_phase_b,
            "C": self._cfg.interval_phase_c,
        }[self.phase()]

    # -- episode sink -----------------------------------------------------------
    def _record(self, env_ids, succ, info) -> None:
        """Called from the frozen curriculum's reset(), i.e. exactly at an episode end.

        Only an env's FIRST episode inside the evaluation counts: an env that falls at
        step 100 restarts and would otherwise contribute a second, shorter episode and
        bias p_hat toward whatever the policy does in the remaining steps.
        """
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self._num_envs, device=self._done_once.device)
        env_ids = env_ids.to(self._done_once.device)
        fresh = ~self._done_once[env_ids]
        if not bool(fresh.any()):
            return
        ids = env_ids[fresh]
        self._success[ids] = succ.to(self._success.device)[fresh]
        self._fell[ids] = info["fall_after_stand"][fresh]
        self._hold[ids] = info["stable_counter"][fresh]
        self._ep_len[ids] = info["episode_length"][fresh]
        # first_stable_step is a GLOBAL step index; convert to an in-episode step.
        first = info["first_stable_step"][fresh]
        in_ep = info["episode_length"][fresh] - (int(info["common_step"]) - first)
        self._ttl_hold[ids] = torch.where(first >= 0, in_ep, torch.full_like(in_ep, -1))
        self._done_once[ids] = True

    def _reset_records(self) -> None:
        self._done_once.zero_()
        self._success.zero_()
        self._fell.zero_()
        self._hold.zero_()
        self._ttl_hold.fill_(-1)
        self._ep_len.zero_()

    # -- the evaluation ---------------------------------------------------------
    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        # inference_mode, NOT no_grad: the training rollout collects under
        # torch.inference_mode(), so the env's per-episode state tensors are inference
        # tensors. Writing to them (which every env reset does) from a plain no_grad
        # block raises "Inplace update to inference tensor outside InferenceMode".
        env = self._env
        curr = self._curr
        cfg = self._cfg
        hz = 50.0  # policy rate; step<->second conversion for the reported durations

        rng_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        if cfg.paired_seed is not None:
            torch.manual_seed(int(cfg.paired_seed))

        self._alg.eval_mode()
        actor = self._alg.get_policy()
        self._reset_records()

        # Freeze WITHOUT a sink first: env.reset() ends the in-flight TRAINING episodes
        # and those must not be recorded as evaluation samples.
        curr.freeze(None)
        try:
            env.reset()
            curr.freeze(self._record)
            obs = env.get_observations().to(self._device)
            for _ in range(int(cfg.max_eval_steps)):
                actions = actor(obs, stochastic_output=False)
                obs, _, _, _ = env.step(actions.to(env.device))
                obs = obs.to(self._device)
                if bool(self._done_once.all()):
                    break
        finally:
            curr.unfreeze()

        # Envs that never terminated inside the horizon (should not happen when
        # max_eval_steps == the episode horizon): count their unfinished episode from
        # the curriculum's live caches rather than dropping them from the sample.
        if not bool(self._done_once.all()):
            snap = curr.latest_episode_snapshot()
            pend = ~self._done_once
            self._success[pend] = snap["stable_counter"][pend] >= self._success_window
            self._fell[pend] = snap["fall_after_stand"][pend]
            self._hold[pend] = snap["stable_counter"][pend]
            self._ep_len[pend] = snap["episode_length"][pend]

        torch.set_rng_state(rng_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        self._alg.train_mode()

        # --- statistics over the first n_eval envs ---
        n = self.n_eval
        succ = self._success[:n]
        n_succ = float(succ.sum().item())
        n_tot = float(n)
        n_fell = float(self._fell[:n].sum().item())

        # torchrun: pool the ranks so every rank decides on the SAME p_hat.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            pooled = torch.tensor([n_succ, n_tot, n_fell], dtype=torch.float64,
                                  device=self._device)
            torch.distributed.all_reduce(pooled, op=torch.distributed.ReduceOp.SUM)
            n_succ, n_tot, n_fell = (float(v) for v in pooled.tolist())

        p_hat = n_succ / max(n_tot, 1.0)
        se = (p_hat * (1.0 - p_hat) / max(n_tot, 1.0)) ** 0.5
        hold_s = self._hold[:n].float() / hz
        ttl = self._ttl_hold[:n]
        ttl_ok = ttl[ttl >= 0].float()

        self.eval_count += 1
        return {
            "p_hat": p_hat,
            "se": se,
            "n_episodes": n_tot,
            "fall_after_stand_rate": n_fell / max(n_tot, 1.0),
            "median_hold_s": float(hold_s.median().item()),
            "mean_hold_s": float(hold_s.mean().item()),
            "mean_episode_s": float(self._ep_len[:n].float().mean().item() / hz),
            "time_to_hold_s": float(ttl_ok.mean().item() / hz) if ttl_ok.numel() else -1.0,
            "reached_hold_rate": float(ttl_ok.numel()) / max(float(n), 1.0),
        }

    def step(self) -> dict[str, float]:
        """Evaluate, apply the advancement decision, and return everything for logging."""
        stats = self.evaluate()
        can_rollback = self._curr.advance_count > 0
        decision = self._policy.decide(stats["p_hat"], can_rollback=can_rollback)

        if decision == EvalAdvancementPolicy.ADVANCE and not self._curr.at_floor:
            self._curr.advance()
        elif decision == EvalAdvancementPolicy.ROLLBACK:
            self._curr.rollback()
        elif decision == EvalAdvancementPolicy.ADVANCE:
            decision = EvalAdvancementPolicy.HOLD  # already at the floor: nothing to do

        stats.update(
            decision=decision,
            lambda_F=self._curr.lambda_F,
            beta=self._curr.beta,
            level=float(self._curr.advance_count),
            required_consecutive=float(self._policy.required_consecutive),
        )
        return stats
