"""Integration probe: run a REAL deterministic evaluation on the from-scratch env (CPU).

Builds Unitree-G1-GetUp-Scratch with a handful of envs, drives DeterministicEvaluator
with a stub actor (zero action = "hold current joint targets", the incremental action
term's no-op), and checks the end-to-end plumbing that the unit tests cannot:

  * every env contributes exactly ONE episode (the sink fires at real terminations);
  * p_hat / SE / the secondary statistics come out well-formed;
  * the curriculum is left UNFROZEN and its rolling diagnostic buffer is untouched;
  * the torch RNG state is restored, and two paired evaluations of the same stub
    policy give the SAME p_hat (the property the anti-fluke guard relies on);
  * the assist force keeps being applied at the current lambda_F during evaluation.

Run:  conda activate unitree_rl_cpu && PYTHONPATH=. python scripts/probe_deterministic_eval.py
"""
import argparse

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500, help="evaluation horizon (steps)")
    args = ap.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg

    import src.tasks  # noqa: F401  (registers the tasks)
    from src.tasks.getup.rl.deterministic_eval import DeterministicEvaluator

    task = "Unitree-G1-GetUp-Scratch"
    env_cfg = load_env_cfg(task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cpu"

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
    env = RslRlVecEnvWrapper(env)
    unwrapped = env.unwrapped

    fs_cfg = getattr(unwrapped, "_fromscratch_cfg")
    curriculum = getattr(unwrapped, "_fs_assist_beta_curriculum")
    eval_cfg = fs_cfg.deterministic_eval
    eval_cfg.max_eval_steps = args.steps

    class StubActor:
        """Deterministic zero-action policy; mimics the actor's call signature."""

        def __call__(self, obs, stochastic_output=False):
            assert stochastic_output is False, "evaluation must use the mean action"
            return torch.zeros(env.num_envs, env.num_actions, device=env.device)

    class StubAlg:
        def __init__(self):
            self.mode = "train"

        def eval_mode(self):
            self.mode = "eval"

        def train_mode(self):
            self.mode = "train"

        def get_policy(self):
            return StubActor()

    alg = StubAlg()
    ev = DeterministicEvaluator(env, alg, eval_cfg, curriculum, "cpu")

    ok = True
    print(f"phase={ev.phase()} interval={ev.interval()} N={ev.n_eval} "
          f"lambda_F={curriculum.lambda_F} beta={curriculum.beta}")

    # RslRlVecEnvWrapper's constructor already reset every env once, so the rolling
    # diagnostic buffer is non-empty before we start: compare against that baseline.
    buffer_before = len(curriculum._buffer)

    torch.manual_seed(1234)
    rng_before = torch.get_rng_state().clone()
    stats = ev.evaluate()
    print("eval 1:", {k: round(v, 4) for k, v in stats.items()})

    rng_ok = bool(torch.equal(torch.get_rng_state(), rng_before))
    print(f"  {'ok ' if rng_ok else 'FAIL'} training RNG state restored")
    ok &= rng_ok

    n_done = int(ev._done_once.sum())
    print(f"  {'ok ' if n_done == env.num_envs else 'FAIL'} every env produced an "
          f"episode: {n_done}/{env.num_envs}")
    ok &= n_done == env.num_envs

    ok &= 0.0 <= stats["p_hat"] <= 1.0 and stats["n_episodes"] == env.num_envs
    print(f"  {'ok ' if 0.0 <= stats['p_hat'] <= 1.0 else 'FAIL'} p_hat in [0,1]")

    frozen = curriculum._frozen
    print(f"  {'ok ' if not frozen else 'FAIL'} curriculum unfrozen after evaluation")
    ok &= not frozen
    buf_ok = len(curriculum._buffer) == buffer_before
    print(f"  {'ok ' if buf_ok else 'FAIL'} diagnostic buffer untouched by evaluation "
          f"({buffer_before} -> {len(curriculum._buffer)} entries)")
    ok &= buf_ok
    print(f"  {'ok ' if alg.mode == 'train' else 'FAIL'} algorithm back in train mode")
    ok &= alg.mode == "train"

    # Paired: same stub policy + same seed => identical p_hat.
    stats2 = ev.evaluate()
    same = abs(stats2["p_hat"] - stats["p_hat"]) < 1e-9
    print(f"  {'ok ' if same else 'FAIL'} paired evaluations reproduce p_hat: "
          f"{stats['p_hat']:.4f} vs {stats2['p_hat']:.4f}")
    ok &= same

    force = float(getattr(unwrapped, "_assistance_force")[0])
    print(f"  {'ok ' if force == curriculum.lambda_F else 'FAIL'} assist force still at "
          f"the current level during/after evaluation: {force}")
    ok &= force == curriculum.lambda_F

    # And the full decision path.
    out = ev.step()
    print("eval 3 + decision:", {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in out.items()})

    env.close()
    print("\nPROBE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
