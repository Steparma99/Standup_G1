"""Unit test for the deterministic-evaluation advancement pipeline.

Covers, with no physics:
  * EvalAdvancementPolicy: base rule, the anti-fluke guard (p_prev < tau_s - 0.08
    requires a confirming second evaluation), rollback after consecutive collapses,
    and the post-rollback re-advance confirmation;
  * the phase-dependent evaluation interval (A: lambda_F>0 -> 50, B: beta-only -> 75,
    C: at the floor -> 100);
  * the curriculum's advance()/rollback() level arithmetic + broadcast tensors;
  * freeze(): during an evaluation, episode ends go to the sink and NEITHER the rolling
    diagnostic buffer NOR the level moves;
  * set_autonomous(False): a FULL, above-threshold K-window no longer advances.
"""
import torch
from collections import namedtuple

import src.tasks.getup.mdp.standing as S
from src.tasks.getup.curriculum_cfg import (
    CoupledAdvancementCfg,
    DeterministicEvalCfg,
)

B = 8
CfgShim = namedtuple("CfgShim", ["params"])

S.compute_standing_status = lambda env, asset, cfg: {
    "stable_counter": env._fake_stable_counter,
    "fall_after_stand": torch.zeros(B, dtype=torch.bool),
    "first_stable_step": torch.full((B,), -1, dtype=torch.long),
}
S.get_curriculum_cfg = lambda env: type("C", (), {"stability": None})()

from src.tasks.getup.mdp.events import (  # noqa: E402  (after the monkeypatch)
    FromScratchAssistBetaCurriculum,
    _ASSIST_FORCE_ATTR,
    _BETA_RESCALER_ATTR,
)
from src.tasks.getup.rl.deterministic_eval import (  # noqa: E402
    DeterministicEvaluator,
    EvalAdvancementPolicy,
)


class FakeAsset:
    def __init__(self):
        self.data = type("D", (), {})()

    def find_bodies(self, name):
        return [0], [name]

    def write_external_wrench_to_sim(self, forces, torques, env_ids=None, body_ids=None):
        pass


class FakeEnv:
    def __init__(self):
        self.num_envs = B
        self.device = "cpu"
        self.scene = {"robot": FakeAsset()}
        self.episode_length_buf = torch.full((B,), 100, dtype=torch.long)
        self.common_step_counter = 0
        self._fake_stable_counter = torch.zeros(B, dtype=torch.long)


def make_curriculum(**over):
    co = CoupledAdvancementCfg(**over)
    params = {
        "asset_cfg": type("A", (), {"name": "robot"})(),
        "head_body": co.head_body,
        "success_window_steps": co.success_window_steps,
        "window_K": co.window_K,
        "success_rate_threshold": co.success_rate_threshold,
        "lambda_F_init": co.lambda_F_init,
        "lambda_F_min": co.lambda_F_min,
        "delta_lambda_F": co.delta_lambda_F,
        "beta_init": co.beta_init,
        "beta_min": co.beta_min,
        "delta_beta": co.delta_beta,
        "assist_unactuated_steps": co.assist_unactuated_steps,
    }
    env = FakeEnv()
    return env, FromScratchAssistBetaCurriculum(CfgShim(params=params), env), co


def check(label, got, want):
    good = got == want
    print(f"  {'ok ' if good else 'FAIL'} {label}: got {got!r}, want {want!r}")
    return good


def test_policy():
    print("\n[1] EvalAdvancementPolicy — base rule + anti-fluke guard")
    ok = True
    cfg = DeterministicEvalCfg(rollback_enabled=False)
    P = EvalAdvancementPolicy(cfg)

    # first evaluation, no history: >= tau_s advances immediately
    ok &= check("first eval 0.90", P.decide(0.90), "advance")
    # a normal dip then recovery from just below tau_s (0.80 > 0.85-0.08) -> advance
    ok &= check("dip to 0.80", P.decide(0.80), "hold")
    ok &= check("back to 0.86 (dip was shallow)", P.decide(0.86), "advance")
    # a DEEP dip: recovery must be confirmed by a second evaluation
    ok &= check("deep dip to 0.60", P.decide(0.60), "hold")
    ok &= check("0.95 right after deep dip (fluke guard)", P.decide(0.95), "hold")
    ok &= check("0.95 confirmed", P.decide(0.95), "advance")
    # exactly at threshold advances (>= is inclusive)
    ok &= check("exactly tau_s", P.decide(0.85), "advance")
    # exactly at the guard boundary (tau_s - 0.08 = 0.77) is NOT "far below"
    P2 = EvalAdvancementPolicy(cfg)
    P2.decide(0.77)
    ok &= check("recovery after p_prev == tau_s-delta", P2.decide(0.90), "advance")
    P3 = EvalAdvancementPolicy(cfg)
    P3.decide(0.7699)
    ok &= check("recovery after p_prev just under the guard", P3.decide(0.90), "hold")
    return ok


def test_rollback():
    print("\n[2] EvalAdvancementPolicy — rollback + re-advance confirmation")
    ok = True
    P = EvalAdvancementPolicy(DeterministicEvalCfg())
    ok &= check("collapse 1/2", P.decide(0.10), "hold")
    ok &= check("collapse 2/2 -> rollback", P.decide(0.05), "rollback")
    ok &= check("recovery needs confirming (1/2)", P.decide(0.90), "hold")
    ok &= check("recovery confirmed (2/2)", P.decide(0.92), "advance")
    # a single bad evaluation between two collapses resets the streak
    P2 = EvalAdvancementPolicy(DeterministicEvalCfg())
    P2.decide(0.10)
    ok &= check("0.50 breaks the collapse streak", P2.decide(0.50), "hold")
    ok &= check("one collapse alone does not roll back", P2.decide(0.10), "hold")
    # rollback is suppressed when there is no level to roll back to
    P3 = EvalAdvancementPolicy(DeterministicEvalCfg())
    P3.decide(0.10)
    ok &= check("collapse at level 0 (can_rollback=False)",
                P3.decide(0.05, can_rollback=False), "hold")
    return ok


def test_levels_and_phases():
    print("\n[3] curriculum advance()/rollback() + phase-dependent interval")
    ok = True
    env, cur, co = make_curriculum()
    ecfg = DeterministicEvalCfg()
    ev = DeterministicEvaluator(env, alg=None, cfg=ecfg, curriculum=cur, device="cpu")

    ok &= check("K-window demoted", cur._autonomous, False)
    ok &= check("N defaults to all envs", ev.n_eval, B)
    ok &= check("phase A at start", ev.phase(), "A")
    ok &= check("interval A", ev.interval(), 50)

    cur.advance()
    ok &= check("lambda_F after 1 advance", cur.lambda_F, 190.0)
    ok &= check("beta after 1 advance", round(cur.beta, 3), 0.98)
    ok &= check("broadcast beta", round(float(getattr(env, _BETA_RESCALER_ATTR)[0]), 3),
                round(cur.beta, 3))
    ok &= check("broadcast force", float(getattr(env, _ASSIST_FORCE_ATTR)[0]), cur.lambda_F)

    cur.rollback()
    ok &= check("rollback restores lambda_F", cur.lambda_F, 200.0)
    ok &= check("rollback restores beta", round(cur.beta, 3), 1.0)
    ok &= check("rollback clamps at the initial level", cur.rollback()[0], 200.0)

    # 20 advances zero the force but leave beta mid-range -> phase B (beta only)
    for _ in range(20):
        cur.advance()
    ok &= check("lambda_F at floor after 20 advances", cur.lambda_F, 0.0)
    ok &= check("beta after 20 advances", round(cur.beta, 3), 0.6)
    ok &= check("phase B", ev.phase(), "B")
    ok &= check("interval B", ev.interval(), 75)

    for _ in range(18):
        cur.advance()
    ok &= check("beta at floor", round(cur.beta, 3), co.beta_min)
    ok &= check("phase C", ev.phase(), "C")
    ok &= check("interval C", ev.interval(), 100)
    ok &= check("at_floor", cur.at_floor, True)
    return ok


def test_freeze():
    print("\n[4] freeze(): evaluation episodes reach the sink, not the buffer/level")
    ok = True
    env, cur, _ = make_curriculum(window_K=8)
    cur.set_autonomous(False)   # what the evaluator does when it takes over
    captured = []

    def sink(env_ids, succ, info):
        captured.append((env_ids.tolist(), succ.tolist(), info))

    env._fake_stable_counter = torch.full((B,), 60, dtype=torch.long)
    cur.freeze(sink)
    for _ in range(4):                      # 32 all-success episode ends
        cur(env)
        cur.reset(torch.arange(B))
    ok &= check("sink saw every episode end", len(captured), 4)
    ok &= check("successes recorded", captured[0][1], [True] * B)
    ok &= check("rolling buffer untouched", len(cur._buffer), 0)
    ok &= check("level untouched", (cur.lambda_F, cur.beta), (200.0, 1.0))
    ok &= check("snapshot carries the episode length",
                int(cur.latest_episode_snapshot()["episode_length"][0]), 100)

    cur.unfreeze()
    cur(env)
    cur.reset(torch.arange(B))
    ok &= check("buffer fills again once unfrozen", len(cur._buffer), B)
    ok &= check("but a full above-tau window still does NOT advance (demoted)",
                cur.advance_count, 0)
    cur(env)
    cur.reset(torch.arange(B))              # buffer now full (16 >= K=8), rate 1.0
    ok &= check("still no autonomous advance", cur.advance_count, 0)
    ok &= check("diagnostic success rate still reported", cur.success_rate, 1.0)
    return ok


def test_sink_first_episode_only():
    print("\n[5] evaluator sink counts each env's FIRST episode only")
    ok = True
    env, cur, _ = make_curriculum()
    ev = DeterministicEvaluator(env, alg=None, cfg=DeterministicEvalCfg(),
                               curriculum=cur, device="cpu")
    ev._reset_records()
    ids = torch.arange(B)
    info = {
        "stable_counter": torch.full((B,), 60, dtype=torch.long),
        "fall_after_stand": torch.zeros(B, dtype=torch.bool),
        "first_stable_step": torch.full((B,), 140, dtype=torch.long),
        "episode_length": torch.full((B,), 200, dtype=torch.long),
        "common_step": 300,
    }
    ev._record(ids, torch.ones(B, dtype=torch.bool), info)
    ok &= check("first episode recorded as success", ev._success.tolist(), [True] * B)
    # in-episode step of the first hold = 200 - (300 - 140) = 40
    ok &= check("time-to-hold converted to in-episode steps", int(ev._ttl_hold[0]), 40)

    info2 = dict(info, stable_counter=torch.zeros(B, dtype=torch.long))
    ev._record(ids, torch.zeros(B, dtype=torch.bool), info2)
    ok &= check("second episode of the same env ignored", ev._success.tolist(), [True] * B)

    ev._reset_records()
    ev._record(torch.tensor([0, 1]), torch.tensor([True, False]),
               {k: (v[:2] if torch.is_tensor(v) else v) for k, v in info.items()})
    ok &= check("partial env_ids handled",
                (ev._done_once.tolist().count(True), ev._success[:2].tolist()),
                (2, [True, False]))
    return ok


def run():
    ok = True
    ok &= test_policy()
    ok &= test_rollback()
    ok &= test_levels_and_phases()
    ok &= test_freeze()
    ok &= test_sink_first_episode_only()
    print("\nDETERMINISTIC EVAL TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
