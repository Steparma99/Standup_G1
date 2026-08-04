"""Unit test for the from-scratch GLOBAL coupled assist-force + beta curriculum.

Exercises the advancement LOGIC deterministically with a fake env/asset (no physics).
Success is now a sustained STABLE HOLD: success_i = 1[stable_counter >= success_window]
(== the χ_t product over the final window), so the shared standing evaluator is
monkeypatched to feed a controllable stable_counter.

Checks:
  * buffer fills to K over successive episode terminations;
  * an advance fires ONLY when the buffer is FULL and its mean >= tau_s;
  * an advance steps BOTH lambda_F and beta down by their deltas, then CLEARS the
    buffer (next advance needs another full K episodes);
  * lambda_F / beta respect their floors;
  * a below-threshold full buffer does NOT advance (and does not clear);
  * success uses the stable_counter cached at the LAST step before reset;
  * a hold SHORTER than success_window is a failure;
  * the global scalars are mirrored into the per-env broadcast tensors read by the
    action term / metrics (_BETA_RESCALER_ATTR / _ASSIST_FORCE_ATTR);
  * the assist wrench is suppressed during the settle window.
"""
import torch
from collections import namedtuple

import src.tasks.getup.mdp.standing as S
from src.tasks.getup.mdp.events import (
    FromScratchAssistBetaCurriculum,
    _BETA_RESCALER_ATTR,
    _ASSIST_FORCE_ATTR,
)
from src.tasks.getup.curriculum_cfg import CoupledAdvancementCfg

B = 8
DEV = "cpu"
CfgShim = namedtuple("CfgShim", ["params"])

# Monkeypatch the shared evaluator: __call__ caches status["stable_counter"], so we
# feed it env._fake_stable_counter and a dummy cfg with a .stability attribute.
S.compute_standing_status = lambda env, asset, cfg: {"stable_counter": env._fake_stable_counter}
S.get_curriculum_cfg = lambda env: type("C", (), {"stability": None})()


class FakeAsset:
    def __init__(self):
        self.data = type("D", (), {})()
        self._wrench_calls = []

    def find_bodies(self, name):
        return [0], [name]

    def write_external_wrench_to_sim(self, forces, torques, env_ids=None, body_ids=None):
        if forces.numel():
            self._wrench_calls.append(float(forces[:, 0, 2].max()))


class FakeEnv:
    def __init__(self):
        self.num_envs = B
        self.device = DEV
        self.scene = {"robot": FakeAsset()}
        self.episode_length_buf = torch.full((B,), 100, dtype=torch.long)  # past settle
        self._fake_stable_counter = torch.zeros(B, dtype=torch.long)


def make(cfg_over=None):
    co = CoupledAdvancementCfg(**(cfg_over or {}))
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
    cur = FromScratchAssistBetaCurriculum(CfgShim(params=params), env)
    return env, cur, co


def episode_end(env, cur, counter_val):
    """One control step with all envs at stable_counter=counter_val, then reset all B."""
    env._fake_stable_counter = torch.full((B,), counter_val, dtype=torch.long)
    cur(env)                       # __call__ caches _last_stable_counter
    cur.reset(torch.arange(B))


def run():
    ok = True
    # success_window default = 50; counter 60 -> success, counter 10 -> fail.
    HOLD, SHORT = 60, 10

    # --- 1. all-success episodes advance BOTH once the buffer fills, then clear ----
    env, cur, co = make({"window_K": 16})
    assert co.window_K == 16 and co.success_window_steps == 50
    ok &= bool((getattr(env, _BETA_RESCALER_ATTR) == co.beta_init).all())
    ok &= bool((getattr(env, _ASSIST_FORCE_ATTR) == co.lambda_F_init).all())

    episode_end(env, cur, HOLD)   # 8 successes (<16) -> no advance
    print(f"after 1 reset: fill={cur.buffer_fill:.2f} adv={cur._advance_count} "
          f"lambda_F={cur._lambda_F} beta={cur._beta:.3f} (expect fill=0.5 adv=0)")
    ok &= (abs(cur.buffer_fill - 0.5) < 1e-9 and cur._advance_count == 0
           and cur._lambda_F == 200.0 and abs(cur._beta - 1.0) < 1e-9)

    episode_end(env, cur, HOLD)   # buffer 16 full, mean 1.0 -> ADVANCE + CLEAR
    print(f"after 2 resets: fill={cur.buffer_fill:.2f} adv={cur._advance_count} "
          f"lambda_F={cur._lambda_F} beta={cur._beta:.3f} (expect fill=0 adv=1 190 0.980)")
    ok &= (cur.buffer_fill == 0.0 and cur._advance_count == 1
           and cur._lambda_F == 190.0 and abs(cur._beta - 0.98) < 1e-9)
    ok &= bool((getattr(env, _BETA_RESCALER_ATTR) == cur._beta).all())
    ok &= bool((getattr(env, _ASSIST_FORCE_ATTR) == cur._lambda_F).all())

    # --- 2. a below-threshold FULL buffer does NOT advance and does NOT clear ------
    env2, cur2, _ = make({"window_K": 16})
    episode_end(env2, cur2, HOLD)    # 8 success
    episode_end(env2, cur2, SHORT)   # 8 fail -> full, mean 0.5 < 0.85
    print(f"below-thresh full buffer: fill={cur2.buffer_fill:.2f} rate={cur2.success_rate:.2f} "
          f"adv={cur2._advance_count} (expect fill=1.0 rate=0.5 adv=0)")
    ok &= (cur2.buffer_fill == 1.0 and abs(cur2.success_rate - 0.5) < 1e-9
           and cur2._advance_count == 0)
    episode_end(env2, cur2, HOLD)    # last16 = 8 fail + 8 success -> 0.5, still no advance
    print(f"  +8 success: rate={cur2.success_rate:.2f} adv={cur2._advance_count} (expect 0.5, 0)")
    ok &= (abs(cur2.success_rate - 0.5) < 1e-9 and cur2._advance_count == 0)
    episode_end(env2, cur2, HOLD)    # last16 all success -> advance
    print(f"  +8 success again: fill={cur2.buffer_fill:.2f} adv={cur2._advance_count} (expect 0, 1)")
    ok &= (cur2._advance_count == 1 and cur2.buffer_fill == 0.0)

    # --- 3. floors: many advances clamp lambda_F at 0 and beta at 0.25 ------------
    env3, cur3, co3 = make({"window_K": 8})
    for _ in range(100):
        episode_end(env3, cur3, HOLD)
    print(f"after 100 all-success advances: lambda_F={cur3._lambda_F} beta={cur3._beta:.3f} "
          f"adv={cur3._advance_count} (expect 0.0 and 0.25)")
    ok &= (cur3._lambda_F == co3.lambda_F_min and abs(cur3._beta - co3.beta_min) < 1e-9)
    ok &= bool((getattr(env3, _BETA_RESCALER_ATTR) == co3.beta_min).all())
    ok &= bool((getattr(env3, _ASSIST_FORCE_ATTR) == co3.lambda_F_min).all())

    # --- 4. exactly-at-threshold (rate == tau_s) DOES advance (>= inclusive) -------
    env4, cur4, _ = make({"window_K": 20})
    cur4._buffer.extend([1.0] * 16 + [0.0] * 3)   # 19 entries, not full
    env4._fake_stable_counter = torch.full((B,), HOLD, dtype=torch.long)
    cur4(env4)                                    # cache a success
    cur4.reset(torch.tensor([0], dtype=torch.long))  # +1 success -> 17/20 = 0.85
    print(f"rate==tau_s (0.85): adv={cur4._advance_count} (expect adv=1)")
    ok &= (cur4._advance_count == 1)

    # --- 5. success uses the LAST cached stable_counter (pre-reset), not a fresh one -
    env5, cur5, _ = make({"window_K": 8})
    env5._fake_stable_counter = torch.full((B,), HOLD, dtype=torch.long)
    cur5(env5)                                    # caches 60 (a real hold)
    env5._fake_stable_counter = torch.zeros(B, dtype=torch.long)  # reset_pose would zero it
    cur5.reset(torch.arange(B))                   # must judge on the CACHED 60 -> success
    print(f"cached-counter success: adv={cur5._advance_count} (expect 1)")
    ok &= (cur5._advance_count == 1)

    # --- 6. a hold SHORTER than success_window is a FAILURE (no advance) -----------
    env6, cur6, _ = make({"window_K": 8})
    for _ in range(4):
        episode_end(env6, cur6, 49)   # 49 < 50 window -> all failures
    print(f"49-step holds (<50 window): fill={cur6.buffer_fill:.2f} rate={cur6.success_rate:.2f} "
          f"adv={cur6._advance_count} (expect rate=0 adv=0)")
    ok &= (cur6.success_rate == 0.0 and cur6._advance_count == 0)

    # --- 7. assist wrench is suppressed during the settle window ------------------
    env7, cur7, co7 = make({"window_K": 8})
    env7._fake_stable_counter = torch.zeros(B, dtype=torch.long)
    env7.episode_length_buf = torch.zeros(B, dtype=torch.long)  # inside settle
    env7.scene["robot"]._wrench_calls.clear()
    cur7(env7)
    settle_force = env7.scene["robot"]._wrench_calls[-1]
    env7.episode_length_buf = torch.full((B,), co7.assist_unactuated_steps, dtype=torch.long)
    cur7(env7)
    active_force = env7.scene["robot"]._wrench_calls[-1]
    print(f"assist force during settle={settle_force} vs active={active_force} "
          f"(expect 0.0 then {co7.lambda_F_init})")
    ok &= (settle_force == 0.0 and active_force == co7.lambda_F_init)

    print("\nCOUPLED CURRICULUM TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
