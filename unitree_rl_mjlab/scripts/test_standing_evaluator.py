"""Unit test for the shared standing evaluator counter logic (section 17: timing test).

Uses a lightweight fake env/asset + monkeypatched feet/capture-point so we exercise the
COUNTER logic deterministically (candidate latch @10 steps, stable hold @50 not 49,
fall-after-stand @20 post-latch bad steps) without driving full physics.
"""
import torch
import src.tasks.getup.mdp.standing as S
from src.tasks.getup.mdp import events as E
from src.tasks.getup.curriculum_cfg import StabilityCfg

B = 3
DEV = "cpu"


class FakeData:
    def __init__(self):
        self.root_link_pos_w = torch.zeros(B, 3)
        self.projected_gravity_b = torch.zeros(B, 3)
        self.root_link_lin_vel_b = torch.zeros(B, 3)
        self.root_link_ang_vel_b = torch.zeros(B, 3)
        self.joint_vel = torch.zeros(B, 29)


class FakeAsset:
    def __init__(self):
        self.data = FakeData()
        self.joint_names = [f"j{i}_shoulder_pitch_joint" for i in range(29)]


class FakeEnv:
    def __init__(self):
        self.num_envs = B
        self.device = DEV
        self.common_step_counter = 0
        self.scene = {"robot": FakeAsset()}


def set_state(asset, h, up, v=0.0, w=0.0, qd=0.0):
    asset.data.root_link_pos_w[:, 2] = h
    asset.data.projected_gravity_b[:, 2] = -up
    asset.data.root_link_lin_vel_b[:] = 0.0
    asset.data.root_link_lin_vel_b[:, 0] = v
    asset.data.root_link_ang_vel_b[:] = 0.0
    asset.data.root_link_ang_vel_b[:, 0] = w
    asset.data.joint_vel[:] = qd


def run():
    cfg = StabilityCfg()
    env = FakeEnv()
    asset = env.scene["robot"]
    # patch feet + capture point to "always OK" so we isolate the counter logic
    S._feet_planted = lambda e, c: torch.ones(B, dtype=torch.bool)
    S._capture_point_margin = lambda e, a: torch.full((B,), 0.10)
    S._body_joint_mask = lambda e, a: torch.ones(29, dtype=torch.bool)
    # seed episode state
    E.get_episode_state(env, asset)

    ok = True

    # --- candidate latch after exactly 10 consecutive candidate steps ---
    set_state(asset, h=0.70, up=0.93)   # candidate but NOT strict-stable (h<0.68? no 0.70>0.68; up 0.93<0.95 -> not stable)
    latched_step = None
    for i in range(15):
        env.common_step_counter = i
        st = S.compute_standing_status(env, asset, cfg)
        if st["standing_latched"].all() and latched_step is None:
            latched_step = i
    print(f"candidate latched at step index {latched_step} (expect 9 = the 10th step)")
    ok &= (latched_step == 9)

    # --- stable hold: 49 steps must NOT reach, 50th must ---
    env2 = FakeEnv(); a2 = env2.scene["robot"]; E.get_episode_state(env2, a2)
    set_state(a2, h=0.72, up=0.97, v=0.0, w=0.0, qd=0.0)  # strict-stable
    reached_at = None
    for i in range(60):
        env2.common_step_counter = i
        st = S.compute_standing_status(env2, a2, cfg)
        n = int(st["stable_counter"][0])
        if st["stable_hold_reached"].all() and reached_at is None:
            reached_at = (i, n)
    print(f"stable_hold_reached first at (step_idx, stable_counter)={reached_at} (expect counter==50)")
    ok &= (reached_at is not None and reached_at[1] == 50)
    # verify 49 consecutive did not trigger: re-run to step 48 fresh
    env3 = FakeEnv(); a3 = env3.scene["robot"]; E.get_episode_state(env3, a3)
    set_state(a3, h=0.72, up=0.97)
    for i in range(49):  # 49 steps -> counter 49
        env3.common_step_counter = i
        st = S.compute_standing_status(env3, a3, cfg)
    print(f"after 49 stable steps: counter={int(st['stable_counter'][0])} reached={bool(st['stable_hold_reached'][0])} (expect 49, False)")
    ok &= (int(st["stable_counter"][0]) == 49 and not bool(st["stable_hold_reached"][0]))

    # --- fall-after-stand after 20 post-latch bad steps ---
    env4 = FakeEnv(); a4 = env4.scene["robot"]; E.get_episode_state(env4, a4)
    set_state(a4, h=0.70, up=0.93)
    for i in range(12):  # latch
        env4.common_step_counter = i; S.compute_standing_status(env4, a4, cfg)
    set_state(a4, h=0.50, up=0.60)  # collapse
    fell_at = None
    for i in range(12, 12 + 25):
        env4.common_step_counter = i
        st = S.compute_standing_status(env4, a4, cfg)
        if st["fall_after_stand"].all() and fell_at is None:
            fell_at = i - 12
    print(f"fall_after_stand triggered after {fell_at} bad post-latch steps (expect 19 = 20th step)")
    ok &= (fell_at == 19)

    print("\nSTANDING EVALUATOR TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
