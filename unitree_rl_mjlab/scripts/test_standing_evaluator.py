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
    set_state(a2, h=0.72, up=0.98, v=0.0, w=0.0, qd=0.0)  # strict-stable
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
    set_state(a3, h=0.72, up=0.98)
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

    # --- each strict-stable gate independently blocks stable_now (isolate one at a time) ---
    def steps_to_hold(setter, n=60):
        e = FakeEnv(); a = e.scene["robot"]; E.get_episode_state(e, a)
        setter(a)
        reached = None
        for i in range(n):
            e.common_step_counter = i
            st = S.compute_standing_status(e, a, cfg)
            if st["stable_hold_reached"].all() and reached is None:
                reached = i
        return reached

    # baseline: fully-good state DOES reach hold within 60 steps.
    good = lambda a: set_state(a, h=0.72, up=0.98, v=0.0, w=0.0, qd=0.0)
    r_good = steps_to_hold(good)
    print(f"baseline good state reaches hold: {r_good is not None} (expect True)")
    ok &= (r_good is not None)

    # upright just below cos(12°)=0.978 blocks the hold (h/vel fine).
    r_up = steps_to_hold(lambda a: set_state(a, h=0.72, up=0.97, v=0.0, w=0.0, qd=0.0))
    print(f"upright=0.97 (<0.978 limit) blocks hold: {r_up is None} (expect True)")
    ok &= (r_up is None)

    # relative height just below 0.90*0.75=0.675 blocks the hold.
    r_h = steps_to_hold(lambda a: set_state(a, h=0.66, up=0.98, v=0.0, w=0.0, qd=0.0))
    print(f"h=0.66 (<0.675 limit) blocks hold: {r_h is None} (expect True)")
    ok &= (r_h is None)

    # lin-vel over threshold (0.15) blocks the hold even though h/upright are fine.
    r_v = steps_to_hold(lambda a: set_state(a, h=0.72, up=0.98, v=0.40, w=0.0, qd=0.0))
    print(f"lin_vel=0.40 (>0.15 limit) blocks hold: {r_v is None} (expect True)")
    ok &= (r_v is None)

    # horizontal ang-vel over threshold (0.35) blocks the hold.
    r_w = steps_to_hold(lambda a: set_state(a, h=0.72, up=0.98, v=0.0, w=1.0, qd=0.0))
    print(f"ang_vel_xy=1.0 (>0.35 limit) blocks hold: {r_w is None} (expect True)")
    ok &= (r_w is None)

    # RMS(qdot) is DEFERRED out of χ_t: a high joint velocity must NOT block the hold.
    r_qd = steps_to_hold(lambda a: set_state(a, h=0.72, up=0.98, v=0.0, w=0.0, qd=5.0))
    print(f"qd_rms=5.0 (gate REMOVED) does NOT block hold: {r_qd is not None} (expect True)")
    ok &= (r_qd is not None)

    # feet not planted blocks the hold even with everything else perfect.
    e5 = FakeEnv(); a5 = e5.scene["robot"]; E.get_episode_state(e5, a5)
    set_state(a5, h=0.72, up=0.98)
    S._feet_planted = lambda e, c: torch.zeros(B, dtype=torch.bool)
    r_feet = None
    for i in range(60):
        e5.common_step_counter = i
        st = S.compute_standing_status(e5, a5, cfg)
        if st["stable_hold_reached"].all() and r_feet is None:
            r_feet = i
    print(f"feet not planted blocks hold: {r_feet is None} (expect True)")
    ok &= (r_feet is None)
    S._feet_planted = lambda e, c: torch.ones(B, dtype=torch.bool)  # restore for later cases

    # capture point is DEFERRED out of χ_t: even a wildly-outside margin must NOT
    # block the hold (balance is now carried by the separate com_over_support reward).
    r_cp = None
    e6 = FakeEnv(); a6 = e6.scene["robot"]; E.get_episode_state(e6, a6)
    set_state(a6, h=0.72, up=0.98)
    S._capture_point_margin = lambda e, a: torch.full((B,), -0.50)  # way outside
    for i in range(60):
        e6.common_step_counter = i
        st = S.compute_standing_status(e6, a6, cfg)
        if st["stable_hold_reached"].all() and r_cp is None:
            r_cp = i
    print(f"capture-point margin=-0.50 (gate REMOVED) does NOT block hold: {r_cp is not None} (expect True)")
    ok &= (r_cp is not None)
    S._capture_point_margin = lambda e, a: torch.full((B,), 0.10)  # restore

    # --- episode reset clears the shared counters (no leakage into the next episode) ---
    e7 = FakeEnv(); a7 = e7.scene["robot"]; E.get_episode_state(e7, a7)
    set_state(a7, h=0.72, up=0.98)
    for i in range(30):
        e7.common_step_counter = i
        S.compute_standing_status(e7, a7, cfg)
    pre_reset_counter = int(get_state(e7)["stable_counter"][0])
    E.reset_episode_state(e7, env_ids=torch.arange(B))
    post_reset_counter = int(get_state(e7)["stable_counter"][0])
    post_reset_latched = bool(get_state(e7)["standing_latched"][0])
    post_reset_first_stable = int(get_state(e7)["first_stable_step"][0])
    print(f"stable_counter before/after reset: {pre_reset_counter}/{post_reset_counter} "
          f"(expect 30/0), latched={post_reset_latched} (expect False), "
          f"first_stable_step={post_reset_first_stable} (expect -1)")
    ok &= (pre_reset_counter == 30 and post_reset_counter == 0
           and not post_reset_latched and post_reset_first_stable == -1)

    # --- fs_progressive_hold / fs_success_bonus reward values track the counter exactly ---
    from src.tasks.getup.mdp import rewards_fromscratch as RF
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    robot_cfg = SceneEntityCfg("robot")
    e8 = FakeEnv(); a8 = e8.scene["robot"]; E.get_episode_state(e8, a8)
    set_state(a8, h=0.72, up=0.98)
    bonus_step = None
    frac_at_25, frac_at_50 = None, None
    for i in range(55):
        e8.common_step_counter = i
        frac = RF.fs_progressive_hold(e8, asset_cfg=robot_cfg)
        bonus = RF.fs_success_bonus(e8, asset_cfg=robot_cfg)
        if i == 24:
            frac_at_25 = float(frac[0])
        if i == 49:
            frac_at_50 = float(frac[0])
        if bool(bonus[0] > 0) and bonus_step is None:
            bonus_step = i
    print(f"fs_progressive_hold at step 25/50 counter: {frac_at_25:.2f}/{frac_at_50:.2f} "
          f"(expect 0.50/1.00), fs_success_bonus fires at step {bonus_step} (expect 49)")
    ok &= (abs(frac_at_25 - 0.50) < 1e-6 and abs(frac_at_50 - 1.00) < 1e-6 and bonus_step == 49)

    print("\nSTANDING EVALUATOR TEST:", "PASS" if ok else "FAIL")
    return ok


def get_state(env):
    return getattr(env, E._EPISODE_STATE_ATTR)


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
