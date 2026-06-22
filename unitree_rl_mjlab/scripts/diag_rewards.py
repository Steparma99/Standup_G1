"""Probe: evaluate EVERY get-up reward term on each reference pose.

Purpose: verify that each reward term rewards / penalizes the pose you expect — i.e.
that you are not penalizing something wrong. The robot is teleported to each canonical
keyframe (HOME standing, KNEES_BENT, SUPINE, PRONE, SIDE_*, SEATED), the sim is
forwarded to populate derived state, and then every reward term registered in the env
is evaluated with its REAL params + weight (read straight from the RewardManager, so
what you see is exactly what training uses).

It prints two things:
  1. A full table: one row per reward term, one column per pose, showing the
     weighted contribution (raw_value * weight). Read each row left-to-right and ask
     "is this the pose where I want this term high / low / penalizing?".
  2. A focused ANKLE-PARALLEL verification: the raw per-foot tilt metric and fired
     status for every pose, PLUS a tilt sweep on the HOME pose (progressively tilting
     the ankles in pitch and roll) so you can SEE exactly where the +20 turns on/off
     and recalibrate `tilt_threshold` if 0.05 is not where you want it.

Run:
  cd unitree_rl_mjlab
  CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl python scripts/diag_rewards.py
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
from prettytable import PrettyTable

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.string import resolve_expr

import src.tasks.getup.mdp as mdp
from src.tasks.getup.config.g1.env_cfgs import unitree_g1_getup_env_cfg
from src.assets.robots.unitree_g1.g1_constants import (
    HOME_KEYFRAME,
    KNEES_BENT_KEYFRAME,
    PRONE_KEYFRAME,
    SEATED_KEYFRAME,
    SIDE_LEFT_KEYFRAME,
    SIDE_RIGHT_KEYFRAME,
    SUPINE_KEYFRAME,
)

# Ordered so the "feet-flat standing" poses (HOME, KNEES_BENT) sit first — the ankle
# parallel reward should fire on those and NOT on the lying poses to their right.
POSES = {
    "HOME(stand)": HOME_KEYFRAME,
    "KNEES_BENT": KNEES_BENT_KEYFRAME,
    "SEATED": SEATED_KEYFRAME,
    "SUPINE": SUPINE_KEYFRAME,
    "PRONE": PRONE_KEYFRAME,
    "SIDE_LEFT": SIDE_LEFT_KEYFRAME,
    "SIDE_RIGHT": SIDE_RIGHT_KEYFRAME,
}

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _write_poses(env: ManagerBasedRlEnv) -> None:
    """Teleport env i to POSES[i] and forward the sim."""
    asset = env.scene["robot"]
    names = asset.joint_names
    env.reset()
    for i, kf in enumerate(POSES.values()):
        pos = tuple(getattr(kf, "pos", None) or (0.0, 0.0, 0.0))
        rot = tuple(getattr(kf, "rot", None) or (1.0, 0.0, 0.0, 0.0))
        root = torch.tensor([[*pos, *rot]], dtype=torch.float32)
        jvec = torch.tensor(
            [resolve_expr(getattr(kf, "joint_pos", {}) or {}, names, 0.0)],
            dtype=torch.float32,
        )
        ids = torch.tensor([i])
        asset.write_root_link_pose_to_sim(root, env_ids=ids)
        asset.write_joint_state_to_sim(jvec, torch.zeros_like(jvec), env_ids=ids)
    env.sim.forward()


def _foot_tilt_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Mean over both feet of ||proj_grav_xy||^2 — the ankle-parallel flatness metric."""
    asset = env.scene["robot"]
    ids = torch.tensor([asset.find_bodies(n)[0][0] for n in FOOT_BODIES])
    quat = asset.data.body_link_quat_w[:, ids, :]
    g = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, quat.shape[1], -1)
    pg = quat_apply_inverse(quat, g)
    return pg[..., :2].pow(2).sum(dim=-1).mean(dim=1)  # [B]


def all_terms_table(env: ManagerBasedRlEnv) -> None:
    """Evaluate every reward term on every pose; print weighted contributions."""
    _write_poses(env)
    rm = env.reward_manager
    pose_names = list(POSES.keys())

    table = PrettyTable()
    table.field_names = ["reward term", "weight", *pose_names]
    table.align = "r"
    table.align["reward term"] = "l"

    for name, cfg in zip(rm._term_names, rm._term_cfgs, strict=False):
        try:
            raw = cfg.func(env, **cfg.params)  # [B]
        except Exception as e:  # noqa: BLE001
            table.add_row([name, cfg.weight, *["ERR"] * len(pose_names)])
            print(f"  [warn] {name}: {e}")
            continue
        weighted = (raw * cfg.weight).tolist()
        table.add_row(
            [name, f"{cfg.weight:g}", *[f"{v:+.3f}" for v in weighted]]
        )

    print("\n=== Weighted reward contribution per pose (raw_value * weight) ===")
    print(table)
    print(
        "Read each row L->R: positive = reward, negative = penalty. Check the high/low\n"
        "cells land on the poses you intend. (Stage-gated terms read 0 off their stage;\n"
        "progress terms like height_progress read ~0 in a static pose — both expected.)"
    )


def ankle_parallel_verification(env: ManagerBasedRlEnv) -> None:
    """Focused check: per-pose metric + fired status, then a HOME tilt sweep."""
    cfg = env.reward_manager.get_term_cfg("style_ankle_parallel")
    threshold = cfg.params.get("tilt_threshold", 0.05)
    reward_val = cfg.params.get("reward", 20.0)

    # --- per-pose ---------------------------------------------------------
    _write_poses(env)
    metric = _foot_tilt_metric(env)
    fired = mdp.style_ankle_parallel(env, **cfg.params)
    print(f"\n=== ANKLE PARALLEL — per pose (threshold={threshold}, reward={reward_val}) ===")
    t = PrettyTable()
    t.field_names = ["pose", "tilt_metric", "< thr?", "reward"]
    t.align = "r"
    t.align["pose"] = "l"
    for i, name in enumerate(POSES):
        below = metric[i].item() < threshold
        t.add_row([name, f"{metric[i]:.4f}", "YES" if below else "no", f"{fired[i]:+.1f}"])
    print(t)
    print(
        "Expected: HOME / KNEES_BENT (feet flat on ground) fire +%.0f; lying poses do not."
        % reward_val
    )

    # --- HOME tilt sweep --------------------------------------------------
    asset = env.scene["robot"]
    names = asset.joint_names
    home_j = torch.tensor(
        [resolve_expr(HOME_KEYFRAME.joint_pos or {}, names, 0.0)], dtype=torch.float32
    ).repeat(env.num_envs, 1)
    home_root = torch.tensor(
        [[*HOME_KEYFRAME.pos, *(HOME_KEYFRAME.rot or (1, 0, 0, 0))]], dtype=torch.float32
    ).repeat(env.num_envs, 1)
    pitch_ids = [k for k, n in enumerate(names) if "ankle_pitch" in n]
    roll_ids = [k for k, n in enumerate(names) if "ankle_roll" in n]

    def sweep(joint_ids, label):
        #print(f"\n=== ANKLE PARALLEL — HOME sweep tilting ankle {label} (threshold={threshold}) ===")
        st = PrettyTable()
        st.field_names = [f"{label} (rad)", "deg", "tilt_metric", "reward"]
        st.align = "r"
        for tilt in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]:
            q = home_j.clone()
            # Tilt as an OFFSET from the HOME pose so 0.0 = the flat standing foot
            # (HOME's ankle_pitch is -0.2, compensating the hip/knee tilt).
            q[:, joint_ids] = home_j[:, joint_ids] + tilt
            asset.write_root_link_pose_to_sim(home_root)
            asset.write_joint_state_to_sim(q, torch.zeros_like(q))
            env.sim.forward()
            m = _foot_tilt_metric(env)[0].item()
            r = mdp.style_ankle_parallel(env, **cfg.params)[0].item()
            st.add_row([f"{tilt:.2f}", f"{tilt*57.3:.0f}", f"{m:.4f}", f"{r:+.1f}"])
        print(st)

    sweep(pitch_ids, "pitch")
    sweep(roll_ids, "roll")
    print(
        "\nThe sweep shows the foot tilt (deg) at which the +%.0f turns off. If that\n"
        "cutoff angle is not where you want 'flat', change `tilt_threshold` in\n"
        "getup_env_cfg.py (0.05 ~ 13deg; e.g. sin^2(angle) = threshold)." % reward_val
    )


def main() -> None:
    cfg = unitree_g1_getup_env_cfg()
    cfg.scene.num_envs = len(POSES)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
    all_terms_table(env)
    ankle_parallel_verification(env)


if __name__ == "__main__":
    main()
