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
    PRONE_KEYFRAME,
    SEATED_KEYFRAME,
    SIDE_LEFT_KEYFRAME,
    SIDE_RIGHT_KEYFRAME,
    SUPINE_KEYFRAME,
    G1_CRITIC_KEYPOINTS,
)

# Ordered so the "feet-flat standing" poses (HOME, KNEES_BENT) sit first — the ankle
# parallel reward should fire on those and NOT on the lying poses to their right.
POSES = {
    "HOME(stand)": HOME_KEYFRAME,
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


BODY_LANDMARKS = [
    # (display_name, body_name_in_xml)
    ("pelvis (root)",          "pelvis"),
    ("torso_link (head proxy)", "torso_link"),
    ("L shoulder",             "left_shoulder_pitch_link"),
    ("R shoulder",             "right_shoulder_pitch_link"),
    ("L elbow",                "left_elbow_link"),
    ("R elbow",                "right_elbow_link"),
    ("L wrist",                "left_wrist_yaw_link"),
    ("R wrist",                "right_wrist_yaw_link"),
    ("L hand (thumb base)",    "left_hand_thumb_0_link"),
    ("R hand (thumb base)",    "right_hand_thumb_0_link"),
    ("L hip",                  "left_hip_pitch_link"),
    ("R hip",                  "right_hip_pitch_link"),
    ("L knee",                 "left_knee_link"),
    ("R knee",                 "right_knee_link"),
    ("L ankle",                "left_ankle_pitch_link"),
    ("R ankle",                "right_ankle_pitch_link"),
]


def home_pose_positions(env: ManagerBasedRlEnv) -> None:
    """Print world-frame positions of key body landmarks at HOME (standing) pose."""
    _write_poses(env)  # env 0 = HOME(stand)
    asset = env.scene["robot"]
    e = 0  # HOME is POSES index 0

    print("\n=== HOME pose — body landmark positions (world frame, env 0) ===")
    t = PrettyTable()
    t.field_names = ["body", "x (m)", "y (m)", "z (m)", "height note"]
    t.align = "r"
    t.align["body"] = "l"
    t.align["height note"] = "l"

    _kp_set = set(G1_CRITIC_KEYPOINTS)
    for label, bname in BODY_LANDMARKS:
        ids, _ = asset.find_bodies(bname)
        if not ids:
            t.add_row([label, "?", "?", "?", f"body '{bname}' not found"])
            continue
        bid = ids[0]
        pos = asset.data.body_link_pos_w[e, bid]
        x, y, z = pos[0].item(), pos[1].item(), pos[2].item()
        note = ""
        if "pelvis" in label:
            note = "<-- pelvis height (height_progress tracks this)"
        elif "torso" in label:
            note = "<-- head proxy (task_head_height lower=0.794)"
        if bname in _kp_set:
            note = ("★ KEYPOINT  " + note).strip()
        t.add_row([label, f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", note])

    # foot sites
    site_names = asset.site_names
    for side, sname in [("L foot site", "left_foot"), ("R foot site", "right_foot")]:
        if sname in site_names:
            sid = site_names.index(sname)
            pos = asset.data.site_pos_w[e, sid]
            t.add_row([side, f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}", "<-- foot contact site"])

    print(t)

    # quick summary of the two reward-relevant heights
    ids_torso, _ = asset.find_bodies("torso_link")
    h_torso = asset.data.body_link_pos_w[e, ids_torso[0], 2].item()
    h_pelvis = asset.data.root_link_pos_w[e, 2].item()
    pg = asset.data.projected_gravity_b[e]
    upright = -pg[2].item()
    # head_link is a geom (not a body) on torso_link at local offset (0.004, 0, -0.044).
    # No body frame exists for it; estimate center from torso_link world pos + offset.
    h_head_geom = h_torso + (-0.044)
    _HEAD_LOWER = 0.794  # must match task_head_height lower= in getup_env_cfg.py
    print(f"\n  pelvis height          : {h_pelvis:.4f} m  (height_progress tracks this; post_base_height targets 0.75)")
    print(f"  torso_link height      : {h_torso:.4f} m  (task_head_height lower={_HEAD_LOWER} → {'FULL reward' if h_torso >= _HEAD_LOWER else 'PARTIAL reward'})")
    print(f"  head_collision center  : {h_head_geom:.4f} m  (torso_link - 0.044; head_link is a geom on torso_link, not a body)")
    print(f"  upright metric (-pg_z) : {upright:.4f}    (task_base_orientation lower=0.99 → {'FULL reward' if upright >= 0.99 else 'PARTIAL reward'})")

    # Dedicated keypoints block — shows HOME reference positions for the 12 critic keypoints
    print("\n=== CRITIC KEYPOINTS — HOME reference positions (★ in table above) ===")
    print("  (NOT YET in critic obs — add ObservationTermCfg + G1_CRITIC_KEYPOINTS to enable)")
    kp_table = PrettyTable()
    kp_table.field_names = ["#", "body name", "x (m)", "y (m)", "z (m)"]
    kp_table.align = "r"
    kp_table.align["body name"] = "l"
    for i, bname in enumerate(G1_CRITIC_KEYPOINTS, 1):
        ids, _ = asset.find_bodies(bname)
        if not ids:
            kp_table.add_row([i, bname, "?", "?", "?"])
            continue
        pos = asset.data.body_link_pos_w[e, ids[0]]
        kp_table.add_row([i, bname, f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}"])
    print(kp_table)
    print(f"  → {len(G1_CRITIC_KEYPOINTS)} keypoints × 3 = {len(G1_CRITIC_KEYPOINTS)*3} critic dims when enabled")


def style_violation_check(env: ManagerBasedRlEnv) -> None:
    """Verify style deviation terms actually fire by deliberately violating each limit.

    Teleports env 0 to HOME standing height, then pushes one joint at a time past its
    limit. If the term is correctly implemented the weighted contribution must be < 0.
    """
    asset = env.scene["robot"]
    names = asset.joint_names
    rm = env.reward_manager

    # Base HOME config used for all tests
    home_root = torch.tensor(
        [[*HOME_KEYFRAME.pos, *(HOME_KEYFRAME.rot or (1, 0, 0, 0))]], dtype=torch.float32
    )
    home_j = torch.tensor(
        [resolve_expr(HOME_KEYFRAME.joint_pos or {}, names, 0.0)], dtype=torch.float32
    )

    def _set(joint_pattern: str, value: float) -> None:
        q = home_j.clone()
        ids = [i for i, n in enumerate(names) if joint_pattern in n]
        q[:, ids] = value
        asset.write_root_link_pose_to_sim(home_root, env_ids=torch.tensor([0]))
        asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=torch.tensor([0]))
        env.sim.forward()

    def _eval(term_name: str) -> float:
        cfg = rm.get_term_cfg(term_name)
        return (cfg.func(env, **cfg.params)[0] * cfg.weight).item()

    checks = [
        # (description, joint_pattern, set_value, term_name, expect_sign)
        ("waist_yaw = 1.5 (limit 1.4)",        "waist_yaw_joint",         1.5,   "style_waist_yaw_deviation",     "<0"),
        ("hip_roll = 1.5 (limit 1.4)",          "hip_roll_joint",          1.5,   "style_hip_deviation",           "<0"),
        ("hip_yaw = 1.0 (limit 0.9)",           "hip_yaw_joint",           1.0,   "style_hip_deviation",           "<0"),
        ("knee = 2.90 (hi_limit 2.85)",         "knee_joint",              2.90,  "style_knee_deviation",          "<0"),
        ("l_shoulder_roll = -0.5 (limit -0.4)", "left_shoulder_roll",     -0.5,  "style_shoulder_roll_deviation", "<0"),
        ("r_shoulder_roll = +0.5 (limit +0.4)", "right_shoulder_roll",     0.5,  "style_shoulder_roll_deviation", "<0"),
        ("no violation at HOME",                 "_NONE_",                  0.0,   "style_waist_yaw_deviation",     "=0"),
    ]

    print("\n=== STYLE DEVIATION — violation sanity check (env 0 at HOME height) ===")
    t = PrettyTable()
    t.field_names = ["test", "term", "weighted value", "expect", "PASS?"]
    t.align = "l"
    t.align["weighted value"] = "r"

    for desc, jpat, jval, term, expect in checks:
        if jpat == "_NONE_":
            # reset to clean HOME
            asset.write_root_link_pose_to_sim(home_root, env_ids=torch.tensor([0]))
            asset.write_joint_state_to_sim(home_j, torch.zeros_like(home_j), env_ids=torch.tensor([0]))
            env.sim.forward()
        else:
            _set(jpat, jval)
        val = _eval(term)
        if expect == "<0":
            ok = "✓" if val < -0.01 else "✗ BUG"
        else:
            ok = "✓" if abs(val) < 0.01 else "✗ BUG"
        t.add_row([desc, term, f"{val:+.3f}", expect, ok])

    print(t)


def main() -> None:
    cfg = unitree_g1_getup_env_cfg()
    cfg.scene.num_envs = len(POSES)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
    home_pose_positions(env)
    all_terms_table(env)
    style_violation_check(env)
    ankle_parallel_verification(env)


if __name__ == "__main__":
    main()
