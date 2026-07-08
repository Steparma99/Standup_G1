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

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
from prettytable import PrettyTable

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.string import resolve_expr

import src.tasks.getup.mdp as mdp
from src.tasks.getup.config.g1.env_cfgs import unitree_g1_getup_env_cfg, _SETTLE_STEPS
from src.tasks.getup.rl.reward_groups import GROUP_ORDER, REWARD_GROUP_MAP
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
# All 6 canonical poses used by the static per-pose table and the ankle/style checks.
# HOME is the standing reference — useful for inspecting what reward terms CAN reach.
POSES = {
    "HOME(stand)": HOME_KEYFRAME,
    "SEATED": SEATED_KEYFRAME,
    "SUPINE": SUPINE_KEYFRAME,
    "PRONE": PRONE_KEYFRAME,
    "SIDE_LEFT": SIDE_LEFT_KEYFRAME,
    "SIDE_RIGHT": SIDE_RIGHT_KEYFRAME,
}

# The 5 poses actually used to reset episodes during training (HOME is NOT a valid
# starting position for standup training). The 6th slot repeats SUPINE so num_envs=6
# is fully covered by the diagnostic env created with len(POSES) envs.
TRAINING_POSES = {
    "SUPINE":      SUPINE_KEYFRAME,
    "PRONE":       PRONE_KEYFRAME,
    "SIDE_LEFT":   SIDE_LEFT_KEYFRAME,
    "SIDE_RIGHT":  SIDE_RIGHT_KEYFRAME,
    "SEATED":      SEATED_KEYFRAME,
    "SUPINE(dup)": SUPINE_KEYFRAME,  # 6th env duplicate — HOME deliberately excluded
}

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _write_keyframes(env: ManagerBasedRlEnv, poses: dict) -> None:
    """Teleport env i to poses[i] (deterministic) and forward the sim."""
    asset = env.scene["robot"]
    names = asset.joint_names
    env.reset()
    for i, kf in enumerate(poses.values()):
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


def _write_poses(env: ManagerBasedRlEnv) -> None:
    """Write the 6 POSES (including HOME) used by static tables and firing-range scans."""
    _write_keyframes(env, POSES)


def _foot_tilt_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Mean over both feet of ||proj_grav_xy||^2 — the ankle-parallel flatness metric."""
    asset = env.scene["robot"]
    ids = torch.tensor([asset.find_bodies(n)[0][0] for n in FOOT_BODIES])
    quat = asset.data.body_link_quat_w[:, ids, :]
    g = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, quat.shape[1], -1)
    pg = quat_apply_inverse(quat, g)
    return pg[..., :2].pow(2).sum(dim=-1).mean(dim=1)  # [B]


def all_terms_table(env: ManagerBasedRlEnv) -> dict[str, list[float]]:
    """Evaluate every reward term on every pose; print weighted contributions.

    Returns {term_name: [weighted value per pose]} so the caller can aggregate the
    same numbers into per-group totals without re-calling the (stateful) term funcs.
    """
    _write_poses(env)
    rm = env.reward_manager
    pose_names = list(POSES.keys())

    table = PrettyTable()
    table.field_names = ["reward term", "group", "weight", *pose_names]
    table.align = "r"
    table.align["reward term"] = "l"
    table.align["group"] = "l"

    weighted_by_term: dict[str, list[float]] = {}
    for name, cfg in zip(rm._term_names, rm._term_cfgs, strict=False):
        group = REWARD_GROUP_MAP.get(name, "?")
        try:
            raw = cfg.func(env, **cfg.params)  # [B]
        except Exception as e:  # noqa: BLE001
            table.add_row([name, group, cfg.weight, *["ERR"] * len(pose_names)])
            print(f"  [warn] {name}: {e}")
            continue
        weighted = (raw * cfg.weight).tolist()
        weighted_by_term[name] = weighted
        table.add_row(
            [name, group, f"{cfg.weight:g}", *[f"{v:+.3f}" for v in weighted]]
        )

    print("\n=== Weighted reward contribution per pose (raw_value * weight) ===")
    print(table)
    print(
        "Read each row L->R: positive = reward, negative = penalty. Check the high/low\n"
        "cells land on the poses you intend. (Stage-gated terms read 0 off their stage;\n"
        "progress terms like height_progress read ~0 in a static pose — both expected.)"
    )

    _group_totals_table(weighted_by_term, pose_names, "per pose (static snapshot)")
    return weighted_by_term


def _group_totals_table(
    weighted_by_term: dict[str, list[float]],
    col_names: list[str],
    context: str,
) -> None:
    """Sum weighted per-term values into the 4 reward groups and print the totals.

    This is the headline balance check: which GROUP dominates. Task should be the
    largest positive; regularization should be a smaller negative — if regularization
    outweighs task, the training gradient points at 'minimize penalties', not 'stand up'.
    """
    n_cols = len(col_names)
    totals = {g: [0.0] * n_cols for g in GROUP_ORDER}
    for term, vals in weighted_by_term.items():
        group = REWARD_GROUP_MAP.get(term)
        if group not in totals:
            continue
        for c in range(n_cols):
            totals[group][c] += vals[c]

    t = PrettyTable()
    t.field_names = ["group", *col_names]
    t.align = "r"
    t.align["group"] = "l"
    for g in GROUP_ORDER:
        t.add_row([g, *[f"{v:+.3f}" for v in totals[g]]])
    # Net row: sum of all groups.
    net = [sum(totals[g][c] for g in GROUP_ORDER) for c in range(n_cols)]
    t.add_row(["NET (all groups)", *[f"{v:+.3f}" for v in net]])

    print(f"\n=== GROUP TOTALS — {context} ===")
    print(t)
    print(
        "Goal: 'task' is the dominant POSITIVE group; 'regularization' a smaller\n"
        "NEGATIVE. If |regularization| >= task, the policy is pulled toward minimizing\n"
        "penalties instead of standing up (the failure mode of the last run)."
    )


def _home_base(env: ManagerBasedRlEnv):
    """Return (asset, joint_names, home_joint_vec[1,nq], home_root[1,7]) for HOME."""
    asset = env.scene["robot"]
    names = asset.joint_names
    home_j = torch.tensor(
        [resolve_expr(HOME_KEYFRAME.joint_pos or {}, names, 0.0)], dtype=torch.float32
    )
    home_root = torch.tensor(
        [[*HOME_KEYFRAME.pos, *(HOME_KEYFRAME.rot or (1, 0, 0, 0))]], dtype=torch.float32
    )
    return asset, names, home_j, home_root


def _set_joints(base_j: torch.Tensor, names: list[str], updates: dict[str, float]) -> torch.Tensor:
    """Clone base_j and overwrite every joint whose name contains a key with its value."""
    q = base_j.clone()
    for pattern, val in updates.items():
        ids = [i for i, n in enumerate(names) if pattern in n]
        if ids:
            q[:, ids] = val
    return q


def _weighted_vec_env0(env: ManagerBasedRlEnv) -> dict[str, float]:
    """Evaluate every reward term (weighted) on env 0 in its current state."""
    rm = env.reward_manager
    out: dict[str, float] = {}
    for name, cfg in zip(rm._term_names, rm._term_cfgs, strict=False):
        try:
            raw = cfg.func(env, **cfg.params)
            out[name] = (raw[0] * cfg.weight).item()
        except Exception:  # noqa: BLE001
            out[name] = float("nan")
    return out


def firing_range_scan(env: ManagerBasedRlEnv) -> None:
    """Per-term AND per-group MIN / MAX firing envelope over a battery of probe states.

    The static per-pose table only shows the 6 canonical keyframes; many style / post-task
    terms only reach their extremes in states those poses do not cover (fast spinning,
    violated posture). This routine drives env 0 through hand-picked probe states designed
    to push each group to BOTH ends of its range, then reports, per term and per group, the
    most-negative and most-positive WEIGHTED value observed (and in which state):

      * HOME_still         — standing, upright, zero velocity  -> post_* terms fire MAX (+)
      * HOME_spin_fast     — standing but high base+joint vel   -> reg vel penalties MIN (-),
                                                                    post velocity rewards -> ~0
      * HOME_violate_pose  — standing but joints past style limits -> style deviation MIN (-),
                                                                    post_standing_posture MIN (-)
      * 6 canonical poses  — HOME + SEATED/SUPINE/PRONE/SIDE_*   -> task terms, hand contact

    Caveats: values are single-step snapshots with the term funcs called directly (like the
    per-pose table), so stateful PROGRESS terms (height_progress, prone_recovery, ...) read
    ~0 here — they need real motion, not a frozen pose. The per-group min/max is the min/max
    of the GROUP TOTAL across states (a real achievable value), not a sum of per-term extremes.
    """
    rm = env.reward_manager
    term_names = list(rm._term_names)
    states: list[tuple[str, dict[str, float]]] = []

    # --- Battery 1: the 6 canonical poses, read all envs in one forward -------------
    _write_poses(env)
    weighted_all = {}
    for name, cfg in zip(rm._term_names, rm._term_cfgs, strict=False):
        try:
            weighted_all[name] = (cfg.func(env, **cfg.params) * cfg.weight)  # [B]
        except Exception:  # noqa: BLE001
            weighted_all[name] = torch.full((env.num_envs,), float("nan"))
    for pose_idx, pose_name in enumerate(POSES):
        states.append((pose_name, {n: weighted_all[n][pose_idx].item() for n in term_names}))

    # --- Battery 2: synthetic HOME probes (env 0) -----------------------------------
    asset, names, home_j, home_root = _home_base(env)
    ids0 = torch.tensor([0])

    # (a) HOME standing, perfectly still -> post-task rewards at their positive max.
    asset.write_root_link_pose_to_sim(home_root, env_ids=ids0)
    asset.write_root_link_velocity_to_sim(torch.zeros(1, 6), env_ids=ids0)
    asset.write_joint_state_to_sim(home_j, torch.zeros_like(home_j), env_ids=ids0)
    env.sim.forward()
    states.append(("HOME_still", _weighted_vec_env0(env)))

    # (b) HOME standing but spinning/translating fast + fast joints -> velocity penalties
    #     fire (reg_arm_vel, joint_vel_limits) and post velocity rewards drop.
    fast_jvel = _set_joints(
        torch.zeros_like(home_j), names,
        {"shoulder": 20.0, "elbow": 20.0, "wrist": 20.0, "hand": 12.0},
    )
    asset.write_root_link_pose_to_sim(home_root, env_ids=ids0)
    asset.write_root_link_velocity_to_sim(
        torch.tensor([[3.0, 0.0, 2.0, 4.0, 4.0, 5.0]]), env_ids=ids0)  # lin xyz, ang xyz
    asset.write_joint_state_to_sim(home_j, fast_jvel, env_ids=ids0)
    env.sim.forward()
    states.append(("HOME_spin_fast", _weighted_vec_env0(env)))

    # (c) HOME standing but posture deliberately violated -> style deviation penalties
    #     and post_standing_posture (home_pose_l2) at their negative min.
    bad_j = _set_joints(
        home_j, names,
        {
            "waist_yaw_joint": 1.5,       # > style_waist_yaw limit 1.4
            "waist_pitch_joint": 0.6,     # > style_waist_upright band 0.25
            "hip_roll": 1.5,              # > style_hip roll_limit 1.4
            "hip_yaw": 1.0,               # > style_hip yaw_limit 0.9
            "knee_joint": 2.90,           # > style_knee hi_limit 2.85
            "left_shoulder_roll": -0.6,   # < style_shoulder left_limit -0.4
            "right_shoulder_roll": 0.6,   # > style_shoulder right_limit 0.4
        },
    )
    asset.write_root_link_pose_to_sim(home_root, env_ids=ids0)
    asset.write_root_link_velocity_to_sim(torch.zeros(1, 6), env_ids=ids0)
    asset.write_joint_state_to_sim(bad_j, torch.zeros_like(bad_j), env_ids=ids0)
    env.sim.forward()
    states.append(("HOME_violate_pose", _weighted_vec_env0(env)))

    # --- Per-term min/max over all probe states -------------------------------------
    state_labels = [lbl for lbl, _ in states]
    tt = PrettyTable()
    tt.field_names = ["reward term", "group", "weight", "min fired", "@min", "max fired", "@max"]
    tt.align = "r"
    tt.align["reward term"] = "l"
    tt.align["group"] = "l"
    tt.align["@min"] = "l"
    tt.align["@max"] = "l"
    for name in term_names:
        cfg = rm.get_term_cfg(name)
        vals = [vec.get(name, float("nan")) for _, vec in states]
        finite = [(v, state_labels[i]) for i, v in enumerate(vals) if v == v]  # drop NaN
        if not finite:
            tt.add_row([name, REWARD_GROUP_MAP.get(name, "?"), f"{cfg.weight:g}",
                        "nan", "-", "nan", "-"])
            continue
        vmin, smin = min(finite, key=lambda p: p[0])
        vmax, smax = max(finite, key=lambda p: p[0])
        tt.add_row([name, REWARD_GROUP_MAP.get(name, "?"), f"{cfg.weight:g}",
                    f"{vmin:+.3f}", smin, f"{vmax:+.3f}", smax])

    print("\n=== FIRING RANGE — per-term weighted MIN / MAX over probe states ===")
    print(tt)
    print(
        "min/max = most-negative / most-positive WEIGHTED value seen across the probe\n"
        "states. Progress terms (height_progress, prone_recovery, supine_rising_prep)\n"
        "read ~0 here — they reward MOTION, not a frozen pose; see the --steps rollout."
    )

    # --- Per-group min/max of the GROUP TOTAL across probe states --------------------
    gt = PrettyTable()
    gt.field_names = ["group", "min total", "@min", "max total", "@max"]
    gt.align = "r"
    gt.align["group"] = "l"
    gt.align["@min"] = "l"
    gt.align["@max"] = "l"
    for g in GROUP_ORDER:
        group_terms = [n for n in term_names if REWARD_GROUP_MAP.get(n) == g]
        per_state = []
        for i, (_, vec) in enumerate(states):
            s = sum(vec.get(n, 0.0) for n in group_terms if vec.get(n, 0.0) == vec.get(n, 0.0))
            per_state.append((s, state_labels[i]))
        vmin, smin = min(per_state, key=lambda p: p[0])
        vmax, smax = max(per_state, key=lambda p: p[0])
        gt.add_row([g, f"{vmin:+.3f}", smin, f"{vmax:+.3f}", smax])

    print("\n=== FIRING RANGE — per-GROUP total MIN / MAX across probe states ===")
    print(gt)
    print(
        "Each cell is the GROUP SUM in a single probe state (a real achievable value).\n"
        "post_task should show a large POSITIVE max at HOME_still; style should show its\n"
        "negative penalties at HOME_violate_pose and positive posture reward at HOME_still."
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


def _collect_rollout_stats(
    env: ManagerBasedRlEnv, steps: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
    """Run `steps` zero-action steps; return (sum, sumsq, min, max, n_samples, any_nan).

    Reads per-term weighted rewards from RewardManager._step_reward after each env.step()
    so stateful progress terms are NOT double-called. All tensors are on the env device.
    """
    rm = env.reward_manager
    n_terms = len(rm._term_names)
    zero_a = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    sum_t   = torch.zeros(n_terms, device=env.device)
    sumsq_t = torch.zeros(n_terms, device=env.device)
    min_t   = torch.full((n_terms,),  float("inf"), device=env.device)
    max_t   = torch.full((n_terms,), -float("inf"), device=env.device)
    n_samples = 0
    any_nan = False
    for _ in range(steps):
        env.step(zero_a)
        step_r = rm._step_reward  # [num_envs, n_terms], weighted
        any_nan |= not torch.isfinite(step_r).all().item()
        sum_t   += step_r.sum(dim=0)
        sumsq_t += step_r.pow(2).sum(dim=0)
        min_t    = torch.minimum(min_t, step_r.min(dim=0).values)
        max_t    = torch.maximum(max_t, step_r.max(dim=0).values)
        n_samples += env.num_envs
    return sum_t, sumsq_t, min_t, max_t, n_samples, any_nan


def _skip_settle_window(env: ManagerBasedRlEnv) -> None:
    """Advance episode_length_buf past the settling window so the reward gate is inactive.

    The training env zeroes ALL rewards for the first _SETTLE_STEPS steps of each episode
    (gate_settling_rewards, mode="step"). In a DIAGNOSTIC rollout we want to see real
    reward values from step 1 (especially --from-home, whose point is the standing values),
    so we fast-forward the per-env step counter past the gate. The action term's settle-hold
    (in_settle = episode_length_buf < _SETTLE_STEPS) also switches off, so the robot is
    policy-controlled (zero action) from the first step — exactly what these rollouts show.
    """
    if _SETTLE_STEPS > 0:
        env.episode_length_buf[:] = _SETTLE_STEPS + 1


def _status(mean: float, std: float) -> str:
    if abs(mean) < 1e-4:
        return "INACTIVE"
    if mean > 0:
        return "TASK_SIGNAL(+)"
    if std < 0.15 * abs(mean):
        return "NOISE_FLOOR(!)"
    return "REG_PENALTY(-)"


def _print_rollout_table(
    env: ManagerBasedRlEnv,
    steps: int,
    title: str,
    footer: str,
) -> None:
    """Collect zero-action stats and print per-term table + group totals.

    Columns: mean/step | min/step | max/step | ep_sum | status
    min/step and max/step are the OBSERVED per-step extremes across ALL envs and steps,
    giving you the actual numerical range each reward can reach in this scenario.
    """
    rm = env.reward_manager
    term_names = list(rm._term_names)
    sum_t, sumsq_t, min_t, max_t, n_samples, any_nan = _collect_rollout_stats(env, steps)

    mean_t = (sum_t / max(n_samples, 1)).cpu()
    var    = (sumsq_t / max(n_samples, 1)).cpu() - mean_t.pow(2)
    std_t  = var.clamp(min=0.0).sqrt()
    ep_sum = mean_t * steps
    min_t  = min_t.cpu()
    max_t  = max_t.cpu()

    table = PrettyTable()
    table.field_names = ["reward term", "group", "mean/step", "min/step", "max/step", "ep_sum", "status"]
    table.align = "r"
    table.align["reward term"] = "l"
    table.align["group"] = "l"
    table.align["status"] = "l"
    weighted_by_term: dict[str, list[float]] = {}
    for i, name in enumerate(term_names):
        m, s = mean_t[i].item(), std_t[i].item()
        mn, mx = min_t[i].item(), max_t[i].item()
        ep = ep_sum[i].item()
        weighted_by_term[name] = [ep]
        table.add_row([
            name, REWARD_GROUP_MAP.get(name, "?"),
            f"{m:+.4f}", f"{mn:+.4f}", f"{mx:+.4f}", f"{ep:+.3f}",
            _status(m, s),
        ])

    print(f"\n=== {title} ===")
    if any_nan:
        print("  [WARN] non-finite per-term reward encountered during the rollout!")
    print(table)
    print(
        "min/step = worst single step across all envs (penalty at its deepest or reward at its lowest).\n"
        "max/step = best single step across all envs (reward at its peak or penalty near zero).\n"
        "ep_sum   = mean/step × steps  (expected episode contribution under this policy/starting state).\n"
        f"{footer}"
    )
    _group_totals_table(weighted_by_term, ["ep_sum"], f"{title} — group sums")


def zero_action_timeseries(env: ManagerBasedRlEnv, steps: int) -> None:
    """Roll out `steps` env-steps under ZERO action from the 5 training keyframes.

    Uses TRAINING_POSES (SUPINE, PRONE, SIDE_LEFT, SIDE_RIGHT, SEATED) — the exact poses
    used to reset episodes in training. HOME is deliberately excluded; it is not a valid
    starting position for standup training.

    Settling mechanics active (matching training):
      - First 30 steps: PD holds current pose (policy not controlling) — settle window.
      - Impact penalties (hand/head contact) fade in over first 8-10 steps via _reset_ramp.
      - Velocity is zeroed at the end of the settle window (settle_zero_velocity event).

    Since the robot never stands under zero action, TASK terms read a LOWER BOUND and
    REG/penalty terms show their passive floor.
    """
    _write_keyframes(env, TRAINING_POSES)
    _skip_settle_window(env)  # show real reward values, not the gated settling zeros
    _print_rollout_table(
        env, steps,
        title=(
            f"ZERO-ACTION TIME-SERIES — {steps} steps from training keyframes "
            f"(SUPINE/PRONE/SIDE_L/SIDE_R/SEATED — HOME excluded)"
        ),
        footer=(
            "Training poses only — HOME is NOT a training start position.\n"
            "Settle window: first 30 steps PD-held (policy passive); hand contact fades in\n"
            "  over steps 0-8 (_GRACE_STEPS), head contact over steps 0-20 (_MASK_STEPS=20).\n"
            "STATUS: TASK_SIGNAL(+) = reward a trained policy can grow by rising.\n"
            "NOISE_FLOOR(!) = near-constant penalty the policy CANNOT reduce.\n"
            "REG_PENALTY(-) = negative that varies with motion (legit shaping). INACTIVE = 0."
        ),
    )


def standing_fall_timeseries(env: ManagerBasedRlEnv, steps: int) -> None:
    """Roll out `steps` env-steps under ZERO action starting from HOME (standing) pose.

    All envs are teleported to the HOME standing configuration with zero velocity, then
    the physics runs with zero action — the robot falls naturally. This reveals:

      * post_task terms at their MAXIMUM in step 1 (robot is upright), then decaying
        as the robot loses balance and descends — shows the realistic firing range.
      * style terms that fire during the descent (style_shank_orientation, style_base_ang_vel
        activate ~0.37-0.49 m; posture terms at ~0.65 m during the fall).
      * task terms reading their natural values at the standing-→-falling trajectory.
      * min/step and max/step capture both ends of each term's range in one scenario.

    Read min/step as "the worst it gets" (robot fully fallen or moving fast) and max/step
    as "the best it can achieve" (robot still upright at the start of the rollout).
    """
    env.reset()
    asset, names, home_j, home_root = _home_base(env)
    # Set ALL envs to HOME standing pose with zero velocity.
    home_j_all   = home_j.repeat(env.num_envs, 1)
    home_root_all = home_root.repeat(env.num_envs, 1)
    asset.write_root_link_pose_to_sim(home_root_all)
    asset.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6))
    asset.write_joint_state_to_sim(home_j_all, torch.zeros_like(home_j_all))
    env.sim.forward()
    _skip_settle_window(env)  # bypass the reward gate so standing values show from step 1

    _print_rollout_table(
        env, steps,
        title=f"STANDING-FALL TIME-SERIES — {steps} steps from HOME pose ({env.num_envs} envs)",
        footer=(
            "Robot starts perfectly standing (zero velocity) then falls with no action.\n"
            "max/step ≈ the best the reward fires (robot upright, step ~1).\n"
            "min/step ≈ the worst value reached (robot on floor or falling fast).\n"
            "ep_sum = total over the fall episode — useful to compare group magnitudes.\n"
            "stable_success_hold and post_stand_on_feet need physics contact settling;\n"
            "  they may read 0 or low here even though the robot starts standing (expected)."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=0,
        help="If >0, run a zero-action time-series rollout of N steps instead of the "
             "static-pose snapshot (predicts the passive reward balance before training).",
    )
    parser.add_argument(
        "--ranges", action="store_true",
        help="Run ONLY the per-term / per-group MIN/MAX firing-range scan (probe states "
             "that push every group — style + post-task included — to both extremes).",
    )
    parser.add_argument(
        "--from-home", type=int, default=0, dest="from_home",
        help="If >0, run a zero-action rollout of N steps starting from the HOME (standing) "
             "pose. The robot falls naturally, revealing post-task and style firing ranges "
             "from the top of the trajectory down. min/step = worst during fall; "
             "max/step = best while still upright.",
    )
    args = parser.parse_args()

    cfg = unitree_g1_getup_env_cfg()
    cfg.scene.num_envs = len(POSES)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)

    if args.ranges:
        firing_range_scan(env)
    elif args.from_home > 0:
        standing_fall_timeseries(env, args.from_home)
    elif args.steps > 0:
        zero_action_timeseries(env, args.steps)
    else:
        home_pose_positions(env)
        all_terms_table(env)
        firing_range_scan(env)
        style_violation_check(env)
        ankle_parallel_verification(env)


if __name__ == "__main__":
    main()
