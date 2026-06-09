"""Quantitative diagnostic for the get-up INITIAL STATE DISTRIBUTION.

Read-only inspection tool (does NOT touch training code). It builds the real
`Unitree-G1-GetUp` env, samples resets, and prints statistics about HOW the robot
spawns each episode:

  PHASE 1 — keyframe selection uniformity
    Resets with all randomization DISABLED (a clone of the cfg with the joint/pose
    perturbation, fall and velocity zeroed) so every spawn is exactly one of the 5
    reference keyframes. Classifies each env against analytic templates and reports
    the selection histogram — should be ~uniform (~20% each across SUPINE, PRONE,
    SIDE_LEFT, SIDE_RIGHT, SEATED).

  PHASE 2 — full reset distribution (the real training reset)
    Resets with the real cfg (per-group joint perturbation, pose tilt, full body
    "yaw", +0.40 m HoST drop, zero root velocity). Reports, at the elevated SPAWN:
      - per joint-family counts (SUPINE/PRONE share joints so they merge here),
      - pelvis spawn-height percentiles,
      - joint perturbation spread per body group + % joints pinned at a soft limit,
      - body orientation spread (projected gravity in body frame) — reveals that the
        ±pi "yaw" perturbation, applied in the body frame to a LYING robot, rolls it
        about its long axis and mixes back/front/side.
    Then steps with ZERO action (PD holds the default pose, as during training) for
    --settle-steps and reports the settled pelvis height, % grounded, the peak floor
    contact forces (head/torso/hands), head-impact counts, NaN and early-termination
    counts.

Run (AMD / CPU):
    MUJOCO_GL=osmesa python scripts/diag_reset_distribution.py --num-envs 512
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.string import resolve_expr
from src.assets.robots.unitree_g1.g1_constants import (
    PRONE_KEYFRAME,
    SEATED_KEYFRAME,
    SIDE_LEFT_KEYFRAME,
    SIDE_RIGHT_KEYFRAME,
    SUPINE_KEYFRAME,
)
from src.tasks.getup.config.g1.env_cfgs import unitree_g1_getup_env_cfg
from src.tasks.getup.getup_env_cfg import (
    _HEAD_IMPACT_PENALTY_THRESHOLD,
    _HEAD_IMPACT_TERMINATION_THRESHOLD,
)

# Reference keyframes in the same order the reset event registers them.
KEYFRAMES = {
    "SUPINE": SUPINE_KEYFRAME,
    "PRONE": PRONE_KEYFRAME,
    "SIDE_LEFT": SIDE_LEFT_KEYFRAME,
    "SIDE_RIGHT": SIDE_RIGHT_KEYFRAME,
    "SEATED": SEATED_KEYFRAME,
}

# Joint groups used to summarize the perturbation spread (regex -> label).
JOINT_GROUPS = {
    "hips": r".*_hip_.*",
    "knees": r".*_knee_.*",
    "ankles": r".*_ankle_.*",
    "waist": r"waist_.*",
    "shoulders": r".*_shoulder_.*",
    "elbows": r".*_elbow_.*",
    "wrists": r".*_wrist_.*",
}

GROUND_HEIGHT = 0.25  # pelvis z below this = "on the ground"


def quat_gravity_b(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Projected gravity (unit down-vector) expressed in the body frame.

    g_b = R(quat)^T @ (0,0,-1). The sign convention is irrelevant here because the
    SAME formula is used for templates and live envs, so classification is
    self-consistent. quat is (...,4) in (w,x,y,z).
    """
    w, x, y, z = quat_wxyz.unbind(-1)
    # Third ROW of R(quat) (i.e. R^T applied to (0,0,-1) = -[R[0,2],R[1,2],R[2,2]]).
    gx = -(2 * (x * z - w * y))
    gy = -(2 * (y * z + w * x))
    gz = -(1 - 2 * (x * x + y * y))
    return torch.stack([gx, gy, gz], dim=-1)


def build_templates(joint_names):
    """Analytic per-keyframe (joint vector, body-frame gravity) templates."""
    j, g, names = [], [], []
    for name, kf in KEYFRAMES.items():
        names.append(name)
        jvec = resolve_expr(getattr(kf, "joint_pos", {}) or {}, joint_names, 0.0)
        j.append(torch.tensor(jvec, dtype=torch.float32))
        rot = torch.tensor(kf.rot, dtype=torch.float32)  # (w,x,y,z)
        g.append(quat_gravity_b(rot))
    return names, torch.stack(j), torch.stack(g)


def classify(joints, grav, t_joints, t_grav, grav_weight):
    """Nearest-template index per env using joints (+ optional gravity)."""
    dj = ((joints[:, None, :] - t_joints[None]) ** 2).sum(-1)
    dg = ((grav[:, None, :] - t_grav[None]) ** 2).sum(-1)
    return (dj + grav_weight * dg).argmin(dim=1)


def pct(t: torch.Tensor):
    a = t.detach().cpu().numpy()
    return np.percentile(a, [0, 5, 50, 95, 100])


def make_env(num_envs, randomized: bool, device="cpu"):
    cfg = unitree_g1_getup_env_cfg(play=False)
    cfg.scene.num_envs = num_envs
    if not randomized:
        p = cfg.events["reset_pose"].params
        p["joint_pos_range"] = {}
        p["pose_range"] = {}
        p["velocity_range"] = {}
        p["fall_height"] = 0.0
    return ManagerBasedRlEnv(cfg, device=device)


def fmt_row(label, p):
    return (
        f"  {label:<12} min={p[0]:6.3f}  p5={p[1]:6.3f}  median={p[2]:6.3f}  "
        f"p95={p[3]:6.3f}  max={p[4]:6.3f}"
    )


def phase1(num_envs, device):
    print("\n" + "=" * 72)
    print("PHASE 1 — keyframe selection uniformity (randomization OFF)")
    print("=" * 72)
    env = make_env(num_envs, randomized=False, device=device)
    rob = env.scene["robot"]
    names, t_j, t_g = build_templates(rob.joint_names)
    env.reset()
    j = rob.data.joint_pos.clone()
    g = quat_gravity_b(rob.data.root_link_quat_w.clone())
    idx = classify(j, g, t_j, t_g, grav_weight=5.0)
    counts = torch.bincount(idx, minlength=len(names))
    total = int(counts.sum())
    print(f"  {total} envs, expected ~{100/len(names):.0f}% each:")
    for i, nm in enumerate(names):
        c = int(counts[i])
        print(f"    {nm:<12} {c:5d}  ({100*c/total:5.1f}%)")
    return env


def phase2(env, settle_steps):
    print("\n" + "=" * 72)
    print("PHASE 2 — full reset distribution (real training reset)")
    print("=" * 72)
    rob = env.scene["robot"]
    names, t_j, t_g = build_templates(rob.joint_names)
    env.reset()

    # ---- SPAWN (elevated, pre-physics) ----
    spawn_z = rob.data.root_link_pos_w[:, 2].clone()
    j = rob.data.joint_pos.clone()
    g = quat_gravity_b(rob.data.root_link_quat_w.clone())

    # Joint-family classification (orientation-invariant). SUPINE/PRONE share joints
    # so they fall into one "SUP/PRONE" bucket here.
    fam_idx = classify(j, g, t_j, t_g, grav_weight=0.0)
    fam_names = [n if n not in ("SUPINE", "PRONE") else "SUP/PRONE" for n in names]
    counts = {}
    for i, nm in zip(fam_idx.tolist(), [fam_names[k] for k in fam_idx.tolist()]):
        counts[nm] = counts.get(nm, 0) + 1
    print(f"\n  Spawn by joint-family ({env.num_envs} envs):")
    for nm in dict.fromkeys(fam_names):
        c = counts.get(nm, 0)
        print(f"    {nm:<12} {c:5d}  ({100*c/env.num_envs:5.1f}%)")

    print("\n  Spawn pelvis height (m) — expect keyframe_z + 0.40 +/- 0.03:")
    print(fmt_row("pelvis_z", pct(spawn_z)))

    # ---- Joint perturbation spread + limit saturation ----
    print("\n  Joint perturbation spread (std across envs, rad) + % at soft limit:")
    limits = rob.data.soft_joint_pos_limits  # (N,J,2)
    at_lim = (
        (j <= limits[..., 0] + 1e-4) | (j >= limits[..., 1] - 1e-4)
    ).float()
    for label, pattern in JOINT_GROUPS.items():
        ids, _ = rob.find_joints(pattern)
        if not ids:
            continue
        ids = torch.tensor(ids)
        std = j[:, ids].std(dim=0).mean().item()
        lim = 100 * at_lim[:, ids].mean().item()
        print(f"    {label:<10} std={std:5.3f}   at_limit={lim:5.1f}%")

    # ---- Orientation spread (reveals the body-frame yaw -> roll mixing) ----
    print("\n  Body orientation at spawn — projected gravity in body frame:")
    print("  (standing ~ gz=-1; lying flat ~ gz~0; the gx/gy spread shows back/")
    print("   front/side mixing caused by the body-frame 'yaw' perturbation)")
    print(fmt_row("grav_b_x", pct(g[:, 0])))
    print(fmt_row("grav_b_y", pct(g[:, 1])))
    print(fmt_row("grav_b_z", pct(g[:, 2])))

    # ---- SETTLE under zero action (PD holds default pose, as in training) ----
    peak = {k: torch.zeros(env.num_envs) for k in
            ("head", "torso", "hand_left", "hand_right")}
    sensor_of = {
        "head": "contact_head", "torso": "contact_torso",
        "hand_left": "contact_hand_left", "hand_right": "contact_hand_right",
    }
    any_nan = False
    early_term = torch.zeros(env.num_envs, dtype=torch.bool)
    zero_a = torch.zeros(env.num_envs, env.action_manager.total_action_dim,
                         device=env.device)
    for step in range(settle_steps):
        obs, rew, term, trunc, _ = env.step(zero_a)
        any_nan |= (not torch.isfinite(obs["actor"]).all()) or (
            not torch.isfinite(rew).all())
        early_term |= term.cpu()
        for k, sname in sensor_of.items():
            f = env.scene.sensors[sname].data.force.norm(dim=-1).amax(dim=1).cpu()
            torch.maximum(peak[k], f, out=peak[k])

    settle_z = rob.data.root_link_pos_w[:, 2].clone()
    print(f"\n  After {settle_steps} steps of ZERO action (PD -> default pose):")
    print(fmt_row("pelvis_z", pct(settle_z)))
    grounded = 100 * (settle_z < GROUND_HEIGHT).float().mean().item()
    print(f"    grounded (z<{GROUND_HEIGHT}): {grounded:5.1f}%")

    print("\n  Peak floor contact force during spawn+settle (N):")
    for k in peak:
        print(fmt_row(k, pct(peak[k])))

    head_pen = int((peak["head"] > _HEAD_IMPACT_PENALTY_THRESHOLD).sum())
    head_term = int((peak["head"] > _HEAD_IMPACT_TERMINATION_THRESHOLD).sum())
    print("\n  Safety at reset:")
    print(f"    head force > penalty thr ({_HEAD_IMPACT_PENALTY_THRESHOLD:.0f} N): "
          f"{head_pen}/{env.num_envs}")
    print(f"    head force > termination thr ({_HEAD_IMPACT_TERMINATION_THRESHOLD:.0f} N): "
          f"{head_term}/{env.num_envs}")
    print(f"    envs terminated within {settle_steps} steps: "
          f"{int(early_term.sum())}/{env.num_envs}")
    print(f"    any NaN in obs/reward: {any_nan}")
    return (not any_nan) and head_term == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--settle-steps", type=int, default=80)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    env1 = phase1(args.num_envs, args.device)
    # Free the Phase-1 env before building the Phase-2 env (avoid holding two full
    # simulations in memory at once).
    del env1
    import gc
    gc.collect()
    env = make_env(args.num_envs, randomized=True, device=args.device)
    ok = phase2(env, args.settle_steps)

    print("\n" + "=" * 72)
    print("RESULT:", "OK — no NaN, no head-impact termination at reset"
          if ok else "WARNING — see flagged numbers above")
    print("=" * 72)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
