#!/usr/bin/env python3
"""Print a compact, sampled summary of training scalars from a TensorBoard run.

Usage:
    python scripts/print_rewards.py <run_dir> [--n 12] [--logdir <root>]

<run_dir> may be:
  - a path to a run directory (absolute or relative to cwd), or
  - just the run-dir NAME, resolved under --logdir
    (default: logs/rsl_rl/g1_getup relative to cwd).

Picks the NEWEST events.out.tfevents.* in the run dir (a resumed run can have
more than one), samples each tag at ~n evenly spaced iterations, and prints them.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from tensorboard.backend.event_processing import event_accumulator

# Standard run-health tags + the smoother_v1 reward terms (new + changed).
DEFAULT_TAGS = [
    "Train/mean_episode_length", "Train/mean_reward",
    "Episode_Termination/no_progress_timeout", "Episode_Termination/time_out",
    "Episode_Termination/ground_penetration", "Episode_Termination/base_vel_explosion",
    "Episode_Termination/joint_vel_explosion",
    "Episode_Metrics/stage/stage0", "Episode_Metrics/stage/stage1",
    "Episode_Metrics/stage/stage2", "Episode_Metrics/stage/stage3",
    "Episode_Metrics/success/candidate", "Episode_Metrics/success/ever_stood",
    "Episode_Metrics/success/stable_hold", "Episode_Metrics/success/fall_after_success",
    "Episode_Metrics/curriculum/assistance_force", "Episode_Metrics/curriculum/beta_rescaler",
    "Episode_Metrics/reward_group/task", "Episode_Metrics/reward_group/style",
    "Episode_Metrics/reward_group/regularization", "Episode_Metrics/reward_group/post_task",
    "Episode_Metrics/contact/feet", "Episode_Metrics/contact/torso",
    # --- smoother_v1 reward terms (new + changed) ---
    "Episode_Reward/reg_arm_vel",             # NEW: arm flailing (less negative = calmer)
    "Episode_Reward/style_waist_upright",     # NEW: trunk straightness (small negative)
    "Episode_Reward/action_rate_l2",          # smoothness (now -2.5e-3)
    "Episode_Reward/action_acc_l2",           # smoothness (now -2.5e-3)
    "Episode_Reward/joint_vel_l2",            # global joint vel (now -2e-4)
    "Episode_Reward/post_base_orientation",        # upright hold
    "Episode_Reward/task_base_orientation",        # uprightness during the rise
    # --- smoother_v3: new monitoring targets ---
    "Episode_Reward/style_shoulder_roll_deviation", # should be near 0 (limit widened to ±0.4)
    "Episode_Reward/post_base_height",              # should be near +10 (target fixed to 0.80m)
    "Episode_Reward/post_feet_parallel",            # should be near +2.5 (clip_min fixed)
    # --- posture_v1: full-body HOME tracking + both-feet grounded ---
    "Episode_Reward/post_standing_posture",  # full-body HOME (legs+waist+arms, weight=12, kp=1.5); near +12 when at HOME
    "Episode_Reward/post_stand_on_feet",     # both-feet contact (weight=5); near +5 when fully grounded
    "Episode_Reward/style_waist_upright",    # trunk straightness (weight=-8); should be near 0
]


def resolve_event_file(run: str, logdir: str) -> str:
    run_dir = run if os.path.isdir(run) else os.path.join(logdir, run)
    if not os.path.isdir(run_dir):
        sys.exit(f"Run directory not found: {run_dir}")
    events = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not events:
        sys.exit(f"No events.out.tfevents.* found in {run_dir}")
    return events[-1]  # newest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="Run directory or run-dir name (see --logdir).")
    ap.add_argument("--n", type=int, default=12, help="Samples per tag (default 12).")
    ap.add_argument("--logdir", default="logs/rsl_rl/g1_getup",
                    help="Root to resolve a bare run name (default: logs/rsl_rl/g1_getup).")
    args = ap.parse_args()

    ev = resolve_event_file(args.run, args.logdir)
    print(f"[INFO] Reading: {ev}\n")
    ea = event_accumulator.EventAccumulator(ev, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    def sample(tag: str) -> None:
        if tag not in available:
            print(f"{tag}: NOT FOUND")
            return
        vals = ea.Scalars(tag)
        n = args.n
        idxs = (range(len(vals)) if len(vals) <= n
                else [round(i * (len(vals) - 1) / (n - 1)) for i in range(n)])
        pts = [(vals[i].step, vals[i].value) for i in idxs]
        print(tag + ":")
        print("  " + "  ".join(f"it{s}={v:.4f}" for s, v in pts))

    for t in DEFAULT_TAGS:
        sample(t)

    if "Train/mean_episode_length" in available:
        print(f"\nlast iteration step seen: {ea.Scalars('Train/mean_episode_length')[-1].step}")


if __name__ == "__main__":
    main()
