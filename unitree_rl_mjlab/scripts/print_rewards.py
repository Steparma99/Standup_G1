#!/usr/bin/env python3
"""Print a compact, sampled summary of training scalars from a TensorBoard run.

Usage:
    python scripts/print_rewards.py <run_dir>
    python scripts/print_rewards.py <run_dir> --n 8
    python scripts/print_rewards.py <run_dir> --all

<run_dir> may be:
  - a path to a run directory (absolute or relative to cwd), or
  - just the run-dir NAME, resolved under --logdir
    (default: logs/rsl_rl/g1_getup relative to cwd).

With --all every scalar tag in the event file is printed (useful for discovery).
Without --all prints the standard health metrics + all Episode_Reward/* terms found.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from tensorboard.backend.event_processing import event_accumulator

HEALTH_TAGS = [
    "Train/mean_episode_length",
    "Train/mean_reward",
    "Episode_Termination/no_progress_timeout",
    "Episode_Termination/time_out",
    "Episode_Termination/ground_penetration",
    "Episode_Termination/base_vel_explosion",
    "Episode_Termination/joint_vel_explosion",
    "Episode_Metrics/stage/stage0",
    "Episode_Metrics/stage/stage1",
    "Episode_Metrics/stage/stage2",
    "Episode_Metrics/stage/stage3",
    "Episode_Metrics/success/candidate",
    "Episode_Metrics/success/ever_stood",
    "Episode_Metrics/success/stable_hold",
    "Episode_Metrics/success/fall_after_success",
    "Episode_Metrics/curriculum/assistance_force",
    "Episode_Metrics/curriculum/beta_rescaler",
    "Episode_Metrics/reward_group/task",
    "Episode_Metrics/reward_group/style",
    "Episode_Metrics/reward_group/regularization",
    "Episode_Metrics/reward_group/post_task",
    "Episode_Metrics/contact/feet",
    "Episode_Metrics/contact/torso",
]


def resolve_run_dir(run: str, logdir: str) -> str:
    if os.path.isdir(run):
        return run
    candidate = os.path.join(logdir, run)
    if os.path.isdir(candidate):
        return candidate
    sys.exit(f"Run directory not found: {run!r} (also tried {candidate!r})")


def resolve_event_file(run_dir: str) -> str:
    events = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not events:
        sys.exit(f"No events.out.tfevents.* found in {run_dir}")
    return events[-1]  # newest (resumed runs can have more than one)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="Run directory or run-dir name.")
    ap.add_argument("--n", type=int, default=12, help="Samples per tag (default 12).")
    ap.add_argument("--logdir", default="logs/rsl_rl/g1_getup",
                    help="Root to resolve a bare run name (default: logs/rsl_rl/g1_getup).")
    ap.add_argument("--all", action="store_true",
                    help="Print every scalar tag in the file (discovery mode).")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run, args.logdir)
    ev = resolve_event_file(run_dir)
    print(f"[INFO] Reading: {ev}\n")

    ea = event_accumulator.EventAccumulator(ev, size_guidance={"scalars": 0})
    ea.Reload()
    available: set[str] = set(ea.Tags().get("scalars", []))

    def sample(tag: str) -> None:
        if tag not in available:
            return
        vals = ea.Scalars(tag)
        n = args.n
        idxs = (list(range(len(vals))) if len(vals) <= n
                else [round(i * (len(vals) - 1) / (n - 1)) for i in range(n)])
        pts = [(vals[i].step, vals[i].value) for i in idxs]
        print(tag + ":")
        print("  " + "  ".join(f"it{s}={v:.4f}" for s, v in pts))

    if args.all:
        for t in sorted(available):
            sample(t)
    else:
        # Health / training overview
        print("=== HEALTH ===")
        for t in HEALTH_TAGS:
            sample(t)

        # All individual reward terms (auto-discovered)
        reward_tags = sorted(t for t in available if t.startswith("Episode_Reward/"))
        if reward_tags:
            print("\n=== INDIVIDUAL REWARD TERMS ===")
            for t in reward_tags:
                sample(t)
        else:
            print("\n[INFO] No Episode_Reward/* tags found.")
            print("Available tag prefixes:", sorted({t.rsplit("/", 1)[0] for t in available}))

    if "Train/mean_episode_length" in available:
        print(f"\nlast iteration step seen: {ea.Scalars('Train/mean_episode_length')[-1].step}")


if __name__ == "__main__":
    main()
