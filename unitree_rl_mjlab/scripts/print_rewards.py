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
Without --all prints the standard health metrics, the CURRENT configured reward stack
(group + weight from the G1 get-up config), and all Episode_Reward/* terms found.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass

from tensorboard.backend.event_processing import event_accumulator

from src.tasks.getup.config.g1.env_cfgs import unitree_g1_getup_env_cfg
from src.tasks.getup.rl.reward_groups import REWARD_GROUP_MAP

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


@dataclass(frozen=True)
class RewardMeta:
    group: str
    weight: float | None


def load_reward_meta() -> tuple[dict[str, RewardMeta], str | None]:
    """Load the current active reward config for the G1 get-up task."""
    try:
        cfg = unitree_g1_getup_env_cfg()
        rewards = getattr(cfg, "rewards", {})
        meta = {
            name: RewardMeta(
                group=REWARD_GROUP_MAP.get(name, "?"),
                weight=getattr(term_cfg, "weight", None),
            )
            for name, term_cfg in rewards.items()
        }
        return meta, None
    except Exception as exc:  # noqa: BLE001
        return {}, f"[WARN] Could not load current reward config: {exc}"


def format_weight(weight: float | None) -> str:
    return "?" if weight is None else f"{weight:g}"


def ordered_reward_tags(
    reward_meta: dict[str, RewardMeta], available: set[str]
) -> list[str]:
    """Return logged reward tags ordered like the current config, then extras."""
    available_reward_tags = {t for t in available if t.startswith("Episode_Reward/")}
    ordered = [
        f"Episode_Reward/{name}"
        for name in reward_meta
        if f"Episode_Reward/{name}" in available_reward_tags
    ]
    extras = sorted(available_reward_tags - set(ordered))
    return ordered + extras


def print_reward_config_summary(
    reward_meta: dict[str, RewardMeta], available: set[str]
) -> None:
    """Print the current reward stack with group/weight and log presence."""
    if not reward_meta:
        return

    print("\n=== CURRENT REWARD CONFIG (G1 GET-UP) ===")
    for name, meta in reward_meta.items():
        tag = f"Episode_Reward/{name}"
        status = "logged in run" if tag in available else "not in run"
        print(
            f"{name:<30} group={meta.group:<14} "
            f"weight={format_weight(meta.weight):>7}  {status}"
        )


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
    reward_meta, reward_meta_warn = load_reward_meta()

    def sample(tag: str, header: str | None = None) -> None:
        if tag not in available:
            return
        vals = ea.Scalars(tag)
        n = args.n
        idxs = (list(range(len(vals))) if len(vals) <= n
                else [round(i * (len(vals) - 1) / (n - 1)) for i in range(n)])
        pts = [(vals[i].step, vals[i].value) for i in idxs]
        print((header or tag) + ":")
        print("  " + "  ".join(f"it{s}={v:.4f}" for s, v in pts))

    if args.all:
        for t in sorted(available):
            sample(t)
    else:
        # Health / training overview
        print("=== HEALTH ===")
        for t in HEALTH_TAGS:
            sample(t)

        if reward_meta_warn:
            print(f"\n{reward_meta_warn}")
        print_reward_config_summary(reward_meta, available)

        # All individual reward terms (auto-discovered)
        reward_tags = ordered_reward_tags(reward_meta, available)
        if reward_tags:
            print("\n=== INDIVIDUAL REWARD TERMS ===")
            for t in reward_tags:
                reward_name = t.split("/", 1)[1]
                meta = reward_meta.get(reward_name)
                if meta is None:
                    sample(t, header=f"{t}  [group=?, weight=?]")
                else:
                    sample(
                        t,
                        header=(
                            f"{t}  [group={meta.group}, "
                            f"weight={format_weight(meta.weight)}]"
                        ),
                    )
        else:
            print("\n[INFO] No Episode_Reward/* tags found.")
            print("Available tag prefixes:", sorted({t.rsplit("/", 1)[0] for t in available}))

    if "Train/mean_episode_length" in available:
        print(f"\nlast iteration step seen: {ea.Scalars('Train/mean_episode_length')[-1].step}")


if __name__ == "__main__":
    main()
