#!/usr/bin/env python3
"""Print the mean assistance force (N) a run had at a given iteration.

Reads Episode_Metrics/curriculum/assistance_force from a run's TensorBoard event
file and prints the value logged at the step closest to the requested iteration.
Used to reproduce training conditions at eval time: feed the result into
GETUP_EVAL_ASSIST_FORCE so scripts/play.py pins that same lift (see env_cfgs.py
play branch), making the rendered video match the checkpoint's metrics.

Usage:
    python scripts/assist_force_at.py <run_dir_or_name> <iteration>
    python scripts/assist_force_at.py 2026-07-10_11-52-02_phase1_stand_v1 1100

Prints a single number (Newtons) to stdout, e.g. "99.3". Prints "0" if the
metric is absent (curriculum disabled). Resolves a bare run name under
--logdir (default logs/rsl_rl/g1_getup).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from tensorboard.backend.event_processing import event_accumulator

_TAG = "Episode_Metrics/curriculum/assistance_force"


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
    return events[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="Run directory or run-dir name.")
    ap.add_argument("iteration", type=int, help="Iteration (checkpoint number).")
    ap.add_argument("--logdir", default="logs/rsl_rl/g1_getup",
                    help="Root to resolve a bare run name.")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run, args.logdir)
    ev = resolve_event_file(run_dir)
    ea = event_accumulator.EventAccumulator(ev, size_guidance={"scalars": 0})
    ea.Reload()

    if _TAG not in set(ea.Tags().get("scalars", [])):
        # Curriculum disabled or metric never logged → no assist.
        print("0")
        return

    vals = ea.Scalars(_TAG)
    # Pick the sample whose step is closest to the requested iteration.
    best = min(vals, key=lambda s: abs(s.step - args.iteration))
    print(f"{best.value:.1f}")


if __name__ == "__main__":
    main()
