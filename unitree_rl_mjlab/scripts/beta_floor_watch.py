#!/usr/bin/env python3
"""Watch a run's TensorBoard log and exit when beta hits its floor for long enough.

Companion to launch.sh's auto-kill: polls Episode_Metrics/curriculum/beta_rescaler
and exits with code 0 once the population-mean beta has stayed at (or below)
``--threshold`` for a contiguous span of at least ``--hold-iters`` iterations.
The shell wrapper in launch.sh then calls kill.sh on the run — at that point the
beta curriculum is finished, so whether the policy is doing well or badly there
is nothing left for this training phase to decide, and burning the weekend's
remaining --max-iterations changes nothing.

Exit codes:
  0  beta floor held for >= hold-iters -> caller should kill the training
  2  the event file stopped growing (training ended/crashed on its own) -> no kill

Usage:
    python scripts/beta_floor_watch.py <run_dir> [--hold-iters 75] [--poll 300]

<run_dir> may be a path or just the run-dir name (resolved under --logdir,
default logs/rsl_rl/g1_getup relative to cwd), same as print_rewards.py.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

from tensorboard.backend.event_processing import event_accumulator

TAG = "Episode_Metrics/curriculum/beta_rescaler"


def resolve_run_dir(run: str, logdir: str) -> str:
    if os.path.isdir(run):
        return run
    cand = os.path.join(logdir, run)
    if os.path.isdir(cand):
        return cand
    sys.exit(f"[beta_floor_watch] run dir not found: {run!r} (logdir={logdir})")


def newest_event_file(run_dir: str) -> str | None:
    files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    return max(files, key=os.path.getmtime) if files else None


def read_beta_scalars(event_file: str) -> list[tuple[int, float]]:
    """[(step, value), ...] sorted by step; empty if the tag is not logged yet."""
    acc = event_accumulator.EventAccumulator(
        event_file, size_guidance={event_accumulator.SCALARS: 0}
    )
    acc.Reload()
    if TAG not in acc.Tags().get("scalars", []):
        return []
    return sorted((s.step, s.value) for s in acc.Scalars(TAG))


def floor_tail_span(scalars: list[tuple[int, float]], threshold: float) -> int:
    """Iteration span of the CONTIGUOUS tail with beta <= threshold (0 if none).

    Contiguity matters: a dip to the floor followed by a breaker re-ramp above
    it must reset the count, so only a genuinely settled floor triggers the kill.
    """
    if not scalars or scalars[-1][1] > threshold:
        return 0
    first_of_tail = len(scalars) - 1
    while first_of_tail > 0 and scalars[first_of_tail - 1][1] <= threshold:
        first_of_tail -= 1
    return scalars[-1][0] - scalars[first_of_tail][0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="run dir path or name")
    ap.add_argument("--logdir", default="logs/rsl_rl/g1_getup")
    ap.add_argument("--hold-iters", type=int, default=75,
                    help="contiguous iterations at the floor before exiting 0")
    ap.add_argument("--threshold", type=float, default=0.2505,
                    help="beta counts as 'at the floor' at/below this "
                         "(slightly above the 0.25 min for float tolerance)")
    ap.add_argument("--poll", type=float, default=300.0,
                    help="seconds between checks")
    ap.add_argument("--stale-polls", type=int, default=6,
                    help="exit 2 (no kill) after this many polls without a new "
                         "logged iteration — training ended on its own")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run, args.logdir)
    print(f"[beta_floor_watch] watching {run_dir}", flush=True)
    print(f"[beta_floor_watch] kill condition: beta <= {args.threshold} for a "
          f"contiguous span of >= {args.hold_iters} iterations", flush=True)

    last_step = -1
    stale = 0
    while True:
        event_file = newest_event_file(run_dir)
        scalars = read_beta_scalars(event_file) if event_file else []
        if scalars:
            step, value = scalars[-1]
            span = floor_tail_span(scalars, args.threshold)
            print(f"[beta_floor_watch] it{step}: beta={value:.4f} "
                  f"floor-span={span}/{args.hold_iters}", flush=True)
            if span >= args.hold_iters:
                print(f"[beta_floor_watch] beta at floor for {span} iterations "
                      "-> curriculum finished, requesting kill", flush=True)
                sys.exit(0)
            stale = stale + 1 if step == last_step else 0
            last_step = step
        else:
            print("[beta_floor_watch] beta tag not logged yet, waiting...",
                  flush=True)
            stale = 0  # startup grace: don't count pre-first-log polls as stale
        if scalars and stale >= args.stale_polls:
            print(f"[beta_floor_watch] no new iterations for {stale} polls — "
                  "training ended on its own, exiting without kill", flush=True)
            sys.exit(2)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
