"""Diagnostic: inspect get-up observations (actor/critic dims, history, NaN).

Run:  python scripts/diag_observations.py
Builds the G1 get-up env on CPU with a few envs and prints:
  - per-term shapes for actor and critic groups (mjlab ObservationManager table)
  - total concatenated dims + how the actor history multiplies them
  - actor/critic config (history_length, corruption, nan_policy, normalization)
  - PER-TERM statistics over reset + random steps: mean/std/min/max/absmax,
    fraction saturated at the configured clip, and NaN/Inf counts.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from prettytable import PrettyTable

from mjlab.envs import ManagerBasedRlEnv

from src.tasks.getup.config.g1.env_cfgs import unitree_g1_getup_env_cfg
from src.tasks.getup.config.g1.rl_cfg import unitree_g1_getup_ppo_runner_cfg

N_ENVS = 4
N_STEPS = 50


def _term_slices(om, group: str):
    """Yield (term_name, start, length, clip) for a concatenated group."""
    idx = 0
    names = om.active_terms[group]
    dims = om.group_obs_term_dim[group]
    for name, shape in zip(names, dims):
        length = int(np.prod(shape))
        clip = om.get_term_cfg(group, name).clip
        yield name, idx, length, clip
        idx += length


class TermStats:
    """Streaming per-term stats: mean, std, min, max, |x|>clip frac, NaN/Inf."""

    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = float("inf")
        self.max = float("-inf")
        self.absmax = 0.0
        self.clipped = 0
        self.nan = 0
        self.inf = 0

    def update(self, x: torch.Tensor, clip):
        nan = torch.isnan(x)
        inf = torch.isinf(x)
        self.nan += int(nan.sum())
        self.inf += int(inf.sum())
        finite = x[~(nan | inf)]
        if finite.numel() == 0:
            return
        self.n += finite.numel()
        self.sum += float(finite.sum())
        self.sumsq += float((finite * finite).sum())
        self.min = min(self.min, float(finite.min()))
        self.max = max(self.max, float(finite.max()))
        self.absmax = max(self.absmax, float(finite.abs().max()))
        if clip is not None:
            lo, hi = clip
            self.clipped += int(((finite <= lo) | (finite >= hi)).sum())

    def row(self, name, length, clip):
        mean = self.sum / self.n if self.n else 0.0
        var = self.sumsq / self.n - mean * mean if self.n else 0.0
        std = float(np.sqrt(max(var, 0.0)))
        clip_str = f"{100*self.clipped/self.n:.2f}%" if (clip and self.n) else "-"
        return [
            name, length,
            f"{mean: .3f}", f"{std: .3f}",
            f"{self.min: .3f}", f"{self.max: .3f}", f"{self.absmax: .3f}",
            clip_str, self.nan, self.inf,
        ]


def main() -> None:
    cfg = unitree_g1_getup_env_cfg()
    cfg.scene.num_envs = N_ENVS
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
    om = env.observation_manager

    print("\n" + "=" * 70)
    print("OBSERVATION TERM TABLE (per-term shapes)")
    print("=" * 70)
    print(om)

    print("=" * 70)
    print("GROUP TOTAL DIMS (concatenated, what the network receives)")
    print("=" * 70)
    for group, dim in om.group_obs_dim.items():
        print(f"  {group:8s} -> {dim}")

    print("\n" + "=" * 70)
    print("GROUP CONFIG (history / corruption / nan_policy)")
    print("=" * 70)
    for group in ("actor", "critic"):
        g = om.cfg[group]
        print(
            f"  {group:8s} history_length={g.history_length} "
            f"corruption={g.enable_corruption} nan_policy={g.nan_policy} "
            f"nan_check_per_term={g.nan_check_per_term} concat={g.concatenate_terms}"
        )
    rl = unitree_g1_getup_ppo_runner_cfg()
    print(f"\n  actor.obs_normalization  = {rl.actor.obs_normalization}")
    print(f"  critic.obs_normalization = {rl.critic.obs_normalization}")

    # --- Per-term streaming stats over reset + random steps. ---
    stats = {g: {} for g in ("actor", "critic")}
    meta = {g: list(_term_slices(om, g)) for g in ("actor", "critic")}
    for g in stats:
        for name, _, _, _ in meta[g]:
            stats[g][name] = TermStats()

    def accumulate(obs_dict):
        for g in ("actor", "critic"):
            t = obs_dict[g]
            t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
            # History flattens each term as [t0..tH-1] contiguously per term, so the
            # per-term slice (length already includes history) is contiguous.
            for name, start, length, clip in meta[g]:
                stats[g][name].update(t[:, start : start + length], clip)

    obs, _ = env.reset()
    accumulate(obs)
    n_act = env.action_manager.total_action_dim
    done_steps = 0
    for _ in range(N_STEPS):
        act = torch.rand(N_ENVS, n_act, device="cpu") * 2 - 1
        try:
            obs, _, _, _, _ = env.step(act)
        except Exception as e:  # noqa: BLE001 — keep partial stats useful
            print(f"\n[WARN] env.step failed after {done_steps} steps: {e}")
            print("[WARN] Reporting stats from reset + completed steps only.\n")
            break
        accumulate(obs)
        done_steps += 1

    print("\n" + "=" * 70)
    print(f"PER-TERM STATS over reset + {N_STEPS} random steps ({N_ENVS} envs)")
    print("(random actions → values larger than a trained policy would produce)")
    print("=" * 70)
    for g in ("actor", "critic"):
        table = PrettyTable()
        table.title = f"Group '{g}'"
        table.field_names = [
            "term", "dim", "mean", "std", "min", "max", "absmax", "%clip", "NaN", "Inf",
        ]
        table.align["term"] = "l"
        total_nan = total_inf = 0
        for name, _, length, clip in meta[g]:
            s = stats[g][name]
            table.add_row(s.row(name, length, clip))
            total_nan += s.nan
            total_inf += s.inf
        print(table)
        print(f"  group '{g}': total NaN={total_nan} Inf={total_inf}\n")

    print("DONE.")


if __name__ == "__main__":
    main()
