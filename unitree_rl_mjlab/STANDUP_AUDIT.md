# G1 Get-Up Standup Audit

Date: 2026-07-07. Scope: full pipeline audit (rewards, geometry, observations/
actions, RL algorithm) while a fresh training run (with today's assist-force
and shank-orientation fixes) is in flight on the server. Every finding below
is backed by a code reference, a captured simulation-probe number, or a
literature citation — no unbacked claims. All probes ran headless on CPU
(`conda env unitree_rl_cpu`) and did not touch the server run.

---

## Executive summary

> **Update 2026-07-07 (after run `2026-07-07_12-08-06_Shank_fixed_resume`):**
> The morning's fixes worked — `contact/feet` rose 0.22→0.68 while
> `contact/torso` fell 0.11→0.086 (torso-lean exploit gone), and `stage2`
> (rising) jumped from ~3% to ~45% of episodes. But a rise-then-crash appeared:
> `ever_stood` peaked 29% (it1180) then declined to ~4%, tracking BOTH curricula
> collapsing together (`assistance_force` 111→2 N, `beta_rescaler` 0.97→0.76)
> while `stable_hold`≈0 — the predicted **F2** signature. **Two fixes applied
> since:** (1) F2 — beta-rescaler now uses the feet-planted qualifier (✅);
> (2) F11 — the SEATED reset pose, found to topple during settling, was removed
> (✅). Both verified locally. See F2 and F11 below.

The pipeline is fundamentally sound and closely tracks its reference (HoST,
arXiv:2502.08378) on every hyperparameter that matters for stability (γ, λ,
clip ratio, stage thresholds, tolerance-function shapes, advantage-combination
formula all match exactly). Remaining findings (F1 cliff, F3 finger noise, F4
friction, F5 entropy) are held for *if/when* the next run's data implicates
them — they are not yet evidenced as the binding constraint.

The four highest-leverage findings, ranked:

1. **F1 — Reward cliff, not reward valley.** A custom simulation probe
   (interpolating pelvis height + posture from three different lying poses up
   to standing) shows the reward landscape is monotonically increasing — there
   is no local optimum at "seated" — but it is **extremely non-uniform**: ~85%
   of the total achievable reward is packed into the last ~15% of the height
   range (h≈0.63→0.69 m), because `post_task`'s hard gate band is only 2 cm
   wide. This is a narrow, multi-condition target, not a smooth slope, and is
   probably the single biggest obstacle to a clean standup.
2. **F2 — The beta-rescaler curriculum has the exact bug we fixed in the
   assist-force curriculum today, but in the second copy of the same logic.**
   It reduces the policy's action authority based on a single-frame torso
   height check, no feet-planted/hold requirement. Your own training numbers
   confirm it's already firing on false positives (β=0.85 while
   `stable_hold`≈0).
3. **F3 — The 14 finger joints add ~50% action-saturation noise for free.**
   Confirmed analytically and against the HoST reference (23 DOF, no hands).
4. **F9 (informational, good news) — "Enhancement 1: LCP" in your
   `Next_Step.md` roadmap is already implemented** under the name `L2C2` in
   `multi_critic_ppo.py`. Don't re-implement it; update the tracker and move
   to Enhancement 2.

---

## Findings, ranked by severity

### F1 [HIGH] — `post_task`'s reward gate is a cliff, not a ramp — ✅ FIXED 2026-07-07

**Update (2026-07-07, after run `2026-07-07_14-50-55` completed its 2000 iters):**
The cliff was confirmed as the binding constraint by live training data. With F2
fixed (β preserved at 0.99), the run converged to a **stable feet-planted crouch
at ~0.5-0.6 m**: as `assistance_force` decayed 69→44 N, `ever_stood` crashed
0.125→0.027 and `stage3`→0.001, while `post_stand_on_feet` (gate 0.50) ROSE to
0.72 and `mean_reward` stayed flat/up — the policy scored higher by crouching
than by risking the unrewarded climb through the 4 cm cliff. **Fix applied:**
`_post_gate` band 0.02 → 0.08 (ramp now [0.57, 0.73] — gradient starts at the
top of the crouch), and `style_foot_displacement`'s hard step-gate at 0.65 m
replaced with the same 0.08-band ramp (it was the residual +7.8 discontinuity).
Probe re-run confirms: the former +60-in-6cm jump is now +24/+28/+18 across
18 cm, endpoints unchanged (standing +94.7, lying ~+3). Band history is a real
tradeoff — 0.06 caused ballistic jumping (commit 71e6646), 0.02 caused the
crouch; 0.08 leans toward "climb" with the co-gated `post_base_lin_vel`
(weight 15) now covering the climb zone to damp jumping. **Watch item:** if the
next run shows the old jump behavior (violent rises, `task_head_impact`
spikes), dial band back toward ~0.05. Original analysis below.

---

**Where:** `_post_gate(asset, H_STAGE2=0.65, band=0.02)` in
`src/tasks/getup/mdp/rewards.py:1271-1282`, applied to 10 `post_task` reward
terms worth up to **49.95** of the ~94.7 max achievable NET reward
(`post_base_ang_vel` 10, `post_base_lin_vel` 15, `post_base_orientation` 10,
`post_base_height` 10, `post_feet_parallel` 2.5, `post_feet_yaw` 2.5,
`stable_success_hold` 10 not included in static since it needs a hold). With
`band=0.02`, the ramp spans only **[0.63 m, 0.67 m]** — a 4 cm window.
`style_foot_displacement` (weight 4, hard-gated the same way) and
`style_ankle_parallel` (weight 1, but reaches +19.99 at full standing) add
further weight concentrated in the same narrow band.

**Evidence — custom interpolation probe** (script + full output in
Appendix A; interpolates pelvis height/orientation/joints linearly between a
resting pose and HOME, evaluates every real reward term via the actual
`RewardManager` at each step):

| t (fraction of path) | SEATED→HOME pelvis z | NET reward | SUPINE→HOME NET | PRONE→HOME NET |
|---|---|---|---|---|
| 0.0 | 0.155 | +12.70 | +5.01 | +6.27 |
| 0.4 | 0.393 | +13.36 | +13.71 | +13.36 |
| 0.7 | 0.572 | +24.07 | +17.40 | +23.83 |
| 0.8 | 0.631 | +30.55 | +21.06 | +29.27 |
| **0.9** | **0.691** | **+90.41** | **+84.16** | **+89.38** |
| 1.0 | 0.750 | +94.70 | +94.70 | +94.70 |

All three paths (from seated, supine, and prone) show the **same shape**: a
gentle, monotonic slope from t=0 to t=0.8 (NET grows 2-3×), then a **+60-point
jump in the single next step** (t=0.8→0.9, pelvis 0.63→0.69 m) as
`post_base_*`, `post_feet_parallel/yaw`, `style_foot_displacement`, and the
remainder of `style_ankle_parallel` all switch on together. This confirms the
gate band, not the term math, is the issue — nothing is mathematically wrong
with any individual term (each fires on the pose you'd want, per the
`diag_rewards.py` static/violation tables in Appendix A), but the *combination*
of several independently-narrow gates at the same height creates one big
combined cliff instead of several overlapping ramps.

**Why this matters for exploration:** a policy using Gaussian action noise
that reaches h≈0.60-0.63 m (already a real achievement — past `H_STAGE1` and
most of the way up) sees almost no local reward signal telling it *which
direction* to keep pushing, because the big payoff only appears once it
simultaneously satisfies height, upright orientation, feet parallel, feet-under-
body, and ankle-flatness all at once. This is a classic "near-sparse reward
behind smooth shaping" problem — the dense terms get you 80% of the way, then
you're back to needing a lucky multi-condition conjunction for the last 20%.
This is consistent with your video ("activates a lot... doesn't reach standing,
sometimes ends seated") and is a more precise, evidence-backed version of your
own initial hypothesis about sitting being a trap — it's not that sitting
itself is rewarding (SEATED's *own* static reward, +12.7, is much smaller than
standing's +94.7 — a 7.5× difference), it's that the region *between* sitting
and standing has a cliff at the very end that's hard to find by local search.

**Recommended fix (only if the current run fails to cross into standing):**
widen `_post_gate`'s band from 0.02 to something closer to `_standing_gate`'s
own default (0.08-0.10), so the post-task payoff ramps in gradually over
~15-20 cm instead of unlocking almost all at once in 4 cm. This is a one-line
parameter change (same category as today's `style_shank_orientation` fix —
no new reward term). Also consider: since the assist-force curriculum's job
is precisely to push the robot through this cliff often enough to learn it,
today's fix (feet-planted decay qualifier) is *more* important given this
finding, not less — the curriculum is the mechanism papering over the cliff
during early training, so keeping it honest matters even more.

### F2 [HIGH] — Beta-rescaler curriculum has the assist-force bug — ✅ FIXED 2026-07-07

**Update (2026-07-07, after the resume run `2026-07-07_12-08-06_Shank_fixed_resume`):**
This was confirmed live and fixed. The run showed the predicted signature —
`ever_stood` peaked at 29% (it1180) then *declined* to ~4% by it1998, and the
decline tracked `assistance_force` (111→2 N) AND `beta_rescaler` (0.97→0.76)
collapsing together right after the peak, while `stable_hold` stayed ≈0
throughout: both crutches were withdrawn on transient successes before genuine
standing consolidated. The feet-planted qualifier from `AssistanceCurriculum`
was applied to `BetaRescalerCurriculum` (`mdp/events.py`, now via the shared
`_both_feet_planted`/`_resolve_feet_planted_cfg` helpers). Beta now only
decrements when `success_head_height` is reached AT THE SAME STEP as both feet
planted. Verified with a targeted unit test (head-spike-without-feet → no
decay; genuine feet-planted stand → decays 1.0→0.98; feet-planted-but-low-head
→ no decay). Original analysis retained below for reference.

---

**Where:** `BetaRescalerCurriculum.__call__`/`reset`, `mdp/events.py:439-457`.

```python
def __call__(self, env, env_ids=None, **kwargs):
    h_head = self._asset.data.body_link_pos_w[:, self._body_id, 2]  # torso_link
    self._reached_success |= h_head >= self._success_head_height     # 0.75 m, ANY frame
```

This is the exact same shape as the assist-force bug fixed earlier today
(single-frame torso height, no feet-planted check, no hold requirement) — just
in the second curriculum that shares the pattern. β directly controls action
authority (`p_target = q_default + β·a`); decaying it on a false positive
doesn't just remove a helpful force, it **permanently reduces how much the
policy is allowed to move away from the default pose** while it hasn't
actually learned to stand. Your pasted training log already shows this firing:
`Episode_Metrics/curriculum/beta_rescaler` down to 0.8501 by it1998 while
`success/stable_hold` is ~0 throughout.

**Literature check:** HoST's own paper uses the identical instantaneous
criterion for *both* curricula ("the head height must reach a target height by
the end of each episode") — so this matches the reference design, not a
deviation. However, HoST's own version checks height **at episode end**, not
**peak-during-episode** — end-of-episode is implicitly harder to game with a
transient mid-episode spike (the robot has to still be there when time runs
out). The assist-force code we fixed today used peak-over-episode (strictly
easier to satisfy than HoST's own check), and `BetaRescalerCurriculum` still
does. Either matching HoST exactly (check height at episode end) or applying
today's feet-planted qualifier (stricter than HoST, consistent with the fix
we already made) would close this gap — the second is more consistent with
what's already in the codebase.

**Recommendation:** if the current run shows β decaying materially faster than
`stable_hold`/`ever_stood` rise (as your last run already did), apply the same
feet-planted-qualified peak-height pattern from `AssistanceCurriculum` to
`BetaRescalerCurriculum`.

### F3 [MEDIUM-HIGH] — 43-DOF action space: fingers add ~50% saturation noise for free — ✅ FIXED 2026-07-08

**Update (2026-07-08):** Implemented. Action space is now 29-DOF (fingers
excluded via `actuator_names=("^(?!.*_hand_).*$",)`); a new zero-action-dim
`HandHoldAction` term (`mdp/actions.py`) PD-holds the 14 finger joints at a
fixed, slightly-curled pose (`HAND_HOLD_JOINT_POS` in `g1_constants.py` — the
Dex3 middle/index/thumb_2 joints have q=0 as a RANGE BOUNDARY, sign-mirrored
left/right, so they hold 0.2-0.3 rad into range instead of at the hard stop).
`reg_hand_vel` reward removed (its job is now the hold actuator's damping).
**Bonus root-cause discovery:** the finger actuator group had NO armature
configured → numerically near-massless joints → any sustained PD error
produced ~1e5 rad/s² and diverged (measured: finger qvel → 9e5 rad/s in 20
steps). This was also the source of the sporadic `joint_vel_explosion`
terminations (~2/iteration) under the old 43-DOF policy. `G1_ACTUATOR_HAND_HOLD`
sets `armature=0.004` (≈ the 4010 wrist motor); a 300-step contact-heavy probe
now shows 0/300 flagged steps vs 2/300 on the pre-change code. Verified:
env builds (`joint_pos: 29`, `hand_hold: 0`), fingers track hold pose within
0.3 rad through ground contact, `diag_rewards.py` clean, 2-iteration PPO smoke
run trains a 29-dim actor with no NaN. Velocity/tracking tasks untouched
(`G1_ARTICULATION` still uses the original `G1_ACTUATOR_HAND`). Old
checkpoints are NOT resumable (actor in/out dims changed) — fresh run required,
which matches the already-decided curricula reset. Original analysis below.

---

**Original finding:**

**Where:** `src/tasks/getup/mdp/actions.py` (43-dim action = 29 body + 14
finger DOF, confirmed via `total_action_dim=43` and XML actuator groups),
`rl_cfg.py:21-29` (`init_std=1.5`, `std_type="scalar"` — one shared std across
all 43 dims), `reg_hand_vel`/`reg_arm_vel` weight = 0 (Phase 1, unshaped).

**Evidence:** `P(|N(0, 1.5)| > 1.0) = 50.5%` (direct calculation — the ±1 clamp
in `actions.py` is hit by just over half of all raw Gaussian action samples at
any point where the policy's mean is near zero, which is most of early
training for the 14 unshaped finger dims specifically, since nothing ever
pulls their mean away from init). This means roughly half of all finger-joint
commands are hard-clamped bang-bang noise throughout training, contributing
to `action_rate_l2`/`action_acc_l2` penalties (small weight, -0.002/-0.003,
but nonzero), consuming PPO's action-space exploration budget, and adding
high-frequency noise to the PD-tracking error the critic observes — all while
contributing nothing to standing (hands touch the ground/torso in some poses,
which the hand-contact-force reward *does* respond to, but finger *pose* itself
is unshaped).

**Literature comparison:** HoST's own G1 has **23 actuated DOF — no hand/finger
joints at all** (confirmed via direct paper fetch: "6 per leg, 5 per arm, and 1
in the waist"). The 43-DOF unified policy is a deliberate extension documented
in your `Next_Step.md` ("Part 1 — XML Migration," appears complete already),
not a mistake — but it means this pipeline is carrying a real, quantifiable
exploration-noise cost the reference method didn't have.

**Recommendation:** not urgent while the run is in flight (this affects
sample efficiency, not correctness). If a future run needs a nudge: enable
`reg_hand_vel`/`reg_arm_vel` at a small nonzero weight (already implemented,
just gated to 0 for "Phase 1"), or lower `init_std` back toward 1.0 for a
later training phase once the settling/curriculum fixes have made early
exploration less of a bottleneck (also see F5 — raising entropy instead may
be the more surgical lever).

### F4 [MEDIUM] — Hand/torso friction favors propping over foot-only balance

**Where:** `FULL_COLLISION` config, `g1_constants.py` — hand friction `(1.5,
0.01, 0.001)` vs foot friction `(0.6,)`; explicitly documented as intentional
("stand-up-critical support contacts... higher sliding friction so the robot
can push off without slipping"). No torso-contact penalty exists (only
head-contact and hand-contact, the latter weight -0.05 and reads exactly 0 at
all 6 canonical poses per the static reward table — it essentially never
fires under normal posture).

**Assessment:** this is a real, quantifiable mechanical incentive toward
hand/torso support (2.5× the friction of feet) combined with a contact penalty
too weak to counteract it at any of the poses actually visited. This was
already identified and deliberately **held in reserve** earlier today (a
torso-contact penalty was designed but not applied, per your request to keep
changes minimal and attributable). This audit doesn't change that
recommendation — just confirms the mechanical basis for it is real and
quantifiable, for when/if it's needed.

### F5 [MEDIUM] — Entropy coefficient is half the reference value

**Where:** `entropy_coef=0.005` (`rl_cfg.py:52`) vs HoST's published `0.01`
(literature-confirmed). Combined with F3's elevated `init_std=1.5` (raised
specifically "for wider initial exploration"), this is an internally
inconsistent pair: exploration is pushed up via the initial std, but the
ongoing incentive to *maintain* exploration (entropy bonus) is lower than the
paper that validated the rest of this hyperparameter set. Not necessarily
wrong — collapsing entropy over training is expected and desired eventually —
but worth a deliberate choice rather than an unnoticed halving.

### F6 [LOW-MEDIUM, informational] — Per-group advantage normalization on sparse groups

**Where:** `multi_critic_ppo.py:104-125` — `post_task`'s advantages are
normalized by their own std before weighting, and early in training that
group is near-always-zero (gated at h>0.63 m). Normalizing by a near-zero std
can amplify whatever rare, noisy signal exists in that group.

**This is not a deviation** — HoST's own paper uses the identical formula
(confirmed: `A = Σ_i w_i · (A_φᵢ − μ_A_φᵢ)/σ_A_φᵢ`, literature-fetched
directly from the paper). It's a known, shared property of the method, not a
bug in this implementation. Worth knowing as context for F1: the more the
`post_task` cliff is crossed only rarely, the noisier this normalization gets,
which is another reason widening the F1 gate band would likely help overall
training stability, not just reward shaping.

### F7 [LOW, informational] — Credit-assignment horizon is short but matches the reference exactly

`γ=0.99`, `λ=0.95`, 50 Hz control (both exactly match HoST). Effective GAE
horizon ≈ 1/(1-γ) ≈ 100 steps (~2s), shorter than the multi-second rise many
starting poses require end-to-end. Not a bug (matches the validated
reference), but explains *why* HoST/this pipeline lean so heavily on dense
potential-shaped rewards (`height_progress`, `prone_recovery`,
`supine_rising_prep`) and the force/β curricula — they're compensating for a
genuinely short raw temporal credit horizon, not incidental extras. Reinforces
that F2's curriculum integrity is load-bearing, not cosmetic.

### F8 [context, already resolved earlier today] 

`AssistanceCurriculum` feet-planted decay qualifier and `style_shank_orientation`
ground-up gate (`height_threshold=0.0`) were implemented and verified (env
build + 40-step rollout, no NaN, gate-value probe) in this same session,
before this audit began. `thigh_orientation` (unwired) and a torso-contact
penalty were designed but deliberately held in reserve per your request to
keep the change minimal and attributable to a single cause.

### F9 [informational, good news] — "LCP" enhancement is already implemented as L2C2

**Where:** `Next_Step.md` "Enhancement 1 — LCP (Lipschitz-Constrained
Policies)" cites arXiv:2410.11825 (Chen/He, *Learning Smooth Humanoid
Locomotion through Lipschitz-Constrained Policies*) as a **planned, not-yet-
done** next step. But `multi_critic_ppo.py:210-232` already implements exactly
this mechanism under the name **`L2C2`**: interpolate consecutive states
s̃ = s + u·(s_{t+1}−s_t), u~U(0,1), and penalize the L2 change in the actor's
mean output (`l2c2_actor_coef=1.0`) and each of the 4 critics' value outputs
(`l2c2_critic_coef=0.1`) between s and s̃ — a differentiable Lipschitz
gradient penalty on the policy, added to the loss every update. This is the
same family of technique the cited paper describes (gradient-penalty-based
Lipschitz bound on policy output vs. input), already wired into training and
already contributing to the smoothness (α=0.5 EMA + action_rate/acc penalties
+ L2C2 are three complementary, not redundant, smoothness mechanisms:
EMA/low-pass shapes the *command signal*, action_rate/acc penalizes the
*realized* joint motion, L2C2 constrains the *policy function* itself).

**Recommendation:** update `Next_Step.md`'s progress tracker to mark LCP done,
and re-scope near-term roadmap effort to Enhancement 2 (balance-embedding
asymmetric critic / Capture Point) as the actual next new item — no need to
re-implement or add a second Lipschitz mechanism.

### F10 [LOW, hygiene]

- `diag_contact_geoms.py` and `pose_lab.py` still load the old 29-DOF
  `unitree_g1/xmls/scene_g1.xml`; `visualize_keyframes.py` has hardcoded, stale
  keyframe values (HOME z=0.8/hip=-0.1/knee=0.3 vs the real HOME z=0.75/
  hip=-0.312/knee=0.669). No training impact — matches the already-deferred
  "Part 1 Step 4" item in `Next_Step.md`.
- `diag_reset_distribution.py`'s printed comment ("expect keyframe_z + 0.40
  ± 0.03") is stale relative to the actual `_RESET_FALL_HEIGHT=0.03` config —
  confirmed by this session's own probe run: observed spawn pelvis heights
  (median 0.223 m across mixed poses, min 0.108 m) are consistent with
  keyframe_z + 0.03-0.07 m, not +0.40 m. Cosmetic only; the actual reset
  behavior (Appendix B) is healthy — 100% grounded after 80 settle steps, no
  NaN, only 1/64 envs approached the head-impact penalty threshold and 0/64
  hit the termination threshold.

### F11 [HIGH] — SEATED reset pose is not a static equilibrium — ✅ REMOVED 2026-07-07

**Where:** `SEATED_KEYFRAME` (`g1_constants.py`), previously in the reset
keyframe set (`config/g1/env_cfgs.py`). SEATED spawns the robot in a deep
hip/knee flexion (hip_pitch −2.24, knee 1.87) at pelvis z=0.155 m with **no
back support** — the torso is upright but nothing holds it there.

**Evidence — settle probe** (`scratchpad/probe_settle.py`, this session; spawns
each keyframe and steps zero policy actions for 60 steps reproducing the real
settle mechanism, logging height + projected-gravity-z each step):

| step | SEATED pelvis z | SEATED gz (−1=upright, 0=lying) |
|---|---|---|
| 0 (spawn) | 0.155 | −0.970 |
| 25 | 0.140 | −0.886 |
| 30 (policy handover) | 0.130 | **−0.752 (still changing fast)** |
| 35 | 0.143 | **−0.050** |
| 40+ | ~0.074 | ~−0.06 (settled ≈ supine) |

SEATED topples backward *through and past* the 30-step settling window: gz is
still moving ~0.13/step at the handover step, and by step 35 the robot is
essentially flat on its back. By contrast SUPINE/PRONE/SIDE settle within
~5-10 steps with small bounded oscillation. So for the ~20-25% of episodes
that spawned SEATED, the policy took control mid-topple with real angular
momentum — a direct, specific explanation for the "huge initial movement in
some positions" seen in the video (it was this one pose, every time). This is
an **environment correctness bug** (the initial condition isn't the condition
it claims to be by handover), not a reward-shaping issue — no reward tuning
fixes it.

**Action taken:** SEATED removed from the reset keyframe set (4 ground-lying
poses remain: SUPINE, PRONE, SIDE_LEFT, SIDE_RIGHT). Per user decision, SEATED
will be re-added later, once the scene supports back-support / props / varied
terrain that make a seated start statically stable — at which point the fix is
to design a genuinely settle-stable seated pose (hands planted for support, or
a CoM-over-base configuration), not merely to lengthen `_SETTLE_STEPS` (which
would just silently convert SEATED episodes into worse-than-SUPINE ones).

---

## Literature comparison table (this pipeline vs. HoST, arXiv:2502.08378)

| Parameter | This pipeline | HoST (paper-confirmed) | Verdict |
|---|---|---|---|
| Actuated DOF | 43 (29 body + 14 fingers) | 23 (legs+arms+waist, no hands) | Deliberate extension — see F3 |
| Initial assist force | 120 N | 200 N | Ours lower; not inherently wrong |
| Force decay / success | 5 N, **feet-planted-qualified peak height** (fixed today) | 20 N, **instantaneous height at episode end** | Ours now stricter than reference (good) |
| Force floor | 0 N | 0 N | Match |
| β start / decrement / floor | 1.0 / 0.02 / 0.25 | 1.0 / 0.02 / 0.25 | Exact match on shape; **success check still matches HoST's simple instantaneous form — see F2** |
| Discount γ | 0.99 | 0.99 | Match |
| GAE λ | 0.95 | not published | — |
| PPO clip ratio | 0.2 | 0.2 | Match |
| Entropy coef | 0.005 | 0.01 | **Ours is half — F5** |
| Episode length | 500 steps (10 s) | 500 steps | Match |
| Control / physics rate | 50 Hz / 200 Hz | 50 Hz / 200 Hz | Match |
| `task_head_height` f_tol | lower=0.794, margin=1, v=0.1 | lower=1.0 (different body ref), margin=1, v=0.1 | Shape matches; threshold differs only because of the torso-proxy reference height |
| `task_base_orientation` f_tol | lower=0.99, margin=1, v=0.05 | lower=0.99, margin=1, v=0.05 | Exact match |
| Stage thresholds | H_stage1=0.45, H_stage2=0.65 | H_stage1=0.45, H_stage2=0.65 | Exact match |
| Reward groups | task/style/regularization/post_task | task/style/regularization/post-task | Same structure/names |
| Advantage combination | per-group GAE → per-group normalize → weighted sum | identical formula (paper-quoted) | Exact match — see F6 |
| Group weights | 2.5 / 2.2 / 0.1 / 1.0 | not published | — |
| Smoothness mechanism | EMA(α=0.5) + action_rate/acc penalty + L2C2 | "smoothness regularization + implicit motion speed bound" (general) | L2C2 ≡ planned "LCP" — see F9 |

---

## Decision tree for the current run

Check these in order once the run has enough iterations to judge (~1500-2000,
matching the log you shared earlier):

1. **`contact/feet` rising and `contact/torso` flat-or-falling, `ever_stood`
   climbing past ~15-20%?** → Today's two fixes are working. Let it continue;
   no action needed yet.
2. **Robot reaches h≈0.55-0.63 m regularly (past `H_STAGE1`, near
   `H_STAGE2`) but `stable_hold`/`ever_stood` stay flat, and `beta_rescaler`
   keeps decaying anyway?** → This is F2 (β curriculum false-positive).
   Apply the feet-planted qualifier to `BetaRescalerCurriculum`, matching
   today's `AssistanceCurriculum` fix.
3. **Robot regularly approaches but never crosses ~0.63-0.65 m, and
   `assistance_force` stays high (correctly, per today's fix) without
   `ever_stood` improving over many iterations?** → This is F1 (the cliff).
   Widen `_post_gate`'s `band` from 0.02 to ~0.08-0.10.
4. **Robot props on torso/hands persistently in the video even once height
   is achievable?** → Apply the reserved torso-contact penalty (F4) and/or
   wire `thigh_orientation` (both already designed in the prior session's
   plan, held back to keep changes attributable).
5. **If none of the above and the robot just isn't exploring toward standing
   at all (height barely changes from spawn)?** → Check F5 (raise entropy_coef
   toward 0.01) before touching reward math further.

Apply one change at a time per your own "change one thing per training run"
rule in `Next_Step.md`.

---

## Appendix A — Reward diagnostics (raw output)

Full outputs captured in this session's scratchpad
(`/tmp/claude-1000/.../scratchpad/`):
- `diag_rewards_static_full.txt` — `scripts/diag_rewards.py` default run:
  per-term weighted contribution at all 6 canonical poses, group totals,
  firing-range scan, style-violation sanity check, ankle-parallel sweep.
- `probe_landscape_out.txt` — custom `probe_landscape.py` (this audit):
  interpolated SEATED→HOME, SUPINE→HOME, PRONE→HOME reward-landscape scan
  (the F1 evidence table above is extracted from this).
- `diag_obs.txt` — `scripts/diag_observations.py`: per-term obs stats, no
  NaN/Inf, `last_action` saturates at ±1 under random-action sampling
  (expected).
- `diag_reset.txt` — `scripts/diag_reset_distribution.py --num-envs 64
  --settle-steps 80`: keyframe-selection uniformity, spawn height
  distribution, settle-health (100% grounded, no NaN, 1/64 envs near
  head-impact penalty threshold, 0/64 terminated).

Key numbers already inlined above; re-run any of these locally with:
```bash
cd unitree_rl_mjlab && conda activate unitree_rl_cpu
python scripts/diag_rewards.py               # static + firing-range tables
python scripts/diag_rewards.py --steps 100   # zero-action time series
python scripts/diag_observations.py
python scripts/diag_reset_distribution.py --num-envs 64
```

## Appendix B — Pulling a server checkpoint for a frame-by-frame video review

No checkpoint from the current (2026-07-07) run exists on this local machine
— training is running on the server. To get a rollout video for visual
inspection once a checkpoint is available:

```bash
# From your local machine, pull the latest checkpoint down (adjust host/path):
scp <server>:~/HumanUP/unitree_rl_mjlab/logs/rsl_rl/g1_getup/<run_dir>/model_*.pt \
    /home/aidapt/HumanUP/unitree_rl_mjlab/logs/rsl_rl/g1_getup/<run_dir>/

# Then render headlessly (no display needed) to an mp4:
cd /home/aidapt/HumanUP/unitree_rl_mjlab
conda activate unitree_rl_cpu
MUJOCO_GL=osmesa python scripts/play.py Unitree-G1-GetUp \
    --checkpoint-file logs/rsl_rl/g1_getup/<run_dir>/model_<N>.pt \
    --video --video-length 300 --num-envs 4 --device cpu
# Output: logs/rsl_rl/g1_getup/<run_dir>/videos/play/*.mp4
```

Send me that path (or the mp4 itself) and I'll review the contact pattern,
joint saturation, and exactly where the rollout stalls as a follow-up
addendum to this audit.
