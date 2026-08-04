"""Structured configuration for the from-scratch staged stand-up curriculum (V2).

All thresholds, level lists, probabilities, and window sizes for the curriculum live
HERE as dataclass fields — nothing is hard-coded inside the implementation functions
(rewards/events/actions read these). This is the single source of truth referenced by:
  * the shared standing evaluator (mdp/standing.py),
  * the global curriculum state machine (mdp/curriculum_fsm.py, M2),
  * the incremental action term (mdp/actions.py IncrementalJointPositionAction),
  * the reward gating / post-task reward set,
  * logging.

Promotion/rollback criteria use the TIGHTENED, iteration-based windows (user decision
2026-08-03): with 2-4k parallel envs a 256-EPISODE window fills in <1 policy iteration,
so "3 consecutive windows" would promote every ~2 iterations off a noisy sample. Windows
are therefore measured in ITERATIONS (with an episode-count floor for statistical mass),
and thresholds are stricter than the original spec (success 0.85 / fall 0.10) because the
chronic failure mode is post-stand collapse, not failure to reach height.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ======================================================================================
# 2. Stability — one unified definition of standing (shared by ALL subsystems)
# ======================================================================================
@dataclass
class StabilityCfg:
    # --- 2.1 standing-candidate (entry into post-task phase) ---
    candidate_height: float = 0.62            # h_pelvis > 0.62 m
    candidate_upright: float = 0.90           # -projected_gravity_z > 0.90
    candidate_consec_steps: int = 10          # 10 steps = 0.2 s to latch
    # --- latch hysteresis (exit) ---
    latch_exit_height: float = 0.55           # exit if h < 0.55 ...
    latch_exit_upright: float = 0.70          # ... or upright < 0.70
    latch_exit_consec_steps: int = 10         # ... for 10 consecutive steps

    # --- 2.2 strict stable-standing (χ_t = stable_now) ---
    # Height is RELATIVE to nominal so the same definition is robust across robots /
    # slightly-imperfect standing poses: h_pelvis > frac * nominal_pelvis_height.
    stable_height_frac: float = 0.90          # h_pelvis > 0.90 * nominal
    nominal_pelvis_height: float = 0.75       # HOME root_link z (measured 0.75 m)
    stable_upright: float = 0.978             # -proj_grav_z > cos(12°) ≈ 0.978
    stable_lin_vel: float = 0.15              # |v_base| < 0.15 m/s (conservative)
    stable_ang_vel: float = 0.35              # |omega_base,xy| < 0.35 rad/s (roll+pitch; NO yaw)
    # RESERVED — deferred out of χ_t (user 2026-08-04). RMS(qdot) is hard to specify
    # correctly and is left for a future revision; capture-point-in-support balance is
    # instead carried by the separate com_over_support reward. Fields kept so the
    # gates can be re-enabled without a schema change.
    stable_joint_vel_rms: float = 0.75        # reserved (RMS(qdot) gate, not in χ_t)
    stable_capture_margin: float = 0.03       # reserved (capture-point gate, not in χ_t)

    # --- 2.3 stable-hold criterion ---
    stable_hold_steps: int = 50               # N_hold = 50 steps = 1.0 s (NOT the old 15)

    # --- 2.4 fall-after-stand ---
    fall_after_stand_height: float = 0.58     # after latch, h < 0.58 ...
    fall_after_stand_consec_steps: int = 20   # ... or lost upright, for 20 consec steps

    # foot-planted force threshold (N) with hysteresis. Lowered 60/30 -> 30/20 for
    # sim2real robustness: a symmetric quiet stand loads ~200 N/foot (m*g/2), so a
    # 20-30 N contact-detection threshold has a ~7x margin yet triggers earlier on
    # noisy real force sensors. (F_z_foot > threshold, the sim2real-friendly criterion.)
    foot_plant_on_n: float = 30.0
    foot_plant_off_n: float = 20.0

    policy_hz: float = 50.0                    # for step<->second conversions

    @property
    def stable_hold_seconds(self) -> float:
        return self.stable_hold_steps / self.policy_hz


# ======================================================================================
# 4. Shared curriculum performance evaluator (iteration-based windows — TIGHTENED)
# ======================================================================================
@dataclass
class PerformanceWindowCfg:
    # A "window" closes after this many POLICY ITERATIONS, provided the episode floor is
    # also met (so low-throughput phases still accumulate statistical mass). This is the
    # core fix vs the spec's raw 256-episode window (which fills in <1 iteration).
    window_iterations: int = 40
    window_min_episodes: int = 512            # floor: don't close a window below this
    promotion_required_windows: int = 3
    rollback_required_windows: int = 2
    cooldown_iterations: int = 120            # no transitions during cooldown

    # --- 4.1 standard promotion condition (TIGHTENED: 0.85 / 0.10) ---
    promote_success_rate: float = 0.85
    promote_fall_after_stand_rate: float = 0.10
    promote_median_stable_duration_s: float = 1.0

    # --- 4.2 standard rollback condition (TIGHTENED: 0.55 x2) ---
    rollback_success_rate: float = 0.55

    # --- 4.3 never advance on insufficient data ---
    min_episodes_per_pose_promotion: int = 512


# ======================================================================================
# 6. Assistance-force curriculum
# ======================================================================================
@dataclass
class AssistanceCurriculumCfg:
    # levels in Newtons; stored/applied internally as weight-ratios (F = r_F * m * g)
    force_levels_n: tuple[float, ...] = (120.0, 100.0, 80.0, 60.0, 40.0, 20.0, 0.0)
    nominal_mass_kg: float = 40.8              # G1 nominal mass; r_F = F / (m*g)
    gravity: float = 9.81
    apply_body: str = "torso_link"             # verify frame == world-vertical at pelvis/torso CoM

    # 6.2 ramp (disabled during settling, linear 0->level over N steps after settle)
    assist_ramp_steps: int = 20
    # 6.3 height gate: 1 below h_lo, smoothstep to 0 at h_hi (must be 0 when stable eval'd)
    gate_height_lo: float = 0.45
    gate_height_hi: float = 0.70

    # 6.6 stricter mastery before beta reduction (UNASSISTED_VALIDATION)
    unassist_min_episodes: int = 512
    unassist_success_rate: float = 0.85
    unassist_fall_after_stand_rate: float = 0.10
    unassist_median_stable_duration_s: float = 1.5


# ======================================================================================
# 7-8. Beta (incremental action rescaler) curriculum + action semantics
# ======================================================================================
@dataclass
class ActionProcessingCfg:
    # 7. incremental semantics: q_d = q_current + beta * s_j * a   (NO q_HOME added)
    # 7.1 per-joint max target increment s_j = v_max * dt_policy (dt=0.02 s)
    #     legs/waist v=2.0 -> 0.04 ; shoulders/elbows v=2.5 -> 0.05 ; wrists v=1.5 -> 0.03
    delta_scale: dict[str, float] = field(default_factory=lambda: {
        ".*_hip_.*": 0.04, ".*_knee_joint": 0.04, ".*_ankle_.*": 0.04,
        "waist_.*": 0.04,
        ".*_shoulder_.*": 0.05, ".*_elbow_joint": 0.05,
        ".*_wrist_.*": 0.03,
    })
    default_delta_scale: float = 0.04          # fallback for any unmatched controlled joint
    # 7.2 disable overlapping mechanisms for the new branch
    ema_alpha: float = 1.0                     # 1.0 == EMA off
    slew_rate_limit: float | None = None       # None == normal slew off
    emergency_rate_limit_rad_s: float = 12.0   # safety clamp only (should never bind)
    clamp_to_soft_limits: bool = True
    # 7.3 initial action ramp (after settling)
    action_ramp_steps: int = 20


@dataclass
class BetaCurriculumCfg:
    beta_levels: tuple[float, ...] = (1.00, 0.85, 0.70, 0.55, 0.40, 0.30, 0.25)
    # extra requirement below this beta: median_stable_duration >= 1.5 s
    tighten_below_beta: float = 0.55
    tighten_median_stable_duration_s: float = 1.5
    # final beta (0.25) mastery gate
    final_min_episodes: int = 512
    final_success_rate: float = 0.85
    final_fall_after_stand_rate: float = 0.10
    final_median_stable_duration_s: float = 2.0
    # emergency recovery: only re-enable assist if rolled all the way to beta=1.0 and
    # success < 0.40 for this many additional windows
    emergency_recovery_success_rate: float = 0.40
    emergency_recovery_windows: int = 3


# ======================================================================================
# 6b. GLOBAL COUPLED assist-force + beta advancement (from-scratch branch)
# ======================================================================================
@dataclass
class CoupledAdvancementCfg:
    """ONE global assist force lambda_F and ONE global action-scale beta, shared by
    every parallel env, advanced TOGETHER on a conservative rolling success-rate
    trigger (user decision 2026-08-04). Replaces the per-env HoST
    AssistanceCurriculum / BetaRescalerCurriculum for the single-pose, no-
    randomization from-scratch run.

    HoST advances the moment ONE episode's head clears H_head — a single lucky
    episode retires the help before the policy is robust. This trigger instead
    requires the WHOLE population to sustain tau_s success over the last K episode
    terminations, where SUCCESS itself is a sustained STABLE HOLD (the product of the
    shared χ_t = is_stable indicator over the final `success_window_steps` of the
    episode), NOT a single-step head touch:

        success_i = ∏_{t=T-N+1}^{T} χ_t   ==   (stable_counter >= N at episode end)
        buffer = deque(maxlen=K)
        on each episode end: buffer.append(success_i)
        if len(buffer) == K and mean(buffer) >= tau_s:
            lambda_F <- max(lambda_F - delta_lambda_F, lambda_F_min)
            beta     <- max(beta     - delta_beta,     beta_min)
            buffer.clear()          # mandatory reset after every advance

    This is STRICTER than HoST (a whole final window must be stable, not one frame),
    so tau_s may need lowering (0.85 -> ~0.75) if early flickering is high. Assist and
    beta are COUPLED (one shared trigger) so difficulty rises along a single axis:
    each advance simultaneously removes support force AND shrinks the per-step action
    authority, exactly when the policy has proven robust at the current level.
    """
    # --- conservative rolling trigger ---
    window_K: int = 128                     # rolling success-buffer length (episodes)
    success_rate_threshold: float = 0.85    # tau_s (lower to ~0.75 if flicker is high)

    # --- success predicate: sustained STABLE HOLD (χ_t product) at episode end ---
    # success_i = (stable_counter >= success_window_steps) at the episode's final
    # step == the shared is_stable indicator held for the whole final window. Started
    # at 50 (1.0 s), matching the reward hold N_hold (fs_success_bonus / fs_progressive
    # _hold) so success and reward align; raise toward 100 (2 s) once flicker is low.
    success_window_steps: int = 50

    # Body the global upward assist wrench is applied to (head/torso proxy).
    head_body: str = "torso_link"

    # --- assist force lambda_F (Newtons, single global scalar) ---
    lambda_F_init: float = 200.0
    lambda_F_min: float = 0.0
    delta_lambda_F: float = 10.0

    # --- action-scale beta (single global scalar) ---
    beta_init: float = 1.0
    beta_min: float = 0.25
    delta_beta: float = 0.02

    # assist is suppressed during the settle window (no launch at t=0); matches the
    # action settle_steps / legacy _ASSIST_UNACTUATED_STEPS convention.
    assist_unactuated_steps: int = 60


# ======================================================================================
# 9-10. Pose randomization + expansion
# ======================================================================================
@dataclass
class PoseRandomizationCfg:
    # keep >=20% of resets at exact canonical SUPINE at every level (anti-forgetting)
    canonical_fraction: float = 0.20
    # per-level (leg, arm, waist, root_z, root_rp, root_yaw) magnitudes; level 0 = none
    levels: tuple[dict[str, float], ...] = field(default_factory=lambda: (
        {"leg": 0.0, "arm": 0.0, "waist": 0.0, "root_z": 0.0, "root_rp": 0.0, "root_yaw": 0.0},
        {"leg": 0.05, "arm": 0.05, "waist": 0.03, "root_z": 0.01, "root_rp": 0.02, "root_yaw": 0.10},
        {"leg": 0.10, "arm": 0.10, "waist": 0.06, "root_z": 0.015, "root_rp": 0.04, "root_yaw": 0.25},
        {"leg": 0.25, "arm": 0.25, "waist": 0.15, "root_z": 0.025, "root_rp": 0.07, "root_yaw": 1.00},
        {"leg": 0.50, "arm": 0.50, "waist": 0.30, "root_z": 0.03, "root_rp": 0.10, "root_yaw": 3.14159},
    ))
    min_episodes_per_level: int = 512


@dataclass
class PoseExpansionCfg:
    # poses introduced one at a time in this order after SUPINE randomization is mastered
    introduction_order: tuple[str, ...] = ("PRONE", "SIDE_LEFT", "SIDE_RIGHT", "SEATED")
    # PRONE
    prone_initial_mix: dict[str, float] = field(default_factory=lambda: {"SUPINE": 0.90, "PRONE": 0.10})
    prone_min_episodes: int = 256
    prone_promote_success: float = 0.60           # -> 80/20 mix
    prone_second_mix: dict[str, float] = field(default_factory=lambda: {"SUPINE": 0.80, "PRONE": 0.20})
    prone_own_curriculum_success: float = 0.75    # + global>=0.80 -> PRONE gets its own randomization
    # SIDE (introduced after SUPINE & PRONE both >= 0.75)
    side_unlock_per_pose_success: float = 0.75
    side_initial_mix: dict[str, float] = field(default_factory=lambda: {
        "SUPINE": 0.60, "PRONE": 0.30, "SIDE_LEFT": 0.05, "SIDE_RIGHT": 0.05})
    side_promote_success: float = 0.60            # -> 10% each
    # SEATED (only after worst active-pose success >= 0.70)
    seated_unlock_worst_success: float = 0.70
    seated_initial_mix: dict[str, float] = field(default_factory=lambda: {
        "SUPINE": 0.45, "PRONE": 0.30, "SIDE_LEFT": 0.10, "SIDE_RIGHT": 0.10, "SEATED": 0.05})
    seated_promote_success: float = 0.60
    # 10.4 per-pose advancement rule (multi-pose)
    multipose_global_success: float = 0.80
    multipose_worst_pose_success: float = 0.70
    multipose_fall_after_stand: float = 0.15
    min_episodes_per_pose: int = 256


# ======================================================================================
# 11-12. Phase-dependent multi-critic weights + task-reward gating
# ======================================================================================
@dataclass
class MultiCriticWeightsCfg:
    # groups: (task, style, regularization, post_task)
    pre_standing: tuple[float, float, float, float] = (2.5, 2.0, 0.1, 0.0)
    post_standing: tuple[float, float, float, float] = (0.3, 0.8, 0.5, 2.5)


@dataclass
class RewardGatingCfg:
    # recovery rewards forced to zero AFTER standing_latched (one shared mask)
    recovery_terms: tuple[str, ...] = (
        "height_progress", "supine_rising_prep", "prone_recovery", "pelvis_height_bridge",
    )


# ======================================================================================
# top-level bundle
# ======================================================================================
@dataclass
class FromScratchCurriculumCfg:
    stability: StabilityCfg = field(default_factory=StabilityCfg)
    performance: PerformanceWindowCfg = field(default_factory=PerformanceWindowCfg)
    assistance: AssistanceCurriculumCfg = field(default_factory=AssistanceCurriculumCfg)
    coupled: CoupledAdvancementCfg = field(default_factory=CoupledAdvancementCfg)
    action: ActionProcessingCfg = field(default_factory=ActionProcessingCfg)
    beta: BetaCurriculumCfg = field(default_factory=BetaCurriculumCfg)
    pose_randomization: PoseRandomizationCfg = field(default_factory=PoseRandomizationCfg)
    pose_expansion: PoseExpansionCfg = field(default_factory=PoseExpansionCfg)
    multi_critic: MultiCriticWeightsCfg = field(default_factory=MultiCriticWeightsCfg)
    reward_gating: RewardGatingCfg = field(default_factory=RewardGatingCfg)
