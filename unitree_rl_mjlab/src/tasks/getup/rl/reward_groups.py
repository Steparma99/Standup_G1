"""Reward-group definitions for the HoST-style multi-critic architecture.

Each reward term is assigned to exactly one of four groups. Each group gets its own
critic; the per-group advantages are combined with ``GROUP_WEIGHTS`` when forming the
PPO surrogate loss (see ``multi_critic_ppo.py``).

NOTE: This term -> group mapping is a *provisional draft*. The user will review and
finalize it. The canonical group order is fixed (``GROUP_ORDER``); only the per-term
assignment and weights are expected to change.
"""

from __future__ import annotations

import torch

# Canonical group order -> critic index 0..3. Do not reorder without also updating
# GROUP_WEIGHTS and any logging that assumes this order.
GROUP_ORDER: tuple[str, ...] = ("task", "style", "regularization", "post_task")

# Default per-group advantage weights (HoST: task 2.5 / style 1.9 / regu 0.1 / post 1.0).
# Style raised 1.9 -> 2.2: posture shaping is important for a correct standup, and the
# style terms are height-gated (only fire once the robot rises), so a higher critic
# weight amplifies them where they matter without touching floor-phase task learning.
# v29: post_task 1.0 -> 1.5. post_task now carries the most individual terms (all the
# terminal arm/leg pose shaping) and the largest total weight budget, yet had the
# LOWEST critic weight — the newest terminal terms (arm_hand_box, leg_width_ordering,
# post_terminal_core) were still climbing, un-plateaued, at the end of the v28 run.
# Raising the group's critic weight amplifies exactly those terminal-pose gradients.
# Overridable from the algorithm config (``reward_group_weights``).
GROUP_WEIGHTS: tuple[float, ...] = (2.5, 2.2, 0.1, 1.5)

# Provisional assignment of every reward term to a group. Must cover ALL active reward
# terms exactly once (enforced by build_group_onehot). Terminal penalties
# (is_terminated, joint_pos_limits) are folded into "task" by default.
REWARD_GROUP_MAP: dict[str, str] = {
    # --- TASK (HoST definitive): high-level objectives, weight 1 each --------------
    "task_head_height": "task",
    "task_base_orientation": "task",
    "task_head_contact": "task",
    # Stage-0 bootstrap: directional progress signals for floor-level learning.
    "height_progress": "task",
    "prone_recovery": "task",
    "supine_rising_prep": "task",
    # Crouch->stand bridge: dense feet-planted pelvis-height climbing signal.
    "pelvis_height_bridge": "task",
    # --- STYLE (HoST definitive): motion shaping ----------------------------------
    # (v9 ablation removed the binary deviation penalties: waist_yaw/hip/knee/
    # shoulder_roll deviation + the weight-0 feet_stumble scaffold. Extra map entries
    # are harmless, but keep this in sync with getup_env_cfg.py's rewards dict.)
    "style_waist_upright": "style",
    "style_foot_displacement": "style",
    "style_foot_distance": "style",
    "style_shank_orientation": "style",
    "style_base_ang_vel": "style",
    "style_ankle_parallel": "style",
    "style_hand_left_contact": "style",
    "style_hand_right_contact": "style",
    # --- REGULARIZATION (HoST definitive): weak shaping penalties ------------------
    "joint_acc_l2": "regularization",
    "action_rate_l2": "regularization",
    "action_acc_l2": "regularization",  # smoothness (2nd action difference)
    "joint_torques_l2": "regularization",
    "joint_power_l2": "regularization",
    "joint_vel_l2": "regularization",
    "reg_arm_vel": "regularization",
    "joint_tracking_error": "regularization",
    "joint_pos_limits": "regularization",
    "joint_vel_limits": "regularization",
    # --- POST-TASK (HoST definitive): hold the standing state, gated h>H_STAGE2 -----
    "post_base_ang_vel": "post_task",
    "post_base_lin_vel": "post_task",
    "anti_jump_velocity": "post_task",
    "post_base_orientation": "post_task",
    "post_base_height": "post_task",
    "post_upper_body_posture": "post_task",
    "post_standing_posture": "post_task",
    "post_home_pose_l2": "post_task",
    "post_bilateral_symmetry": "post_task",
    "post_stand_on_feet": "post_task",
    "post_feet_parallel": "post_task",
    "post_feet_yaw": "post_task",
    "com_over_support": "post_task",
    "stable_success_hold": "post_task",
    # v23 functional final-pose terms (exact-HOME imitation terms above kept at
    # weight 0 in env_cfgs.py; entries stay so the map still covers them).
    "arms_in_front": "post_task",
    "post_joint_stillness": "post_task",
    "standing_alive_bonus": "post_task",
    # v26 task-space terminal "quiet standing" region: geometric-mean core +
    # crossed-legs (signed lateral foot ordering) + worst-leg shank verticality.
    "post_terminal_core": "post_task",
    "leg_width_ordering": "post_task",
    "shank_worst_leg": "post_task",
    # v27 task-space ARM/HAND terminal-pose terms (arms_in_front decomposed).
    "arm_hand_box": "post_task",
    "arm_elbow_flexion": "post_task",
    "arm_overextension": "post_task",
    "arm_clearance": "post_task",
    # v31 banded joint-space final-pose groups (legs / waist / upper). upper folds in
    # the hand-workspace box (sqrt(r_arms·r_hand)). Replace the disabled joint-space
    # imitation terms + the weak task-space arm bands.
    "pose_legs": "post_task",
    "pose_waist": "post_task",
    "pose_upper": "post_task",
}


def build_group_onehot(
    active_terms: list[str],
    group_map: dict[str, str] | None = None,
    group_order: tuple[str, ...] = GROUP_ORDER,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build the [num_terms, num_groups] aggregation matrix.

    ``grouped_reward = step_reward @ onehot`` maps per-term rewards (ordered like
    ``RewardManager.active_terms``) to per-group sums.

    Raises:
        ValueError: if any active term is missing from the group map, or maps to an
            unknown group. This is the reward-coverage guard from the plan.
    """
    group_map = group_map if group_map is not None else REWARD_GROUP_MAP
    group_to_idx = {g: i for i, g in enumerate(group_order)}

    missing = [t for t in active_terms if t not in group_map]
    if missing:
        raise ValueError(
            f"Reward terms missing from REWARD_GROUP_MAP: {missing}. "
            f"Every active reward term must be assigned to a group."
        )

    num_terms = len(active_terms)
    num_groups = len(group_order)
    onehot = torch.zeros(num_terms, num_groups, device=device)
    for term_idx, term in enumerate(active_terms):
        group = group_map[term]
        if group not in group_to_idx:
            raise ValueError(
                f"Reward term '{term}' maps to unknown group '{group}'. "
                f"Valid groups: {group_order}."
            )
        onehot[term_idx, group_to_idx[group]] = 1.0
    return onehot
