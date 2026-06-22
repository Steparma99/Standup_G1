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
# Overridable from the algorithm config (``reward_group_weights``).
GROUP_WEIGHTS: tuple[float, ...] = (2.5, 1.9, 0.1, 1.0)

# Provisional assignment of every reward term to a group. Must cover ALL active reward
# terms exactly once (enforced by build_group_onehot). Terminal penalties
# (is_terminated, joint_pos_limits) are folded into "task" by default.
REWARD_GROUP_MAP: dict[str, str] = {
    # --- TASK (HoST definitive): high-level objectives, weight 1 each --------------
    "task_head_height": "task",
    "task_base_orientation": "task",
    # Stage-0 bootstrap: directional progress signals for floor-level learning.
    "height_progress": "task",
    "prone_recovery": "task",
    "supine_rising_prep": "task",
    # --- STYLE (HoST definitive): motion shaping ----------------------------------
    "style_waist_yaw_deviation": "style",
    "style_waist_upright": "style",
    "style_hip_deviation": "style",
    "style_knee_deviation": "style",
    "style_shoulder_roll_deviation": "style",
    "style_foot_displacement": "style",
    "style_foot_distance": "style",
    "style_shank_orientation": "style",
    "style_base_ang_vel": "style",
    "style_ankle_parallel": "style",
    "style_feet_stumble": "style",
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
    "post_base_orientation": "post_task",
    "post_base_height": "post_task",
    "post_upper_body_posture": "post_task",
    "post_standing_posture": "post_task",
    "post_stand_on_feet": "post_task",
    "post_feet_parallel": "post_task",
    "post_feet_yaw": "post_task",
    "stable_success_hold": "post_task",
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
