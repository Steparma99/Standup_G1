"""Action-semantics + action-bound test (section 17) for IncrementalJointPositionAction.

Verifies q_d = q_current + beta*s_j*a (HOME NOT added) and |increment| <= beta*s_j.
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("GETUP_TRAIN_INITIAL_BETA", "1.0")
import torch
import src.tasks.getup
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

cfg = load_env_cfg("Unitree-G1-GetUp-Scratch"); cfg.scene.num_envs = 8
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
env.reset()
at = env.action_manager._terms["joint_pos"]
tids = at._target_ids
robot = env.scene["robot"]

# step past settle (60) + action ramp (20) so ramp == 1.0
zero = torch.zeros(env.num_envs, env.action_space.shape[1])
for _ in range(90):
    env.step(zero)

# now drive a known action directly through the action term
a_val = 0.5
full = torch.full((env.num_envs, env.action_space.shape[1]), a_val)
q_before = robot.data.joint_pos[:, tids].clone()
at.process_actions(full)
increment = at.scaled_increment                       # beta*s_j*a (pre-clamp)
applied = at.applied_target                           # final clamped target
beta = 1.0
s_j = at._delta_scale                                  # [1, A]

ok = True

# 1. increment == beta * s_j * a  (exactly, pre-clamp)
expected_inc = beta * s_j * a_val
err_inc = (increment - expected_inc).abs().max().item()
print(f"increment vs beta*s_j*a: max_err={err_inc:.2e} (expect ~0)")
ok &= err_inc < 1e-6

# 2. HOME not added: applied - q_before should equal increment where unclamped.
#    (clamp only reduces magnitude, so applied-delta <= increment everywhere)
delta_applied = applied - q_before
# where not at a joint limit, delta_applied == increment
unclamped = (delta_applied - increment).abs() < 1e-6
frac_matching = unclamped.float().mean().item()
print(f"applied-delta == increment on {frac_matching*100:.0f}% of joints (rest clamped by limits)")
# HOME test: if HOME were added, delta would be ~ (HOME - q_before) + increment, huge.
from src.assets.robots.unitree_g1.g1_constants import HOME_KEYFRAME
from mjlab.utils.string import resolve_expr
home = torch.tensor([resolve_expr(dict(HOME_KEYFRAME.joint_pos), robot.joint_names, 0.0)])[:, tids]
home_delta = (home - q_before).abs().mean().item()
print(f"mean|HOME - q_current| = {home_delta:.3f} rad; mean|applied-delta| = {delta_applied.abs().mean().item():.4f} rad")
ok &= (delta_applied.abs().mean().item() < 0.5 * home_delta)  # applied delta is tiny, NOT a jump to HOME

# 3. action bound: |increment| <= beta*s_j exactly at |a|=1
full1 = torch.ones(env.num_envs, env.action_space.shape[1])
at.process_actions(full1)
bound_ok = (at.scaled_increment.abs() <= (beta * s_j + 1e-6)).all().item()
print(f"|increment| <= beta*s_j at |a|=1: {bound_ok}")
ok &= bool(bound_ok)

print("\nINCREMENTAL ACTION TEST:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
