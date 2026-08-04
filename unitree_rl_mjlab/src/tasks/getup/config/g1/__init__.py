from mjlab.tasks.registry import register_mjlab_task

from src.tasks.getup.rl import GetUpOnPolicyRunner
from .env_cfgs import unitree_g1_getup_env_cfg
from .env_cfgs_fromscratch import unitree_g1_getup_fromscratch_env_cfg
from .rl_cfg import unitree_g1_getup_ppo_runner_cfg

register_mjlab_task(
    task_id="Unitree-G1-GetUp",
    env_cfg=unitree_g1_getup_env_cfg(),
    play_env_cfg=unitree_g1_getup_env_cfg(play=True),
    rl_cfg=unitree_g1_getup_ppo_runner_cfg(),
    runner_cls=GetUpOnPolicyRunner,
)

# From-scratch staged-curriculum V2 (M1 launchable baseline). Parallel task — the legacy
# Unitree-G1-GetUp above is unchanged.
register_mjlab_task(
    task_id="Unitree-G1-GetUp-Scratch",
    env_cfg=unitree_g1_getup_fromscratch_env_cfg(),
    play_env_cfg=unitree_g1_getup_fromscratch_env_cfg(play=True),
    rl_cfg=unitree_g1_getup_ppo_runner_cfg(),
    runner_cls=GetUpOnPolicyRunner,
)
