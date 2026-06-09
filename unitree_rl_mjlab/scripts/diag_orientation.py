"""Probe: orientation + key reward signals for each reference pose.

Sets the robot to each keyframe (SUPINE/PRONE/SIDE_*/SEATED/HOME) and reports the
raw orientation signals and the current value of the orientation-related rewards,
so the body_up_exp / prone_supine_righting redesign is grounded in real numbers.

Run:  CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl python scripts/diag_orientation.py
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
from prettytable import PrettyTable

from mjlab.envs import ManagerBasedRlEnv

import src.tasks.getup.mdp as mdp
from src.tasks.getup.config.g1.env_cfgs import unitree_g1_getup_env_cfg
from src.assets.robots.unitree_g1.g1_constants import (
    HOME_KEYFRAME,
    PRONE_KEYFRAME,
    SEATED_KEYFRAME,
    SIDE_LEFT_KEYFRAME,
    SIDE_RIGHT_KEYFRAME,
    SUPINE_KEYFRAME,
)
from mjlab.envs.mdp.actions.actions import JointPositionAction  # noqa: F401
from mjlab.utils.string import resolve_expr  # joint regex -> vector

POSES = {
    "HOME(stand)": HOME_KEYFRAME,
    "SUPINE": SUPINE_KEYFRAME,
    "PRONE": PRONE_KEYFRAME,
    "SIDE_LEFT": SIDE_LEFT_KEYFRAME,
    "SIDE_RIGHT": SIDE_RIGHT_KEYFRAME,
    "SEATED": SEATED_KEYFRAME,
}


def main() -> None:
    cfg = unitree_g1_getup_env_cfg()
    cfg.scene.num_envs = len(POSES)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
    asset = env.scene["robot"]
    names = asset.joint_names

    # Write one pose per env, then forward the sim to populate derived state.
    env.reset()
    for i, kf in enumerate(POSES.values()):
        pos = tuple(getattr(kf, "pos", None) or (0.0, 0.0, 0.0))
        rot = tuple(getattr(kf, "rot", None) or (1.0, 0.0, 0.0, 0.0))
        # Lift HOME a touch so it is clearly "standing", others use their rest z.
        root = torch.tensor([[*pos, *rot]], dtype=torch.float32)
        jvec = torch.tensor(
            [resolve_expr(getattr(kf, "joint_pos", {}) or {}, names, 0.0)],
            dtype=torch.float32,
        )
        ids = torch.tensor([i])
        asset.write_root_link_pose_to_sim(root, env_ids=ids)
        asset.write_joint_state_to_sim(jvec, torch.zeros_like(jvec), env_ids=ids)
    env.sim.forward()

    pg = asset.data.projected_gravity_b  # [B,3]
    tilt_xy = torch.norm(pg[:, :2], dim=1)
    z = asset.data.root_link_pos_w[:, 2]

    # NEW body_up_exp (true uprightness) evaluated on the injected state.
    body_up = mdp.body_up_exp(env, k=4.0, asset_cfg=mdp.SceneEntityCfg("robot"))
    # Prone/supine hemisphere gates used by the new split rewards.
    prone_gate = torch.clamp(pg[:, 0], 0.0, 1.0)    # 1 prone, 0 side/supine
    supine_gate = torch.clamp(-pg[:, 0], 0.0, 1.0)  # 1 supine, 0 side/prone
    sit_up = torch.clamp(-pg[:, 2], 0.0, 1.0)       # 0 flat -> 1 upright

    table = PrettyTable()
    table.field_names = [
        "pose", "z(m)", "pg_x", "pg_z", "tilt_xy",
        "body_up(NEW)", "prone_gate", "supine_gate", "sit_up",
    ]
    table.align = "r"
    table.align["pose"] = "l"
    for i, name in enumerate(POSES):
        table.add_row([
            name, f"{z[i]:.3f}",
            f"{pg[i,0]:.2f}", f"{pg[i,2]:.2f}", f"{tilt_xy[i]:.2f}",
            f"{body_up[i]:.3f}", f"{prone_gate[i]:.2f}", f"{supine_gate[i]:.2f}",
            f"{sit_up[i]:.2f}",
        ])
    print(table)
    print(
        "\nKey: pg_* = projected_gravity_b. Upright => pg_z=-1, tilt_xy=0.\n"
        "body_up(NEW) = exp(-4*(1+pg_z)): ~0 from any flat pose, sharp toward upright.\n"
        "prone/supine_gate from pg_x separate the two hemispheres for the split rewards."
    )


if __name__ == "__main__":
    main()
