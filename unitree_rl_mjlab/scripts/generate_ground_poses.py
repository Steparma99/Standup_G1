"""Generate physically-valid ground keyframes (SUPINE / PRONE) for the G1.

For each pose we:
  1. build the base orientation as a clean axis-angle rotation (quaternion),
  2. set the joints to a neutral posture,
  3. auto-compute the base height so the lowest collision geom just rests on the
     floor (no initial penetration -> the robot does not "explode" at reset),
  4. render the result and print the exact (pos, rot) to paste into
     src/assets/robots/unitree_g1/g1_constants.py.

Axis convention for the G1 base: x = forward, y = left, z = up.
  SUPINE (on the back, face up)  = rotate -90 deg about +Y
  PRONE  (face down)             = rotate +90 deg about +Y

Usage:
    conda run -n unitree_rl_cpu python scripts/generate_ground_poses.py
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
XML_PATH = HERE.parent / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"
OUT_DIR = HERE.parent / "keyframe_check"

# Each pose: rotation axis, angle (deg), and joint targets (regex -> radians).
# Joints left at 0.0 = neutral (legs/arms extended along the body).
POSES = {
    "SUPINE": {"axis": (0.0, 1.0, 0.0), "angle_deg": -90.0, "joint_pos": {r".*": 0.0}},
    "PRONE": {"axis": (0.0, 1.0, 0.0), "angle_deg": 90.0, "joint_pos": {r".*": 0.0}},
}

VIEWS = [(90.0, -15.0, "side"), (50.0, -45.0, "diag")]
CLEARANCE = 0.05  # start slightly above the floor, then let it settle
SETTLE_STEPS = 1500  # physics steps to let the robot come to rest on the ground


def joint_name_to_qposadr(model):
    out = {}
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            out[name] = int(model.jnt_qposadr[j])
    return out


def free_joint_qposadr(model):
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[j])
    return None


def axis_angle_to_quat(axis, angle_deg):
    quat = np.zeros(4)
    ax = np.array(axis, dtype=float)
    mujoco.mju_axisAngle2Quat(quat, ax, math.radians(angle_deg))
    return quat  # (w, x, y, z)


def lowest_geom_z(model, data):
    """World z of the lowest point of any robot collision geom (bounding-sphere)."""
    zmin = math.inf
    for g in range(model.ngeom):
        # Skip world/static geoms (floor) — bodyid 0 is the worldbody.
        if int(model.geom_bodyid[g]) == 0:
            continue
        # geom_rbound is the bounding-sphere radius; subtracting it is conservative.
        z = float(data.geom_xpos[g, 2]) - float(model.geom_rbound[g])
        zmin = min(zmin, z)
    return zmin


def main():
    OUT_DIR.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    joint_adr = joint_name_to_qposadr(model)
    free_adr = free_joint_qposadr(model)

    w = min(1024, int(model.vis.global_.offwidth))
    h = min(768, int(model.vis.global_.offheight))
    renderer = mujoco.Renderer(model, height=h, width=w)
    cam = mujoco.MjvCamera()

    print(f"[INFO] {len(joint_adr)} actuated DOF\n")

    for name, p in POSES.items():
        mujoco.mj_resetData(model, data)
        quat = axis_angle_to_quat(p["axis"], p["angle_deg"])

        # Set orientation first, joints, then forward-kinematics at a safe height.
        data.qpos[free_adr : free_adr + 3] = (0.0, 0.0, 1.0)  # temp high
        data.qpos[free_adr + 3 : free_adr + 7] = quat
        import re

        for jname, adr in joint_adr.items():
            for pattern, value in p["joint_pos"].items():
                if re.fullmatch(pattern, jname):
                    data.qpos[adr] = value
        mujoco.mj_forward(model, data)

        # Drop the base so the lowest geom sits CLEARANCE above the floor.
        z_low = lowest_geom_z(model, data)
        data.qpos[free_adr + 2] = 1.0 - z_low + CLEARANCE
        mujoco.mj_forward(model, data)

        # Let the robot fall and settle on the ground (joints held at neutral
        # via ctrl=0). The resting qpos is a physically-valid, stable keyframe.
        data.ctrl[:] = 0.0
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data)

        base_pos = data.qpos[free_adr : free_adr + 3].copy()
        base_quat = data.qpos[free_adr + 3 : free_adr + 7].copy()
        # Re-normalize quaternion (integration drift).
        base_quat /= np.linalg.norm(base_quat)
        lin_vel = np.linalg.norm(data.qvel[free_adr : free_adr + 3])

        # Numeric check: projected gravity z in base frame (0 = lying flat).
        rotmat = np.zeros(9)
        mujoco.mju_quat2Mat(rotmat, base_quat)
        grav_b = rotmat.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        print(f"=== {name} (after settling) ===")
        print(f"  rot (w,x,y,z) = ({base_quat[0]:+.4f}, {base_quat[1]:+.4f}, {base_quat[2]:+.4f}, {base_quat[3]:+.4f})")
        print(f"  pos (x,y,z)   = ({base_pos[0]:+.4f}, {base_pos[1]:+.4f}, {base_pos[2]:+.4f}) m")
        print(f"  projected_gravity_b_z = {grav_b[2]:+.3f}   (0 = flat on ground)")
        print(f"  residual base speed   = {lin_vel:.4f} m/s   (~0 = stable, holds position)")
        base_z = float(base_pos[2])

        lookat = np.array([0.0, 0.0, max(base_z, 0.2)])
        for az, el, label in VIEWS:
            cam.lookat[:] = lookat
            cam.distance = 2.8
            cam.azimuth = az
            cam.elevation = el
            renderer.update_scene(data, camera=cam)
            Image.fromarray(renderer.render()).save(OUT_DIR / f"{name}_settled_{label}.png")
        print(f"  saved {name}_settled_side.png / {name}_settled_diag.png\n")

    renderer.close()


if __name__ == "__main__":
    main()
