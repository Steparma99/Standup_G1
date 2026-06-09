"""Pose design lab: place a candidate ground pose on the floor (real-collision
bisection, no dynamics), report which BODIES touch the floor + penetration, and
render multi-view PNGs for visual inspection. Used to design SIDE / SEATED
keyframes that rest on the intended body parts.

Edit CANDIDATES below and run:
  MUJOCO_GL=osmesa conda run -n unitree_rl_cpu python scripts/pose_lab.py
PNGs go to keyframe_check/<name>_<view>.png
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
XML_PATH = HERE.parent / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"
OUT_DIR = HERE.parent / "keyframe_check"

CLEARANCE = 0.02  # uniform small spawn gap (settles in first steps, no penetration)

# ----------------------------------------------------------------------------
# Candidate poses. axis/angle = base orientation; joints = regex->rad (last wins)
# ----------------------------------------------------------------------------
CANDIDATES = {
    # SIDE_LEFT: lie on LEFT side. Bottom (left) arm tucked forward beside torso;
    # top (RIGHT) arm reaches down-forward to plant the palm flat on the floor.
    "SIDE_LEFT": {
        "axis": (1.0, 0.0, 0.0), "angle_deg": -90.0,
        "joints": {
            r".*_ankle_pitch_joint": -0.2,
            r".*_hip_pitch_joint": -0.3,
            r".*_knee_joint": 0.4,
            # bottom (left) arm: extend forward along the floor in line with body
            r"left_shoulder_pitch_joint": 1.3,
            r"left_shoulder_roll_joint": 0.0,
            r"left_elbow_joint": 0.3,
            # top (right) arm: roll swings it down toward floor (body+Y), pitch +
            # elbow reach the palm forward to plant in front of the chest
            r"right_shoulder_pitch_joint": 0.4,
            r"right_shoulder_roll_joint": 1.0,
            r"right_elbow_joint": 0.5,
        },
    },
    # SEATED: butt on floor, knees bent up, feet flat on floor in front (the
    # natural pre-stand sitting pose). Slight back lean so the pelvis rolls down.
    "SEATED": {
        "axis": (0.0, 1.0, 0.0), "angle_deg": -14.0,
        "joints": {
            r".*_hip_pitch_joint": -2.24,   # deep hip flexion: thighs up toward chest
            r".*_knee_joint": 1.87,         # knees bent so feet plant level with the butt
            r".*_ankle_pitch_joint": -0.45,  # feet flat on the floor
            r".*_shoulder_pitch_joint": 0.3,
            r"left_shoulder_roll_joint": 0.25,
            r"right_shoulder_roll_joint": -0.25,
            r".*_elbow_joint": 0.5,
        },
    },
}

VIEWS = [(90.0, -10.0, "side"), (0.0, -10.0, "front"), (50.0, -35.0, "diag")]


def joint_qposadr(model):
    out = {}
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            out[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)] = int(model.jnt_qposadr[j])
    return out


def free_adr(model):
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[j])
    return None


def axis_angle_to_quat(axis, angle_deg):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array(axis, float), math.radians(angle_deg))
    return q


def floor_contacts(model, data):
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    out = []
    for c in range(data.ncon):
        con = data.contact[c]
        b1, b2 = int(model.geom_bodyid[con.geom1]), int(model.geom_bodyid[con.geom2])
        if (b1 == 0) != (b2 == 0):
            rb = b2 if b1 == 0 else b1
            out.append((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, rb), float(con.dist)))
    return out


def min_floor_dist(model, data):
    mujoco.mj_forward(model, data)
    ds = [d for _, d in floor_contacts(model, data)]
    return min(ds) if ds else math.inf, len(ds)


def expand_joints(model, jadr, jdict):
    targets = {}
    for jname in jadr:
        for pat, val in jdict.items():
            if re.fullmatch(pat, jname):
                targets[jname] = float(val)
    return targets


def place(model, data, jadr, fadr, quat, targets):
    mujoco.mj_resetData(model, data)
    data.qpos[fadr : fadr + 3] = (0.0, 0.0, 1.0)
    data.qpos[fadr + 3 : fadr + 7] = quat
    for jn, v in targets.items():
        data.qpos[jadr[jn]] = v
    # bisection on height
    hi, lo = 2.0, -0.5
    for _ in range(45):
        mid = 0.5 * (hi + lo)
        data.qpos[fadr + 2] = mid
        md, n = min_floor_dist(model, data)
        if (n > 0) and (md < 0.0):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    data.qpos[fadr + 2] = hi + CLEARANCE
    mujoco.mj_forward(model, data)
    return hi + CLEARANCE


def main():
    OUT_DIR.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    jadr, fadr = joint_qposadr(model), free_adr(model)
    w = min(900, int(model.vis.global_.offwidth))
    h = min(700, int(model.vis.global_.offheight))
    renderer = mujoco.Renderer(model, height=h, width=w)
    cam = mujoco.MjvCamera()

    for name, spec in CANDIDATES.items():
        quat = axis_angle_to_quat(spec["axis"], spec["angle_deg"])
        targets = expand_joints(model, jadr, spec["joints"])
        z = place(model, data, jadr, fadr, quat, targets)
        # drop just below rest to read the contact set
        data.qpos[fadr + 2] = z - CLEARANCE - 2e-3
        mujoco.mj_forward(model, data)
        cs = floor_contacts(model, data)
        bodies = {}
        for b, d in cs:
            bodies[b] = min(bodies.get(b, 0.0), d)
        rotn = quat / np.linalg.norm(quat)
        print(f"\n=== {name}  rot=({rotn[0]:+.4f},{rotn[1]:+.4f},{rotn[2]:+.4f},{rotn[3]:+.4f})  spawn_z={z:.3f} ===")
        print("  contacts at rest:")
        for b in sorted(bodies):
            print(f"    {b:28s} {bodies[b]*1000:+7.2f} mm")
        # render at spawn height
        data.qpos[fadr + 2] = z
        mujoco.mj_forward(model, data)
        for az, el, label in VIEWS:
            cam.lookat[:] = (0.0, 0.0, max(z, 0.25))
            cam.distance, cam.azimuth, cam.elevation = 2.6, az, el
            renderer.update_scene(data, camera=cam)
            Image.fromarray(renderer.render()).save(OUT_DIR / f"lab_{name}_{label}.png")
    renderer.close()
    print(f"\nPNGs in {OUT_DIR}")


if __name__ == "__main__":
    main()
