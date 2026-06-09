"""Compute final, penetration-free ground keyframes for all 5 poses with a
uniform spawn clearance, and print ready-to-paste constant blocks.

For each pose: set base orientation + joints, bisection on base height with real
collision to find the just-touching height, add CLEARANCE, print pos/rot/joints.

  MUJOCO_GL=osmesa conda run -n unitree_rl_cpu python scripts/final_keyframes.py
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
XML = HERE.parent / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"
CLEARANCE = 0.02

# SUPINE/PRONE keep their existing (working) joint sets; only height is recomputed
# for a uniform clearance. SIDE/SEATED use the new pose-lab designs. SIDE_RIGHT is
# the left-right mirror of SIDE_LEFT (swap L<->R, negate roll/yaw).
_SUP_PRO_JOINTS = {
    r".*_elbow_joint": 1.0,
    r"left_shoulder_roll_joint": 0.2,
    r"left_shoulder_pitch_joint": 0.2,
    r"right_shoulder_roll_joint": -0.2,
    r"right_shoulder_pitch_joint": 0.2,
    r".*_ankle_pitch_joint": -0.4,
}
_SIDE_LEFT_JOINTS = {
    r".*_ankle_pitch_joint": -0.2,
    r".*_hip_pitch_joint": -0.3,
    r".*_knee_joint": 0.4,
    r"left_shoulder_pitch_joint": 1.3,
    r"left_shoulder_roll_joint": 0.0,
    r"left_elbow_joint": 0.3,
    r"right_shoulder_pitch_joint": 0.4,
    r"right_shoulder_roll_joint": 1.0,
    r"right_elbow_joint": 0.5,
}
_SIDE_RIGHT_JOINTS = {
    r".*_ankle_pitch_joint": -0.2,
    r".*_hip_pitch_joint": -0.3,
    r".*_knee_joint": 0.4,
    r"right_shoulder_pitch_joint": 1.3,
    r"right_shoulder_roll_joint": 0.0,
    r"right_elbow_joint": 0.3,
    r"left_shoulder_pitch_joint": 0.4,
    r"left_shoulder_roll_joint": -1.0,
    r"left_elbow_joint": 0.5,
}
_SEATED_JOINTS = {
    r".*_hip_pitch_joint": -2.24,
    r".*_knee_joint": 1.87,
    r".*_ankle_pitch_joint": -0.45,
    r".*_shoulder_pitch_joint": 0.3,
    r"left_shoulder_roll_joint": 0.25,
    r"right_shoulder_roll_joint": -0.25,
    r".*_elbow_joint": 0.5,
}

POSES = {
    "SUPINE":     {"axis": (0, 1, 0), "deg": -90.0, "joints": _SUP_PRO_JOINTS},
    "PRONE":      {"axis": (0, 1, 0), "deg":  90.0, "joints": _SUP_PRO_JOINTS},
    "SIDE_LEFT":  {"axis": (1, 0, 0), "deg": -90.0, "joints": _SIDE_LEFT_JOINTS},
    "SIDE_RIGHT": {"axis": (1, 0, 0), "deg":  90.0, "joints": _SIDE_RIGHT_JOINTS},
    "SEATED":     {"axis": (0, 1, 0), "deg": -14.0, "joints": _SEATED_JOINTS},
}


def main():
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    jadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): int(m.jnt_qposadr[j])
            for j in range(m.njnt) if m.jnt_type[j] in (2, 3)}
    fadr = next(int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == 0)

    def setpose(joints, ax, deg, base):
        mujoco.mj_resetData(m, d)
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array(ax, float), math.radians(deg))
        d.qpos[fadr:fadr + 3] = (0, 0, base)
        d.qpos[fadr + 3:fadr + 7] = q
        for jn, a in jadr.items():
            for p, v in joints.items():
                if re.fullmatch(p, jn):
                    d.qpos[a] = v
        mujoco.mj_forward(m, d)
        return q

    def min_floor(base, joints, ax, deg):
        setpose(joints, ax, deg, base)
        md = math.inf
        for c in range(d.ncon):
            con = d.contact[c]
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            if (b1 == 0) != (b2 == 0):
                md = min(md, float(con.dist))
        return md

    for name, p in POSES.items():
        hi, lo = 2.5, -0.5
        for _ in range(50):
            mid = 0.5 * (hi + lo)
            md = min_floor(mid, p["joints"], p["axis"], p["deg"])
            if md < 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-4:
                break
        z = hi + CLEARANCE
        q = setpose(p["joints"], p["axis"], p["deg"], z)
        q = q / np.linalg.norm(q)
        # expand joints for printing
        targets = {}
        for jn in jadr:
            for pat, v in p["joints"].items():
                if re.fullmatch(pat, jn):
                    targets[jn] = float(v)
        print(f"\n# {name}: rest+{CLEARANCE*100:.0f}cm clearance, no penetration")
        print(f"{name}_KEYFRAME = EntityCfg.InitialStateCfg(")
        print(f"    pos=({0.0:.3f}, {0.0:.3f}, {z:.3f}),")
        print(f"    rot=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}),")
        print("    joint_pos={")
        # print using the regex dict (compact) instead of every joint
        for pat, v in p["joints"].items():
            print(f'        r"{pat}": {v},')
        print("    },")
        print("    joint_vel={\".*\": 0.0},")
        print(")")


if __name__ == "__main__":
    main()
