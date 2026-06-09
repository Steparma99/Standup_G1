"""Raw-MuJoCo physical contact map per pose, independent of sensors/actions.

For each keyframe it lowers the base by bisection until the robot just rests on
the floor (0 clearance), runs mj_forward, and lists every robot geom in contact
with the floor plus the penetration depth. This answers, physically: in each
pose, WHICH body parts touch the ground (e.g. do the hands touch in the side
poses? does supine rest on the back/torso?).

Run:
  MUJOCO_GL=osmesa conda run -n unitree_rl_cpu python scripts/diag_contact_geoms.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.assets.robots.unitree_g1.g1_constants import (  # noqa: E402
    PRONE_KEYFRAME,
    SEATED_KEYFRAME,
    SIDE_LEFT_KEYFRAME,
    SIDE_RIGHT_KEYFRAME,
    SUPINE_KEYFRAME,
)

XML_PATH = HERE.parent / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"

POSES = {
    "SUPINE": SUPINE_KEYFRAME,
    "PRONE": PRONE_KEYFRAME,
    "SIDE_LEFT": SIDE_LEFT_KEYFRAME,
    "SIDE_RIGHT": SIDE_RIGHT_KEYFRAME,
    "SEATED": SEATED_KEYFRAME,
}


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


def geom_name(model, gid):
    gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    bid = int(model.geom_bodyid[gid])
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"
    return gname or f"[{bname}]"


def floor_contacts(model, data):
    """Return list of (robot_geom_name, dist) for robot-vs-floor contacts."""
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    out = []
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = con.geom1, con.geom2
        if g1 == floor_gid:
            out.append((geom_name(model, g2), float(con.dist)))
        elif g2 == floor_gid:
            out.append((geom_name(model, g1), float(con.dist)))
    return out


def set_pose(model, data, kf, jadr, fadr, base_z):
    mujoco.mj_resetData(model, data)
    pos = list(getattr(kf, "pos", (0, 0, 0)) or (0, 0, 0))
    pos[2] = base_z
    data.qpos[fadr : fadr + 3] = pos
    data.qpos[fadr + 3 : fadr + 7] = getattr(kf, "rot", None) or (1, 0, 0, 0)
    for jname, adr in jadr.items():
        for pat, val in (getattr(kf, "joint_pos", {}) or {}).items():
            if re.fullmatch(pat, jname):
                data.qpos[adr] = val
    mujoco.mj_forward(model, data)


def min_floor_dist(model, data):
    cs = floor_contacts(model, data)
    return min((d for _, d in cs), default=1.0)


def rest_height(model, data, kf, jadr, fadr):
    """Bisection: lowest base_z with min penetration ~0 (just touching)."""
    hi, lo = 2.0, -0.5
    for _ in range(45):
        mid = 0.5 * (hi + lo)
        set_pose(model, data, kf, jadr, fadr, mid)
        if min_floor_dist(model, data) > 0:  # still floating
            hi = mid
        else:  # touching/penetrating
            lo = mid
    return hi


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    jadr = joint_qposadr(model)
    fadr = free_adr(model)

    for name, kf in POSES.items():
        z = rest_height(model, data, kf, jadr, fadr)
        set_pose(model, data, kf, jadr, fadr, z + 1e-4)  # tiny so contacts register
        # nudge into contact: drop a hair below rest to get the contact set
        set_pose(model, data, kf, jadr, fadr, z - 2e-3)
        cs = floor_contacts(model, data)
        kf_z = (getattr(kf, "pos", (0, 0, 0)) or (0, 0, 0))[2]
        print(f"\n=== {name}  keyframe_z={kf_z:.3f}  rest_z={z:.3f}  "
              f"clearance={kf_z - z:+.3f} m ===")
        if not cs:
            print("  (no floor contacts at rest)")
        seen = {}
        for gname, d in cs:
            seen[gname] = min(seen.get(gname, 0.0), d)
        for gname in sorted(seen):
            print(f"  {gname:32s} penetration={seen[gname]*1000:+7.2f} mm")


if __name__ == "__main__":
    main()
