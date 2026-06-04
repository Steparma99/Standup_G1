"""Interactive viewer to inspect, EDIT LIVE and validate the G1 initial poses.

Loads the REAL keyframes from g1_constants.py, lets you flip between them, and
lets you rotate / raise the robot in real time to fix a pose. When it looks
right, press C to print the exact (pos, rot) numbers to paste into
g1_constants.py.

KEYS (inside the viewer window):
  Load a pose:
    1  SUPINE     2  PRONE     3  HOME     4  KNEES_BENT
  Edit the pose LIVE (rotations are about the WORLD axes, 5 deg per press):
    W / S   pitch  + / -   (tilt forward / backward -- the supine<->prone axis)
    A / D   roll   + / -   (tilt left / right)
    Q / E   yaw    + / -   (turn around vertical)
    Z / X   height - / +   (lower / raise the base, 1 cm per press)
  Other:
    P   play / pause physics   (test if the pose holds or collapses)
    0   reset to the loaded keyframe (undo your edits)
    C   print current pos/rot   (copy these into g1_constants.py)
    mouse = orbit / zoom camera

After every edit it prints base_z and projected_gravity_b_z
(0 = lying flat, -1 = upright, +1 = upside down).

Usage:
    # interactive 3D window:
    conda run -n unitree_rl_cpu python scripts/pose_viewer.py
    # headless sanity check (no window):
    conda run -n unitree_rl_cpu python scripts/pose_viewer.py --check
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # make `import src...` work from anywhere

from src.assets.robots.unitree_g1.g1_constants import (  # noqa: E402
    HOME_KEYFRAME,
    KNEES_BENT_KEYFRAME,
    PRONE_KEYFRAME,
    SUPINE_KEYFRAME,
)

XML_PATH = HERE.parent / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"

POSES = {
    ord("1"): ("SUPINE", SUPINE_KEYFRAME),
    ord("2"): ("PRONE", PRONE_KEYFRAME),
    ord("3"): ("HOME", HOME_KEYFRAME),
    ord("4"): ("KNEES_BENT", KNEES_BENT_KEYFRAME),
}

ROT_STEP_DEG = 5.0
Z_STEP = 0.01


def joint_name_to_qposadr(model):
    out = {}
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            out[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)] = int(model.jnt_qposadr[j])
    return out


def free_joint_qposadr(model):
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[j])
    return None


def projected_gravity_z(quat) -> float:
    rotmat = np.zeros(9)
    mujoco.mju_quat2Mat(rotmat, np.asarray(quat, dtype=float))
    return float((rotmat.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0]))[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="headless: print all poses and exit")
    parser.add_argument("--start", default="SUPINE", help="pose to start on")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    joint_adr = joint_name_to_qposadr(model)
    free_adr = free_joint_qposadr(model)

    # Editable pose state. "kf" remembers the loaded keyframe so '0' can reset.
    st = {
        "pos": np.zeros(3),
        "quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "joints": {},
        "label": "",
        "kf": None,
        "paused": True,
    }

    def write_pose_to_data():
        mujoco.mj_resetData(model, data)
        data.qpos[free_adr : free_adr + 3] = st["pos"]
        data.qpos[free_adr + 3 : free_adr + 7] = st["quat"]
        for jname, adr in joint_adr.items():
            for pattern, value in st["joints"].items():
                if re.fullmatch(pattern, jname):
                    data.qpos[adr] = value
        mujoco.mj_forward(model, data)

    def report():
        print(f"    base_z = {st['pos'][2]:+.3f} m   "
              f"projected_gravity_b_z = {projected_gravity_z(st['quat']):+.3f}  "
              f"(0=flat, -1=upright, +1=upside-down)")

    def load_keyframe(label, kf):
        st["label"], st["kf"] = label, kf
        st["pos"] = np.array(getattr(kf, "pos", (0, 0, 0)) or (0, 0, 0), dtype=float)
        st["quat"] = np.array(getattr(kf, "rot", None) or (1, 0, 0, 0), dtype=float)
        st["joints"] = dict(getattr(kf, "joint_pos", {}) or {})
        st["paused"] = True
        write_pose_to_data()
        print(f"--> loaded {label}  (paused; edit W/S A/D Q/E Z/X, P=test, 0=reset, C=print)")
        report()

    # Headless check: print every pose and exit.
    if args.check:
        for _, (label, kf) in POSES.items():
            print(f"[{label}]")
            load_keyframe(label, kf)
        return

    from mujoco import viewer as mj_viewer

    def rotate_world(axis, deg):
        delta = np.zeros(4)
        mujoco.mju_axisAngle2Quat(delta, np.asarray(axis, dtype=float), math.radians(deg))
        new = np.zeros(4)
        mujoco.mju_mulQuat(new, delta, st["quat"])  # pre-multiply = rotate in world frame
        st["quat"] = new / np.linalg.norm(new)
        st["paused"] = True
        write_pose_to_data()
        report()

    def shift_z(dz):
        st["pos"][2] += dz
        st["paused"] = True
        write_pose_to_data()
        report()

    def print_values():
        w, x, y, z = st["quat"]
        px, py, pz = st["pos"]
        print("    ---- copy into g1_constants.py ----")
        print(f"      pos=({px:.3f}, {py:.3f}, {pz:.3f}),")
        print(f"      rot=({w:.4f}, {x:.4f}, {y:.4f}, {z:.4f}),")

    actions = {
        ord("W"): lambda: rotate_world((0, 1, 0), +ROT_STEP_DEG),
        ord("S"): lambda: rotate_world((0, 1, 0), -ROT_STEP_DEG),
        ord("A"): lambda: rotate_world((1, 0, 0), +ROT_STEP_DEG),
        ord("D"): lambda: rotate_world((1, 0, 0), -ROT_STEP_DEG),
        ord("Q"): lambda: rotate_world((0, 0, 1), +ROT_STEP_DEG),
        ord("E"): lambda: rotate_world((0, 0, 1), -ROT_STEP_DEG),
        ord("Z"): lambda: shift_z(-Z_STEP),
        ord("X"): lambda: shift_z(+Z_STEP),
        ord("C"): print_values,
        ord("0"): lambda: load_keyframe(st["label"], st["kf"]),
    }

    def key_callback(keycode):
        try:
            if keycode in POSES:
                load_keyframe(*POSES[keycode])
            elif keycode == ord("P"):
                st["paused"] = not st["paused"]
                print(f"    physics {'PAUSED' if st['paused'] else 'RUNNING'}")
            elif keycode in actions:
                actions[keycode]()
        except Exception as exc:  # never let a key crash the viewer
            print(f"    [key error] {exc}")

    start = next((p for p in POSES.values() if p[0] == args.start.upper()), POSES[ord("1")])

    print(__doc__)
    with mj_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        load_keyframe(*start)
        dt = model.opt.timestep
        while viewer.is_running():
            t0 = time.time()
            if not st["paused"]:
                data.ctrl[:] = 0.0
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
