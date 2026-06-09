"""Generate physically-valid ground keyframes (SUPINE / PRONE / SIDE) for the G1.

For each pose we:
  1. build the base orientation as a clean axis-angle rotation (quaternion),
  2. set the joints to a neutral posture,
  3. auto-compute the base height so the lowest collision geom just rests on the
     floor (no initial penetration -> the robot does not "explode" at reset),
  4. render the result and print the exact (pos, rot) to paste into
     src/assets/robots/unitree_g1/g1_constants.py.

Axis convention for the G1 base: x = forward, y = left, z = up.
  SUPINE     (on the back, face up)    = rotate -90 deg about +Y
  PRONE      (face down)               = rotate +90 deg about +Y
  SIDE_LEFT  (left side on ground)     = rotate +90 deg about +X  (+Y goes to -Z)
  SIDE_RIGHT (right side on ground)    = rotate -90 deg about +X  (-Y goes to -Z)

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

# Each pose: rotation axis, angle (deg), joint targets (regex -> radians) and a
# `settle` flag. When settle=True the base free joint is dropped onto the floor
# under physics while the JOINTS ARE HELD FIXED at their targets (see settle
# loop), so the stored joint_pos matches the stored base height exactly — no
# initial floor penetration.
#
# Axis/angle reminder (body frame x=fwd, y=left, z=up):
#   rotation about +X by -90deg  -> body LEFT  (+Y) points DOWN  => lies on LEFT side
#   rotation about +X by +90deg  -> body RIGHT (-Y) points DOWN  => lies on RIGHT side
#
# Side poses fold BOTH arms across the chest (strong elbow bend, arms slightly
# forward + tucked in) so they stay inside the body silhouette and never reach
# the floor when the torso rolls onto its side.
# Same near-neutral config as SUPINE/PRONE (which lie perfectly flat), so when
# rolled 90deg about X the body lies FLAT on its side instead of propping on a
# folded arm. Arms kept close to the torso (small roll, light elbow) so the
# bottom arm just lies alongside the body at floor level rather than sticking out.
_SIDE_JOINTS = {
    r".*_elbow_joint": 0.3,
    r"left_shoulder_roll_joint": 0.15,
    r"right_shoulder_roll_joint": -0.15,
    r".*_ankle_pitch_joint": -0.2,
}

# Seated upright on the ground: hips flexed ~90deg, legs extended forward,
# knees slightly bent, arms resting slightly forward. The base (pelvis) settles
# onto the buttocks; the hold-joints settle finds the resting lean/height.
_SEATED_JOINTS = {
    r".*_hip_pitch_joint": -1.5,    # flex hips ~90deg -> thighs point forward
    r".*_knee_joint": 0.3,          # legs almost straight out front
    r".*_ankle_pitch_joint": 0.0,
    r".*_shoulder_pitch_joint": 0.3,
    r"left_shoulder_roll_joint": 0.2,
    r"right_shoulder_roll_joint": -0.2,
    r".*_elbow_joint": 0.6,
}

POSES = {
    "SUPINE": {"axis": (0.0, 1.0, 0.0), "angle_deg": -90.0, "joint_pos": {r".*": 0.0}, "settle": True},
    "PRONE": {"axis": (0.0, 1.0, 0.0), "angle_deg": 90.0, "joint_pos": {r".*": 0.0}, "settle": True},
    "SIDE_LEFT": {"axis": (1.0, 0.0, 0.0), "angle_deg": -90.0, "joint_pos": _SIDE_JOINTS, "settle": True},
    "SIDE_RIGHT": {"axis": (1.0, 0.0, 0.0), "angle_deg": +90.0, "joint_pos": _SIDE_JOINTS, "settle": True},
    "SEATED": {"axis": (0.0, 1.0, 0.0), "angle_deg": 0.0, "joint_pos": _SEATED_JOINTS, "settle": True},
}

VIEWS = [(90.0, -15.0, "side"), (50.0, -45.0, "diag")]
CLEARANCE = 0.03  # start slightly above the floor, then let it settle
SETTLE_STEPS = 2000  # physics steps to let the base come to rest on the ground


def joint_name_to_qposadr(model):
    out = {}
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            out[name] = int(model.jnt_qposadr[j])
    return out


def joint_name_to_dofadr(model):
    out = {}
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            out[name] = int(model.jnt_dofadr[j])
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


def _min_floor_contact_dist(model, data) -> tuple[float, int]:
    """Run collision at the current qpos and return the most-negative distance of
    any robot-vs-floor contact, plus the number of such contacts.

    dist < 0 = penetration, dist > 0 = gap. Floor geoms live on the worldbody
    (bodyid 0). Self-collisions are ignored so they can't fool the height search.
    """
    mujoco.mj_forward(model, data)
    min_dist = math.inf
    n = 0
    for c in range(data.ncon):
        con = data.contact[c]
        b1 = int(model.geom_bodyid[con.geom1])
        b2 = int(model.geom_bodyid[con.geom2])
        if (b1 == 0) != (b2 == 0):  # exactly one side is the floor
            min_dist = min(min_dist, float(con.dist))
            n += 1
    if n == 0:
        return math.inf, 0  # floating: no floor contact at all
    return min_dist, n


def settle_height_kinematic(model, data, free_adr, clearance, tol=1e-3) -> None:
    """Find the resting base height with REAL mesh collision, no dynamics.

    Bisection: start high (no contact) and low (penetrating), converge on the
    height where the lowest robot point just touches the floor, then add
    `clearance`. Uses MuJoCo's actual narrowphase so the gap is tight and
    accurate (unlike a bounding-sphere estimate) and there is no integrator
    instability (unlike free-settling under gravity).
    """
    h_high = 2.0  # certainly floating
    h_low = -0.5  # certainly penetrating for any ground pose
    for _ in range(40):
        h_mid = 0.5 * (h_high + h_low)
        data.qpos[free_adr + 2] = h_mid
        min_dist, n = _min_floor_contact_dist(model, data)
        penetrating = (n > 0) and (min_dist < 0.0)
        if penetrating:
            h_low = h_mid
        else:
            h_high = h_mid
        if (h_high - h_low) < tol:
            break
    # h_high is the lowest non-penetrating height (just touching). Add clearance.
    data.qpos[free_adr + 2] = h_high + clearance
    mujoco.mj_forward(model, data)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    joint_adr = joint_name_to_qposadr(model)
    joint_dof = joint_name_to_dofadr(model)
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

        # Expand the regex joint dict to per-joint qpos targets (last match wins).
        joint_targets: dict[str, float] = {}
        for jname, adr in joint_adr.items():
            for pattern, value in p["joint_pos"].items():
                if re.fullmatch(pattern, jname):
                    joint_targets[jname] = float(value)
        for jname, value in joint_targets.items():
            data.qpos[joint_adr[jname]] = value
        mujoco.mj_forward(model, data)

        # Kinematic placement (NO dynamics): bisection on base height using real
        # mesh collision finds the resting height where the lowest robot point
        # just touches the floor; CLEARANCE adds a tiny air gap. This keeps the
        # joint configuration EXACTLY as specified (no gravity drift) and
        # guarantees no penetration, with a gap small enough that the first
        # physics steps close it gently at training time. Free-settling under
        # gravity was abandoned: it let the arms flop, so the stored joints no
        # longer matched the height.
        settle_height_kinematic(model, data, free_adr, CLEARANCE)

        base_pos = data.qpos[free_adr : free_adr + 3].copy()
        base_quat = data.qpos[free_adr + 3 : free_adr + 7].copy()
        # Re-normalize quaternion.
        base_quat /= np.linalg.norm(base_quat)
        lin_vel = 0.0  # kinematic placement: robot is at rest by construction

        # Numeric check: projected gravity z in base frame (0 = lying flat).
        rotmat = np.zeros(9)
        mujoco.mju_quat2Mat(rotmat, base_quat)
        grav_b = rotmat.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        # Penetration check: most-negative contact distance against the floor
        # (dist < 0 means geoms overlap -> contact spike at reset = BAD).
        min_contact_dist = 0.0
        for c in range(data.ncon):
            min_contact_dist = min(min_contact_dist, float(data.contact[c].dist))

        print(f"=== {name} (kinematic placement) ===")
        print(f"  rot (w,x,y,z) = ({base_quat[0]:+.4f}, {base_quat[1]:+.4f}, {base_quat[2]:+.4f}, {base_quat[3]:+.4f})")
        print(f"  pos (x,y,z)   = ({base_pos[0]:+.4f}, {base_pos[1]:+.4f}, {base_pos[2]:+.4f}) m")
        print(f"  projected_gravity_b_z = {grav_b[2]:+.3f}   (0 = flat on ground)")
        print(f"  active contacts={data.ncon}  min_contact_dist={min_contact_dist:+.4f} m  (>= 0 = no penetration)")
        print(f"  joint_pos (copy into g1_constants.py):")
        for jname, value in sorted(joint_targets.items()):
            if abs(value) > 1e-9:
                print(f"      \"{jname}\": {value},")
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
