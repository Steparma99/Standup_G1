"""Render the G1 init-state keyframes (HOME / KNEES_BENT / SUPINE) to PNG.

Loads the raw MuJoCo model, applies each keyframe's base pose + joint targets
(the regex joint_pos dicts are expanded against the model joint names exactly
like mjlab does at reset), runs forward kinematics and saves offscreen renders
from a few angles. Works on AMD/CPU because it uses the osmesa offscreen GL
backend — no interactive window required.

Usage:
    conda run -n unitree_rl_cpu python scripts/visualize_keyframes.py
"""

from __future__ import annotations

import os

# Must be set before importing mujoco. osmesa = pure-software offscreen GL,
# works on AMD/headless without a display server.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import re
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

# --- Keyframe definitions (mirrored from src/assets/robots/unitree_g1/g1_constants.py) ---
# Kept as plain dicts here so this script has no dependency on the mjlab config
# stack; if you change a keyframe in g1_constants.py, update it here too.
KEYFRAMES = {
    "HOME": {
        "pos": (0.0, 0.0, 0.8),
        "rot": (1.0, 0.0, 0.0, 0.0),
        "joint_pos": {
            r".*_hip_pitch_joint": -0.1,
            r".*_knee_joint": 0.3,
            r".*_ankle_pitch_joint": -0.2,
            r".*_shoulder_pitch_joint": 0.35,
            r".*_elbow_joint": 0.87,
            r"left_shoulder_roll_joint": 0.18,
            r"right_shoulder_roll_joint": -0.18,
        },
    },
    "KNEES_BENT": {
        "pos": (0.0, 0.0, 0.78),
        "rot": (1.0, 0.0, 0.0, 0.0),
        "joint_pos": {
            r".*_hip_pitch_joint": -0.312,
            r".*_knee_joint": 0.669,
            r".*_ankle_pitch_joint": -0.363,
            r".*_elbow_joint": 0.6,
            r"left_shoulder_roll_joint": 0.2,
            r"left_shoulder_pitch_joint": 0.2,
            r"right_shoulder_roll_joint": -0.2,
            r"right_shoulder_pitch_joint": 0.2,
        },
    },
    "SUPINE": {
        "pos": (0.0, 0.0, 0.1),
        "rot": (-0.866, -0.010, 0.500, 0.018),
        "joint_pos": {r".*": 0.0},
    },
}

# Camera angles to render per keyframe: (azimuth, elevation, label).
VIEWS = [
    (90.0, -10.0, "side"),
    (135.0, -25.0, "diag"),
]

HERE = Path(__file__).resolve().parent
XML_PATH = HERE.parent / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"
OUT_DIR = HERE.parent / "keyframe_check"


def joint_name_to_qposadr(model: mujoco.MjModel) -> dict[str, int]:
    """Map each hinge/slide joint name to its qpos address."""
    out = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = model.jnt_type[j]
        if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            out[name] = int(model.jnt_qposadr[j])
    return out


def find_free_joint_qposadr(model: mujoco.MjModel) -> int | None:
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[j])
    return None


def apply_keyframe(model, data, kf, joint_adr: dict[str, int]) -> None:
    mujoco.mj_resetData(model, data)

    # Base free joint: [x y z qw qx qy qz].
    free_adr = find_free_joint_qposadr(model)
    if free_adr is not None:
        data.qpos[free_adr : free_adr + 3] = kf["pos"]
        data.qpos[free_adr + 3 : free_adr + 7] = kf["rot"]

    # Expand regex joint targets against the actual joint names (last match wins).
    for name, adr in joint_adr.items():
        for pattern, value in kf["joint_pos"].items():
            if re.fullmatch(pattern, name):
                data.qpos[adr] = value

    mujoco.mj_forward(model, data)


def main() -> None:
    if not XML_PATH.exists():
        raise FileNotFoundError(f"Model not found: {XML_PATH}")
    OUT_DIR.mkdir(exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    joint_adr = joint_name_to_qposadr(model)
    print(f"[INFO] Model loaded: {model.njnt} joints, {len(joint_adr)} actuated DOF")

    w = min(1024, int(model.vis.global_.offwidth))
    h = min(768, int(model.vis.global_.offheight))
    renderer = mujoco.Renderer(model, height=h, width=w)

    cam = mujoco.MjvCamera()
    free_adr = find_free_joint_qposadr(model)

    for kf_name, kf in KEYFRAMES.items():
        apply_keyframe(model, data, kf, joint_adr)

        # Numeric sanity: pelvis height + projected-gravity z (body up-ness).
        base_z = float(data.qpos[free_adr + 2]) if free_adr is not None else float("nan")
        # gravity in world is (0,0,-1); rotate into base frame via base quat.
        quat = data.qpos[free_adr + 3 : free_adr + 7] if free_adr is not None else None
        grav_b_z = float("nan")
        if quat is not None:
            rotmat = np.zeros(9)
            mujoco.mju_quat2Mat(rotmat, quat)
            R = rotmat.reshape(3, 3)
            grav_b = R.T @ np.array([0.0, 0.0, -1.0])
            grav_b_z = float(grav_b[2])
        print(
            f"[{kf_name:10s}] base_z={base_z:+.3f} m  "
            f"projected_gravity_b_z={grav_b_z:+.3f}  "
            f"(=-1 upright, +1 upside-down)"
        )

        # lookat = base position, raised a bit toward body center.
        lookat = np.array(kf["pos"], dtype=float)
        lookat[2] = max(lookat[2], 0.3)
        for azimuth, elevation, label in VIEWS:
            cam.lookat[:] = lookat
            cam.distance = 2.6
            cam.azimuth = azimuth
            cam.elevation = elevation
            renderer.update_scene(data, camera=cam)
            pixels = renderer.render()
            out = OUT_DIR / f"{kf_name}_{label}.png"
            Image.fromarray(pixels).save(out)
            print(f"        saved {out.relative_to(HERE.parent)}")

    renderer.close()
    print(f"\n[DONE] PNGs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
