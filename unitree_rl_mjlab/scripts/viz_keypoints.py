"""Visualize G1 HOME pose with the 12 critic keypoints as colored spheres.

  BLUE  spheres = upper body  (shoulders / elbows / wrists)
  GREEN spheres = lower body  (hips / knees / ankles)

Interactive:
  cd unitree_rl_mjlab
  MUJOCO_GL=glfw conda run -n unitree_rl_cpu python scripts/viz_keypoints.py

Headless (saves PNG to /tmp/g1_keypoints.png):
  MUJOCO_GL=osmesa conda run -n unitree_rl_cpu python scripts/viz_keypoints.py --png
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import mujoco
import mujoco.viewer

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.assets.robots.unitree_g1.g1_constants import (
    G1_CRITIC_KEYPOINTS,
    G1_XML,
    HOME_KEYFRAME,
    get_assets,
)
from mjlab.utils.string import resolve_expr

# Scene XML sits next to the robot XML: adds floor plane, checker texture, light.
_SCENE_XML = G1_XML.parent / "scene_g1_body29_hand14.xml"

# ── colours ────────────────────────────────────────────────────────────────────
_UPPER = frozenset({
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_elbow_link",          "right_elbow_link",
    "left_wrist_yaw_link",      "right_wrist_yaw_link",
})
_RGBA_UPPER = np.array([0.05, 0.45, 1.00, 0.90])   # blue
_RGBA_LOWER = np.array([0.05, 0.90, 0.20, 0.90])   # green
_SPHERE_R   = 0.04                                   # [m]

_SHORT = {
    "left_shoulder_pitch_link":  "L_shoulder",
    "right_shoulder_pitch_link": "R_shoulder",
    "left_elbow_link":           "L_elbow",
    "right_elbow_link":          "R_elbow",
    "left_wrist_yaw_link":       "L_wrist",
    "right_wrist_yaw_link":      "R_wrist",
    "left_hip_pitch_link":       "L_hip",
    "right_hip_pitch_link":      "R_hip",
    "left_knee_link":            "L_knee",
    "right_knee_link":           "R_knee",
    "left_ankle_pitch_link":     "L_ankle",
    "right_ankle_pitch_link":    "R_ankle",
}

# ── model building ─────────────────────────────────────────────────────────────

def _build_model() -> mujoco.MjModel:
    """Load scene (floor + lights), inject sphere sites at keypoints, compile."""
    spec = mujoco.MjSpec.from_file(str(_SCENE_XML))
    spec.assets = get_assets(spec.meshdir)
    added = 0
    for bname in G1_CRITIC_KEYPOINTS:
        body = spec.worldbody.find_child(bname)
        if body is None:
            print(f"  [warn] body '{bname}' not found in spec")
            continue
        site       = body.add_site()
        site.name  = f"kp_{_SHORT[bname]}"
        site.pos   = np.zeros(3)
        site.size  = np.full(3, _SPHERE_R)
        site.rgba  = _RGBA_UPPER.copy() if bname in _UPPER else _RGBA_LOWER.copy()
        added += 1
    print(f"  Injected {added}/{len(G1_CRITIC_KEYPOINTS)} keypoint sphere sites")
    return spec.compile()


# ── pose setting ───────────────────────────────────────────────────────────────

def _set_home_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Apply HOME_KEYFRAME: free-joint root + all hinge joints."""
    # Root free joint (always the first FREE joint in the G1 model)
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            adr = model.jnt_qposadr[jid]
            data.qpos[adr:adr + 3] = HOME_KEYFRAME.pos         # xyz
            data.qpos[adr + 3:adr + 7] = [1., 0., 0., 0.]     # quat w x y z (identity)
            break

    # Hinge joints: match HOME_KEYFRAME regex patterns
    jnames = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    vals = resolve_expr(HOME_KEYFRAME.joint_pos or {}, jnames, 0.0)
    for jname, val in zip(jnames, vals):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data.qpos[model.jnt_qposadr[jid]] = val

    mujoco.mj_forward(model, data)


# ── console table ──────────────────────────────────────────────────────────────

def _print_table(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    print(f"\n  {'label':<14} {'x':>8} {'y':>8} {'z':>8}  marker colour")
    print("  " + "-" * 56)
    for bname in G1_CRITIC_KEYPOINTS:
        bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        pos  = data.xpos[bid] if bid >= 0 else np.zeros(3)
        col  = "BLUE  (upper)" if bname in _UPPER else "GREEN (lower)"
        print(f"  {_SHORT[bname]:<14} {pos[0]:8.4f} {pos[1]:8.4f} {pos[2]:8.4f}  {col}")
    print()


# ── PNG save (headless fallback) ───────────────────────────────────────────────

def _save_png(model: mujoco.MjModel, data: mujoco.MjData,
              path: str = "/tmp/g1_keypoints.png") -> None:
    renderer = mujoco.Renderer(model, height=480, width=640)
    # Side view: camera looking along -Y axis
    renderer.update_scene(data)
    pixels = renderer.render()
    try:
        from PIL import Image
        Image.fromarray(pixels).save(path)
        print(f"  Saved → {path}")
        return
    except ImportError:
        pass
    try:
        import imageio
        imageio.imsave(path, pixels)
        print(f"  Saved → {path}")
        return
    except ImportError:
        pass
    # Last resort: numpy
    np.save(path.replace(".png", ".npy"), pixels)
    print(f"  PIL/imageio not found — saved raw array → {path.replace('.png', '.npy')}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--png", action="store_true",
                        help="render to /tmp/g1_keypoints.png instead of opening viewer")
    args = parser.parse_args()

    print("\nBuilding G1 model with keypoint sphere sites...")
    model = _build_model()
    data  = mujoco.MjData(model)
    _set_home_pose(model, data)
    _print_table(model, data)

    if args.png:
        print("Rendering to PNG (headless, no physics)...")
        _save_png(model, data)
        return

    paused = True   # physics off by default

    def _reset() -> None:
        _set_home_pose(model, data)
        data.qvel[:] = 0.0

    def _key_callback(key: int) -> None:
        nonlocal paused
        if key in (ord('P'), ord('p')):
            paused = not paused
            print("Physics", "PAUSED" if paused else "RUNNING")
        elif key in (ord('R'), ord('r')):
            _reset()
            print("Reset to HOME")

    STEPS_PER_FRAME = 5   # physics steps per render sync

    print("Opening MuJoCo viewer  (physics PAUSED at start):")
    print("  BLUE  spheres = shoulders / elbows / wrists")
    print("  GREEN spheres = hips / knees / ankles")
    print("  P = toggle physics on/off")
    print("  R = reset to HOME pose")
    print("  L = toggle site labels")
    print("  Esc = quit\n")

    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=_key_callback) as v:
        while v.is_running():
            if not paused:
                for _ in range(STEPS_PER_FRAME):
                    mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()
