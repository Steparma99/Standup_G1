"""Interactive viewer to inspect, EDIT LIVE and validate the G1 initial poses.

Loads the REAL keyframes from g1_constants.py, lets you flip between them, and
lets you rotate / raise the robot in real time to fix a pose. When it looks
right, press C to print the exact (pos, rot) numbers to paste into
g1_constants.py.

KEYS (inside the viewer window):
  Load a pose:
    1  SUPINE     2  PRONE     3  HOME     4  KNEES_BENT
    5  SIDE_LEFT  6  SIDE_RIGHT  7  SEATED
  Edit the pose LIVE (rotations are about the WORLD axes, 5 deg per press):
    W / S   pitch  + / -   (tilt forward / backward -- the supine<->prone axis)
    A / D   roll   + / -   (tilt left / right)
    Q / E   yaw    + / -   (turn around vertical)
    Z / X   height - / +   (lower / raise the base, 1 cm per press)
  Other:
    P   play / pause physics   (test if the pose holds or collapses)
    G   toggle COLLISION geoms (group 3) on/off  (see hand/head/foot capsules)
    0   reset to the loaded keyframe (undo your edits)
    C   print current pos/rot   (copy these into g1_constants.py)
    J   print joint positions + reward diagnostics (see below)
    mouse = orbit / zoom camera

After every edit it prints base_z, projected_gravity_b_z, foot heights, and
ankle tilt metric (reward-relevant values). Press J for full joint list +
computed reward values.

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
    SEATED_KEYFRAME,
    SIDE_LEFT_KEYFRAME,
    SIDE_RIGHT_KEYFRAME,
    SUPINE_KEYFRAME,
)

XML_PATH = HERE.parent / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"

POSES = {
    ord("1"): ("SUPINE", SUPINE_KEYFRAME),
    ord("2"): ("PRONE", PRONE_KEYFRAME),
    ord("3"): ("HOME", HOME_KEYFRAME),
    ord("4"): ("KNEES_BENT", KNEES_BENT_KEYFRAME),
    ord("5"): ("SIDE_LEFT", SIDE_LEFT_KEYFRAME),
    ord("6"): ("SIDE_RIGHT", SIDE_RIGHT_KEYFRAME),
    ord("7"): ("SEATED", SEATED_KEYFRAME),
}

ROT_STEP_DEG = 5.0
Z_STEP = 0.001

# ---------------------------------------------------------------------------
# Reward parameter constants — keep in sync with getup_env_cfg.py / rewards.py
# ---------------------------------------------------------------------------
_POST_BASE_HEIGHT_TARGET = 0.757   # getup_env_cfg.py post_base_height target_height
_POST_BASE_HEIGHT_SCALE  = 20.0   # getup_env_cfg.py post_base_height scale
_POST_BASE_HEIGHT_WEIGHT = 10.0   # getup_env_cfg.py post_base_height weight
_ANKLE_PARALLEL_THR      = 0.05   # getup_env_cfg.py style_ankle_parallel tilt_threshold
_ANKLE_PARALLEL_REWARD   = 20.0   # getup_env_cfg.py style_ankle_parallel reward
_POST_FEET_PAR_SCALE     = 20.0   # getup_env_cfg.py post_feet_parallel scale
_POST_FEET_PAR_CLIP_MIN  = 0.001  # getup_env_cfg.py post_feet_parallel clip_min
_POST_FEET_PAR_WEIGHT    = 2.5    # getup_env_cfg.py post_feet_parallel weight
_POST_ORI_SCALE          = 8.0    # getup_env_cfg.py post_base_orientation scale
_POST_ORI_WEIGHT         = 10.0   # getup_env_cfg.py post_base_orientation weight
_WAIST_LOWER             = -0.25  # getup_env_cfg.py style_waist_upright lower
_WAIST_UPPER             = 0.25   # getup_env_cfg.py style_waist_upright upper
_WAIST_WEIGHT            = -8.0   # getup_env_cfg.py style_waist_upright weight (negative)
_SHANK_ORI_LOWER         = 0.8    # getup_env_cfg.py style_shank_orientation lower
_SHANK_ORI_MARGIN        = 1.0    # getup_env_cfg.py style_shank_orientation margin
_SHANK_ORI_VAL_AT_MARGIN = 0.1   # getup_env_cfg.py style_shank_orientation value_at_margin
_SHANK_ORI_WEIGHT        = 10.0   # getup_env_cfg.py style_shank_orientation weight
_POST_GATE_THRESHOLD     = 0.65   # rewards.py H_STAGE2
_POST_GATE_BAND          = 0.06   # rewards.py _post_gate band
_TASK_HEAD_HEIGHT_LOWER  = 0.80   # getup_env_cfg.py task_head_height lower (torso_link z at HOME)
_TASK_HEAD_HEIGHT_MARGIN = 1.0    # getup_env_cfg.py task_head_height margin
_TASK_HEAD_HEIGHT_VAM    = 0.1    # getup_env_cfg.py task_head_height value_at_margin
_TASK_HEAD_HEIGHT_WEIGHT = 1.0    # getup_env_cfg.py task_head_height weight
_FOOT_DISP_SCALE         = 2.0    # getup_env_cfg.py style_foot_displacement scale
_FOOT_DISP_CLIP_MIN      = 0.0    # getup_env_cfg.py style_foot_displacement clip_min
_FOOT_DISP_WEIGHT        = 2.5    # getup_env_cfg.py style_foot_displacement weight
_FEET_YAW_SCALE          = 10.0   # getup_env_cfg.py post_feet_yaw scale
_FEET_YAW_WEIGHT         = 2.5    # getup_env_cfg.py post_feet_yaw weight


# ---------------------------------------------------------------------------
# Geometry helpers (raw MuJoCo, no env/sensor layer)
# ---------------------------------------------------------------------------

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


def _rotmat_from_quat(quat) -> np.ndarray:
    """Return 3×3 rotation matrix (body→world) from quaternion (w,x,y,z)."""
    rotmat = np.zeros(9)
    mujoco.mju_quat2Mat(rotmat, np.asarray(quat, dtype=float))
    return rotmat.reshape(3, 3)


def projected_gravity_z(quat) -> float:
    return float((_rotmat_from_quat(quat).T @ np.array([0.0, 0.0, -1.0]))[2])


def _proj_gravity(quat) -> np.ndarray:
    """Gravity [0,0,-1] in body frame; shape (3,)."""
    return _rotmat_from_quat(quat).T @ np.array([0.0, 0.0, -1.0])


def _body_xquat(model, data, name) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xquat[bid].copy() if bid >= 0 else np.array([1., 0., 0., 0.])


def _body_xpos(model, data, name) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy() if bid >= 0 else np.zeros(3)


def _site_xpos(model, data, name) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xpos[sid].copy() if sid >= 0 else np.zeros(3)


def _foot_heights(model, data) -> tuple[float, float]:
    """Z of left and right foot sites (world frame; ~0 when touching floor)."""
    zL = float(_site_xpos(model, data, "left_foot")[2])
    zR = float(_site_xpos(model, data, "right_foot")[2])
    return zL, zR


def _ankle_tilt(model, data) -> tuple[float, float]:
    """Per-foot tilt metric ||proj_grav_xy||² (0=flat, grows with tilt)."""
    tilts = []
    for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        g = _proj_gravity(_body_xquat(model, data, name))
        tilts.append(float(g[0] ** 2 + g[1] ** 2))
    return tilts[0], tilts[1]


def _shank_cos(model, data) -> tuple[float, float]:
    """Vertical component of the knee→ankle unit vector per leg (1=vertical shank)."""
    cosines = []
    for side in ("left", "right"):
        knee = _body_xpos(model, data, f"{side}_knee_link")
        ankle = _body_xpos(model, data, f"{side}_ankle_roll_link")
        vec = knee - ankle
        cosines.append(float(vec[2] / (np.linalg.norm(vec) + 1e-6)))
    return cosines[0], cosines[1]


def _f_tol(x: float, lower: float, margin: float, value_at_margin: float) -> float:
    """Gaussian tolerance (one-sided lower bound, upper=inf)."""
    if x >= lower:
        return 1.0
    d = (lower - x) / margin
    scale = math.sqrt(-2.0 * math.log(value_at_margin))
    return math.exp(-0.5 * (d * scale) ** 2)


def _post_gate(h: float) -> float:
    """Smooth post-task gate [0,1]: ramps from 0→1 over [0.59, 0.71] m."""
    return min(max((h - (_POST_GATE_THRESHOLD - _POST_GATE_BAND)) / (2.0 * _POST_GATE_BAND), 0.0), 1.0)


# ---------------------------------------------------------------------------
# Reward diagnostics printer
# ---------------------------------------------------------------------------

def _print_reward_diagnostics(model, data, joint_adr, base_quat, base_pos: np.ndarray) -> None:
    """Compute and print reward values for the current pose (matching training params)."""
    base_z = float(base_pos[2])
    gate = _post_gate(base_z)

    # task_head_height (torso_link proxy; always active, no gate)
    torso_z = float(_body_xpos(model, data, "torso_link")[2])
    head_rwd = _f_tol(torso_z, _TASK_HEAD_HEIGHT_LOWER, _TASK_HEAD_HEIGHT_MARGIN, _TASK_HEAD_HEIGHT_VAM) * _TASK_HEAD_HEIGHT_WEIGHT
    print(f"  task_head_height     : torso_link_z={torso_z:.4f}  lower={_TASK_HEAD_HEIGHT_LOWER:.2f}  "
          f"→ reward={head_rwd:+.3f}  (no gate, weight={_TASK_HEAD_HEIGHT_WEIGHT})")

    # post_base_height
    h_err = base_z - _POST_BASE_HEIGHT_TARGET
    h_rwd = math.exp(-_POST_BASE_HEIGHT_SCALE * h_err ** 2) * _POST_BASE_HEIGHT_WEIGHT * gate
    print(f"  post_base_height     : pelvis_z={base_z:+.3f} m  target={_POST_BASE_HEIGHT_TARGET:.3f}  "
          f"err={h_err:+.4f}  → reward={h_rwd:+.2f}  (gate={gate:.2f}, weight={_POST_BASE_HEIGHT_WEIGHT})")

    # post_base_orientation
    g_base = _proj_gravity(base_quat)
    theta_xy_sq = float(g_base[0] ** 2 + g_base[1] ** 2)
    ori_rwd = math.exp(-_POST_ORI_SCALE * theta_xy_sq) * _POST_ORI_WEIGHT * gate
    print(f"  post_base_orientation: ||theta_xy||²={theta_xy_sq:.4f}  "
          f"→ reward={ori_rwd:+.2f}  (gate={gate:.2f}, weight={_POST_ORI_WEIGHT})")

    # style_ankle_parallel (NO gate — fires at all heights)
    tL, tR = _ankle_tilt(model, data)
    mean_tilt = (tL + tR) / 2.0
    ankle_rwd = _ANKLE_PARALLEL_REWARD * math.exp(-mean_tilt / (2.0 * _ANKLE_PARALLEL_THR))
    print(f"  style_ankle_parallel : tilt_L={tL:.4f}  tilt_R={tR:.4f}  mean={mean_tilt:.4f}  "
          f"(thr={_ANKLE_PARALLEL_THR})  → reward={ankle_rwd:+.2f}  (no gate)")

    # style_foot_displacement (CoM-in-support; gated above H_STAGE2)
    base_xy = base_pos[:2]
    fL = _site_xpos(model, data, "left_foot")[:2]
    fR = _site_xpos(model, data, "right_foot")[:2]
    d2L = float(np.sum((fL - base_xy) ** 2))
    d2R = float(np.sum((fR - base_xy) ** 2))
    cd2L = max(d2L, _FOOT_DISP_CLIP_MIN)
    cd2R = max(d2R, _FOOT_DISP_CLIP_MIN)
    rL = math.exp(-_FOOT_DISP_SCALE * cd2L)
    rR = math.exp(-_FOOT_DISP_SCALE * cd2R)
    foot_disp_rwd = (rL + rR) * _FOOT_DISP_WEIGHT * gate
    print(f"  style_foot_disp      : L d²={d2L:.4f}→clamp={cd2L:.4f}→exp={rL:.4f}  "
          f"R d²={d2R:.4f}→clamp={cd2R:.4f}→exp={rR:.4f}  "
          f"→ reward={foot_disp_rwd:+.3f}  (gate={gate:.2f}, max={(2*_FOOT_DISP_WEIGHT):+.1f})")

    # post_feet_parallel
    zL, zR = _foot_heights(model, data)
    diff = max(abs(zL - zR), _POST_FEET_PAR_CLIP_MIN)
    feet_par_rwd = math.exp(-_POST_FEET_PAR_SCALE * diff) * _POST_FEET_PAR_WEIGHT * gate
    print(f"  post_feet_parallel   : foot_z_L={zL:+.4f}  foot_z_R={zR:+.4f}  "
          f"|diff|={diff:.4f} (clip={_POST_FEET_PAR_CLIP_MIN})  "
          f"→ reward={feet_par_rwd:+.2f}  (gate={gate:.2f}, weight={_POST_FEET_PAR_WEIGHT})")

    # post_feet_yaw (hip yaw = 0 → toes forward)
    yL = float(data.qpos[joint_adr["left_hip_yaw_joint"]]) if "left_hip_yaw_joint" in joint_adr else 0.0
    yR = float(data.qpos[joint_adr["right_hip_yaw_joint"]]) if "right_hip_yaw_joint" in joint_adr else 0.0
    feet_yaw_rwd = math.exp(-_FEET_YAW_SCALE * (yL ** 2 + yR ** 2)) * _FEET_YAW_WEIGHT * gate
    print(f"  post_feet_yaw        : yaw_L={yL:+.4f} rad  yaw_R={yR:+.4f} rad  "
          f"sum_sq={yL**2+yR**2:.5f}  → reward={feet_yaw_rwd:+.3f}  "
          f"(gate={gate:.2f}, max={_FEET_YAW_WEIGHT:+.1f})")

    # style_waist_upright (gate_lo=0.62, band=0.08 from getup_env_cfg.py)
    waist_gate_lo, waist_band = 0.62, 0.08
    waist_gate = min(max((base_z - (waist_gate_lo - waist_band)) / (2.0 * waist_band), 0.0), 1.0)
    wp = float(data.qpos[joint_adr["waist_pitch_joint"]]) if "waist_pitch_joint" in joint_adr else 0.0
    wr = float(data.qpos[joint_adr["waist_roll_joint"]]) if "waist_roll_joint" in joint_adr else 0.0
    over_p = max(wp - _WAIST_UPPER, 0.0) + max(_WAIST_LOWER - wp, 0.0)
    over_r = max(wr - _WAIST_UPPER, 0.0) + max(_WAIST_LOWER - wr, 0.0)
    waist_pen = _WAIST_WEIGHT * (over_p ** 2 + over_r ** 2) * waist_gate  # weight is negative
    print(f"  style_waist_upright  : pitch={wp:+.3f}  roll={wr:+.3f}  "
          f"band=[{_WAIST_LOWER},{_WAIST_UPPER}]  "
          f"→ penalty={waist_pen:+.4f}  (gate={waist_gate:.2f}, weight={_WAIST_WEIGHT})")

    # style_shank_orientation (gated above H_STAGE1=0.45)
    shank_gate = min(max((base_z - (0.45 - 0.08)) / (2.0 * 0.08), 0.0), 1.0)
    cos_L, cos_R = _shank_cos(model, data)
    mean_cos = (cos_L + cos_R) / 2.0
    shank_rwd = _f_tol(mean_cos, _SHANK_ORI_LOWER, _SHANK_ORI_MARGIN, _SHANK_ORI_VAL_AT_MARGIN) * _SHANK_ORI_WEIGHT * shank_gate
    print(f"  style_shank_orient   : cos_L={cos_L:.3f}  cos_R={cos_R:.3f}  mean={mean_cos:.3f}  "
          f"(lower={_SHANK_ORI_LOWER})  → reward={shank_rwd:+.2f}  (gate={shank_gate:.2f}, weight={_SHANK_ORI_WEIGHT})")

    print(f"  ── post_gate @ h={base_z:.3f} m = {gate:.3f}  "
          f"({'ACTIVE' if gate > 0.95 else 'PARTIAL' if gate > 0.05 else 'OFF'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="headless: print all poses and exit")
    parser.add_argument("--start", default="SUPINE", help="pose to start on")
    parser.add_argument("--collision", action="store_true",
                        help="start with collision geoms (group 3) visible")
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
        "show_collision": args.collision,
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
        base_z = st["pos"][2]
        grav_z = projected_gravity_z(st["quat"])
        zL, zR = _foot_heights(model, data)
        tL, tR = _ankle_tilt(model, data)
        mean_tilt = (tL + tR) / 2.0
        gate = _post_gate(base_z)
        print(f"    base_z={base_z:+.3f} m  grav_z={grav_z:+.3f}  (0=flat,-1=upright)  "
              f"post_gate={gate:.2f}")
        print(f"    foot_z_L={zL:+.5f}  foot_z_R={zR:+.5f}  ankle_tilt={mean_tilt:.4f}  "
              f"(0=flat,>0.05=crooked)")

    def load_keyframe(label, kf):
        st["label"], st["kf"] = label, kf
        st["pos"] = np.array(getattr(kf, "pos", (0, 0, 0)) or (0, 0, 0), dtype=float)
        st["quat"] = np.array(getattr(kf, "rot", None) or (1, 0, 0, 0), dtype=float)
        st["joints"] = dict(getattr(kf, "joint_pos", {}) or {})
        st["paused"] = True
        write_pose_to_data()
        print(f"--> loaded {label}  (paused; edit W/S A/D Q/E Z/X, P=test, 0=reset, C=print, J=rewards)")
        report()

    # Headless check: print every pose and exit.
    if args.check:
        for _, (label, kf) in POSES.items():
            print(f"\n[{label}]")
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

    def print_joints_and_rewards():
        print("\n    ---- joint positions (copy into g1_constants.py) ----")
        print('    joint_pos={')
        for jname in sorted(joint_adr):
            val = float(data.qpos[joint_adr[jname]])
            if abs(val) > 1e-6:  # skip near-zero joints
                print(f'      "{jname}": {val:.4f},')
        print('    }')
        print("\n    ---- reward diagnostics (params from env_cfgs.py) ----")
        _print_reward_diagnostics(model, data, joint_adr, st["quat"], st["pos"])

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
        ord("J"): print_joints_and_rewards,
        ord("0"): lambda: load_keyframe(st["label"], st["kf"]),
    }

    def key_callback(keycode):
        try:
            if keycode in POSES:
                load_keyframe(*POSES[keycode])
            elif keycode == ord("P"):
                st["paused"] = not st["paused"]
                print(f"    physics {'PAUSED' if st['paused'] else 'RUNNING'}")
            elif keycode == ord("G"):
                st["show_collision"] = not st["show_collision"]
                print(f"    collision geoms (group 3) "
                      f"{'VISIBLE' if st['show_collision'] else 'hidden'}")
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
            # group 2 = visual meshes, group 3 = collision geoms (hidden by default)
            viewer.opt.geomgroup[3] = 1 if st["show_collision"] else 0
            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
