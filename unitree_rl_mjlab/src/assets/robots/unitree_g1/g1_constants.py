"""Unitree G1 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

try:
  from mjlab.actuator.delayed_actuator import DelayedActuatorCfg
except (ImportError, ModuleNotFoundError):
  DelayedActuatorCfg = None

##
# MJCF and assets.
##

G1_XML: Path = (
  SRC_PATH / "assets" / "robots" / "g1" / "g1_body29_hand14.xml"
)
assert G1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  assets_dir = G1_XML.parent / meshdir
  for asset_path in assets_dir.rglob("*"):
    if asset_path.is_file():
      assets[str(asset_path.relative_to(assets_dir))] = asset_path.read_bytes()
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Motor specs (from Unitree).
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=5.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_ACTUATOR_4010 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)
G1_ACTUATOR_HAND = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_hand_(thumb|middle|index)_\d_joint",),
  stiffness=5.0,
  damping=0.2,
  effort_limit=1.4,
)

# Waist pitch/roll and ankles are 4-bar linkages with 2 5020 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

# ---------------------------------------------------------------------------
# HoST get-up PD override.
#
# HoST sets hip Kp=200, knee Kp=275, and arm (shoulder/elbow/wrist) Kp=100
# (compensating the sim-to-real torque gap observed in joint phase plots, and
# making the arms responsive enough to push off the ground); damping (Kd) is
# unchanged everywhere. The baseline groups above mix joint *function* with motor
# *type* — the 7520-14 group is hip_pitch + hip_yaw + waist_yaw, and the 7520-22
# group is hip_roll + knee — so we cannot just bump a group's stiffness without
# dragging waist_yaw to 200 or hip_roll to 275. We therefore SPLIT those two
# groups by function below, and likewise give the 5020 arm group its own HoST
# stiffness; each sub-group keeps its own motor's armature / effort_limit /
# damping (those are physical, per motor) but takes the HoST stiffness. These are
# used ONLY by get_g1_supine_robot_cfg_host() via G1_ARTICULATION_HOST; the shared
# G1_ARTICULATION / G1_ACTION_SCALE are left untouched, so the velocity task and
# every other G1 task keep their original gains. (waist_yaw and the 4010 wrist
# pitch/yaw keep their armature-derived stiffness — HoST has no override for them.)
# ---------------------------------------------------------------------------
HOST_HIP_KP = 200.0
HOST_KNEE_KP = 275.0
# Arm stiffness (shoulder/elbow/wrist_roll, all 5020 motors). HoST uses Kp=100 for
# shoulder/elbow/wrist; the baseline armature-derived stiffness here is ~0.9, which
# makes the arms far too compliant to push during get-up (they barely move). Boost
# to the HoST value so the arms are responsive. Kd / effort_limit / armature stay
# at the physical 5020 values.
HOST_ARM_KP = 100.0

# Hip pitch + hip yaw (7520-14 motor) — HoST Kp=200.
G1_ACTUATOR_HOST_HIP_PITCH_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
  stiffness=HOST_HIP_KP,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
# Waist yaw (7520-14 motor) — UNCHANGED (armature-derived stiffness).
G1_ACTUATOR_HOST_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
# Hip roll (7520-22 motor) — HoST Kp=200.
G1_ACTUATOR_HOST_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=HOST_HIP_KP,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
# Knee (7520-22 motor) — HoST Kp=275.
G1_ACTUATOR_HOST_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=HOST_KNEE_KP,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
# Arms: shoulder (pitch/roll/yaw) + elbow + wrist_roll (5020 motor) — HoST Kp=100.
# Same target set as G1_ACTUATOR_5020 but with the HoST arm stiffness so the arms
# are responsive enough to push off the ground during get-up. Replaces
# G1_ACTUATOR_5020 in G1_ARTICULATION_HOST.
G1_ACTUATOR_HOST_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=HOST_ARM_KP,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)

##
# Anatomical keypoints — body links used as critic observations and as the
# geometric HOME reference for the final "maintain standing" task stage.
# 12 bodies × 3 (XYZ) = 36 critic dims when enabled.
# Wire into critic obs via mdp.body_keypoints + SceneEntityCfg(body_names=G1_CRITIC_KEYPOINTS).
G1_CRITIC_KEYPOINTS: tuple[str, ...] = (
  "left_shoulder_pitch_link",
  "right_shoulder_pitch_link",
  "left_elbow_link",
  "right_elbow_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "left_hip_pitch_link",
  "right_hip_pitch_link",
  "left_knee_link",
  "right_knee_link",
  "left_ankle_pitch_link",
  "right_ankle_pitch_link",
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.75),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_shoulder_pitch_joint": 0.20,
    ".*_elbow_joint": 0.60,
    "left_shoulder_roll_joint": 0.20,
    "right_shoulder_roll_joint": -0.20,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.78),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

# Robot lying flat on its back (supine, face up). Orientation is a -90° rotation
# about the +Y axis: quat (w,x,y,z) = (cos(-45°), 0, sin(-45°), 0) = (0.7071, 0, -0.7071, 0),
# settled with physics (scripts/generate_ground_poses.py) to the resting pose below.
# pos z = pelvis height of the robot at rest on the floor (no initial penetration).
# NOTE: the previous value (-0.866,-0.010,0.500,0.018) was only a 60° tilt — the robot
# was reclined, not lying down, with legs in the air and feet through the floor.
SUPINE_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.000, 0.000, 0.078),  # rest_z 0.073 + ~0.5cm: spawn just above the floor
    rot=(0.7071, 0.0000, -0.7071, 0.0000),
    joint_pos={
    ".*_elbow_joint": 1,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
    ".*_ankle_pitch_joint": -0.4,
  },
  joint_vel={".*": 0.0},
)

# Robot lying flat on its front (prone, face down). Orientation is a +90° rotation
# about the +Y axis, settled with physics to the resting pose below.
PRONE_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.000, 0.000, 0.139),  # rest_z 0.134 + ~0.5cm: spawn just above the floor
    rot=(0.7071, 0.0000, 0.7071, 0.0000),
    joint_pos={
    ".*_elbow_joint": 1,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
    ".*_ankle_pitch_joint": -0.4,
  },  joint_vel={".*": 0.0},
)

# Robot lying on its LEFT side. Orientation is a -90° roll about the +X axis
# (w,x,y,z)=(0.7071,-0.7071,0,0): body +Y (left) rotates down to the floor. The
# bottom (left) arm extends forward along the floor in line with the body; the
# top (right) arm rolls down and reaches its palm forward to plant on the floor
# in front of the chest (a push-up prop, so the hand is usable to push up).
# pos.z is rest height + ~0.5cm clearance (real-collision bisection,
# scripts/diag_contact_geoms.py): starts just above the floor, no penetration.
SIDE_LEFT_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.000, 0.000, 0.193),  # rest_z 0.188 + ~0.5cm: spawn just above the floor
    rot=(0.7071, -0.7071, 0.0000, 0.0000),
    joint_pos={
        ".*_ankle_pitch_joint": -0.2,
        ".*_hip_pitch_joint": -0.3,
        ".*_knee_joint": 0.4,
        # bottom (left) arm: extend forward along the floor in line with the body
        "left_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0,
        "left_elbow_joint": 0.5,
        # top (right) arm: roll swings it down toward the floor, pitch+elbow reach
        # the palm forward to plant it in front of the chest (push-up prop)
        "right_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_elbow_joint": 0.5,
    },
    joint_vel={".*": 0.0},
)

# Robot lying flat on its RIGHT side. Mirror of SIDE_LEFT: a +90° roll about +X
# (w,x,y,z)=(0.7071,0.7071,0,0). Same (symmetric) joints.
SIDE_RIGHT_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.000, 0.000, 0.193),  # rest_z 0.188 + ~0.5cm: spawn just above the floor
    rot=(0.7071, 0.7071, 0.0000, 0.0000),
    joint_pos={
        ".*_ankle_pitch_joint": -0.2,
        ".*_hip_pitch_joint": -0.3,
        ".*_knee_joint": 0.4,
        # bottom (right) arm: extend forward along the floor in line with the body
        "left_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0,
        "left_elbow_joint": 0.5,
        # top (left) arm: roll swings it down toward the floor, pitch+elbow reach
        # the palm forward to plant it in front of the chest (push-up prop)
        "right_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": 0,
        "right_elbow_joint": 0.5,
    },
    joint_vel={".*": 0.0},
)

# Robot seated on the ground: torso near-upright with a slight backward lean
# (-14° about +Y), deep hip flexion + bent knees so it rests on a stable tripod —
# buttocks (pelvis) + both feet flat on the floor (the natural pre-stand sitting
# pose). pos.z is rest height + ~0.5cm clearance (real-collision bisection,
# scripts/diag_contact_geoms.py): starts seated on the floor, no penetration.
SEATED_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.000, 0.000, 0.155),  # rest_z 0.150 + ~0.5cm: starts seated ON the floor
    rot=(0.9925, 0.0000, -0.1219, 0.0000),
    joint_pos={
        # deep hip+knee flexion with a slight backward lean so the robot rests on
        # a stable tripod: buttocks (pelvis) + both feet flat on the floor.
        ".*_hip_pitch_joint": -2.24,
        ".*_knee_joint": 1.87,
        ".*_ankle_pitch_joint": -0.45,
        ".*_shoulder_pitch_joint": 0.3,
        ".*_elbow_joint": 0.5,
        "left_shoulder_roll_joint": 0.25,
        "right_shoulder_roll_joint": -0.25,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3. Stand-up-critical support contacts also get condim=3
# with higher sliding friction so the robot can push off the floor without
# excessive slipping on hands, elbows, pelvis, wrists, torso, and head.
# The head (head_collision sphere) is condim=3 + friction so it can act as a
# support / roll-over contact, not just an impact-detection point.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    r"^(left|right)_foot[1-7]_collision$": 3,
    r"^(left|right)_hand_collision$": 3,
    r"^(left|right)_wrist_collision$": 3,
    r"^(left|right)_elbow_yaw_collision$": 3,
    r"^pelvis_collision$": 3,
    r"^torso_collision$": 3,
    r"^head_collision$": 3,
    ".*_collision": 1,
  },
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={
    r"^(left|right)_foot[1-7]_collision$": (0.6,),
    r"^(left|right)_hand_collision$": (1.5, 0.01, 0.001),
    r"^(left|right)_wrist_collision$": (1.5, 0.01, 0.001),
    r"^(left|right)_elbow_yaw_collision$": (1.0, 0.005, 0.0001),
    r"^pelvis_collision$": (0.8, 0.005, 0.0001),
    r"^torso_collision$": (0.6, 0.005, 0.0001),
    r"^head_collision$": (0.8, 0.005, 0.0001),
  },
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={
    r"^(left|right)_foot[1-7]_collision$": 3,
    r"^(left|right)_hand_collision$": 3,
    r"^(left|right)_wrist_collision$": 3,
    r"^(left|right)_elbow_yaw_collision$": 3,
    r"^pelvis_collision$": 3,
    r"^torso_collision$": 3,
    r"^head_collision$": 3,
    ".*_collision": 1,
  },
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={
    r"^(left|right)_foot[1-7]_collision$": (0.6,),
    r"^(left|right)_hand_collision$": (1.5, 0.01, 0.001),
    r"^(left|right)_wrist_collision$": (1.5, 0.01, 0.001),
    r"^(left|right)_elbow_yaw_collision$": (1.0, 0.005, 0.0001),
    r"^pelvis_collision$": (0.8, 0.005, 0.0001),
    r"^torso_collision$": (0.6, 0.005, 0.0001),
    r"^head_collision$": (0.8, 0.005, 0.0001),
  },
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
    G1_ACTUATOR_HAND,
  ),
  soft_joint_pos_limit_factor=0.9,
)

# HoST get-up articulation: same motors as G1_ARTICULATION but with the 7520-14
# and 7520-22 groups split by function so hip Kp=200 and knee Kp=275 (see the
# HoST PD override block above). Covers all 29 DOF exactly once.
G1_ARTICULATION_HOST = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_HOST_ARM,
    G1_ACTUATOR_HOST_HIP_PITCH_YAW,
    G1_ACTUATOR_HOST_WAIST_YAW,
    G1_ACTUATOR_HOST_HIP_ROLL,
    G1_ACTUATOR_HOST_KNEE,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
    G1_ACTUATOR_HAND,
  ),
  soft_joint_pos_limit_factor=0.98,
)


def get_g1_robot_cfg() -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


def get_g1_supine_robot_cfg() -> EntityCfg:
  """Get G1 robot configuration starting in supine (lying on back) position."""
  return EntityCfg(
    init_state=SUPINE_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


def get_g1_supine_robot_cfg_host() -> EntityCfg:
  """G1 supine config with HoST get-up PD gains (hip Kp=200, knee Kp=275).

  Identical to get_g1_supine_robot_cfg() except it uses G1_ARTICULATION_HOST, so
  only the get-up task picks up the HoST stiffness override. All other joints keep
  their armature-derived stiffness and damping (Kd) is unchanged everywhere.
  """
  return EntityCfg(
    init_state=SUPINE_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION_HOST,
  )


def get_g1_prone_robot_cfg() -> EntityCfg:
  """Get G1 robot configuration starting in prone (lying face down) position."""
  return EntityCfg(
    init_state=PRONE_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


def _wrap_with_delay(
  base_cfg: BuiltinPositionActuatorCfg,
  delay_min_lag: int = 0,
  delay_max_lag: int = 3,
) -> DelayedActuatorCfg:
  """Wrap a BuiltinPositionActuatorCfg with per-env action delay."""
  if DelayedActuatorCfg is None:
    raise ImportError("DelayedActuatorCfg is unavailable in this mjlab version")
  return DelayedActuatorCfg(
    base_cfg=base_cfg,
    delay_target="position",
    delay_min_lag=delay_min_lag,
    delay_max_lag=delay_max_lag,
    delay_per_env_phase=True,
  )

def get_g1_supine_robot_cfg_with_delay() -> EntityCfg:
  """G1 supine config with DelayedActuator wrappers for action-delay DR.

  Use this instead of get_g1_supine_robot_cfg() when _DR_ACTION_DELAY_ENABLE
  is True in env_cfgs.py. The delay range is set via dr.sync_actuator_delays
  at reset, so delay_min/max_lag here only define the buffer capacity.
  """
  articulation_delayed = EntityArticulationInfoCfg(
    actuators=(
      _wrap_with_delay(G1_ACTUATOR_5020),
      _wrap_with_delay(G1_ACTUATOR_7520_14),
      _wrap_with_delay(G1_ACTUATOR_7520_22),
      _wrap_with_delay(G1_ACTUATOR_4010),
      _wrap_with_delay(G1_ACTUATOR_WAIST),
      _wrap_with_delay(G1_ACTUATOR_ANKLE),
      _wrap_with_delay(G1_ACTUATOR_HAND),
    ),
    soft_joint_pos_limit_factor=0.9,
  )
  return EntityCfg(
    init_state=SUPINE_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=articulation_delayed,
  )


G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())

  viewer.launch(robot.spec.compile())
