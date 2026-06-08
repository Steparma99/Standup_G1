"""Unitree G1 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.delayed_actuator import DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

G1_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
)
assert G1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  assets_dir = G1_XML.parent / "assets"
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

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
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
    pos=(0.000, 0.000, 0.072),
    rot=(0.7247, 0.0000, -0.6891, 0.0000),
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
   pos=(0.000, 0.000, 0.099),
    rot=(0.6919, 0.0000, 0.7220, 0.0000),
    joint_pos={
    ".*_elbow_joint": 1,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
    ".*_ankle_pitch_joint": -0.4,
  },  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3. Stand-up-critical support contacts also get condim=3
# with higher sliding friction so the robot can push off the floor without
# excessive slipping on hands, elbows, pelvis, wrists, and torso.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    r"^(left|right)_foot[1-7]_collision$": 3,
    r"^(left|right)_hand_collision$": 3,
    r"^(left|right)_wrist_collision$": 3,
    r"^(left|right)_elbow_yaw_collision$": 3,
    r"^pelvis_collision$": 3,
    r"^torso_collision$": 3,
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
  ),
  soft_joint_pos_limit_factor=0.9,
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
  return DelayedActuatorCfg(
    base_cfg=base_cfg,
    delay_target="position",
    delay_min_lag=delay_min_lag,
    delay_max_lag=delay_max_lag,
    delay_per_env_phase=True,
  )


G1_ARTICULATION_DELAYED = EntityArticulationInfoCfg(
  actuators=(
    _wrap_with_delay(G1_ACTUATOR_5020),
    _wrap_with_delay(G1_ACTUATOR_7520_14),
    _wrap_with_delay(G1_ACTUATOR_7520_22),
    _wrap_with_delay(G1_ACTUATOR_4010),
    _wrap_with_delay(G1_ACTUATOR_WAIST),
    _wrap_with_delay(G1_ACTUATOR_ANKLE),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_supine_robot_cfg_with_delay() -> EntityCfg:
  """G1 supine config with DelayedActuator wrappers for action-delay DR.

  Use this instead of get_g1_supine_robot_cfg() when _DR_ACTION_DELAY_ENABLE
  is True in env_cfgs.py. The delay range is set via dr.sync_actuator_delays
  at reset, so delay_min/max_lag here only define the buffer capacity.
  """
  return EntityCfg(
    init_state=SUPINE_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION_DELAYED,
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
