"""Get-up task observation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

_H_STANDING = 0.65
_H_RISING = 0.40
_H_SUPPORT = 0.20


def _flatten_contact_found(sensor: ContactSensor, env: ManagerBasedRlEnv) -> torch.Tensor:
    found = sensor.data.found
    if found is None:
        return torch.zeros((env.num_envs, 1), device=env.device)
    if found.ndim == 1:
        return (found > 0).float().unsqueeze(1)
    return (found > 0).float()


def _flatten_contact_force(sensor: ContactSensor, env: ManagerBasedRlEnv) -> torch.Tensor:
    force = sensor.data.force
    if force is None:
        return torch.zeros((env.num_envs, 3), device=env.device)
    if force.ndim == 2:
        force_flat = force
    else:
        force_flat = force.flatten(start_dim=1)
    return torch.sign(force_flat) * torch.log1p(torch.abs(force_flat))


def base_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return pelvis/root height [B, 1]."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2:3]


def body_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Backward-compatible alias for pelvis height [B, 1]."""
    return base_height(env, asset_cfg=asset_cfg)


def head_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return head-height proxy [B, 1].

    G1 does not expose a separate head body state in the current mjlab asset
    wrapper, so we use the tracked torso-link height as the stable proxy already
    used elsewhere in this task.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]


def feet_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Binary ground-contact flag per foot [B, N_feet]."""
    sensor = env.scene[sensor_name]
    return _flatten_contact_found(sensor, env)


def pd_tracking_error(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Per-joint PD tracking error (commanded target - measured position) [B, A]."""
    term = env.action_manager.get_term(action_name)
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, term.target_ids]
    return term.applied_target - q


def applied_joint_target(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Applied low-pass-filtered joint target [B, A]."""
    term = env.action_manager.get_term(action_name)
    return term.applied_target


def contact_state(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Flattened binary contact state for a named sensor [B, N]."""
    sensor: ContactSensor = env.scene[sensor_name]
    return _flatten_contact_found(sensor, env)


def contact_force(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Flattened contact force for a named sensor [B, N*3] with log compression."""
    sensor: ContactSensor = env.scene[sensor_name]
    return _flatten_contact_force(sensor, env)


def whole_body_com(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Whole-body center-of-mass in world frame [B, 3]."""
    asset: Entity = env.scene[asset_cfg.name]
    root_id = asset.data.indexing.root_body_id
    return asset.data.data.subtree_com[:, root_id, :3]


def feet_positions(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Flattened foot site positions in world frame [B, N_sites*3]."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.site_pos_w[:, asset_cfg.site_ids, :3].flatten(start_dim=1)


def stage_gates(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return four stage indicators [B, 4] for get-up progress."""
    asset: Entity = env.scene[asset_cfg.name]
    h = asset.data.root_link_pos_w[:, 2]
    stage0 = (h < _H_SUPPORT).float()
    stage1 = ((h >= _H_SUPPORT) & (h < _H_RISING)).float()
    stage2 = ((h >= _H_RISING) & (h < _H_STANDING)).float()
    stage3 = (h >= _H_STANDING).float()
    return torch.stack((stage0, stage1, stage2, stage3), dim=1)


def foot_friction(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Mean tangential foot friction coefficient [B, 1]."""
    geom_ids = asset_cfg.geom_ids
    if geom_ids is None or len(geom_ids) == 0:
        return torch.zeros((env.num_envs, 1), device=env.device)

    friction = env.sim.model.geom_friction
    if friction.ndim == 3:
        coeff = friction[:, geom_ids, 0]
    else:
        coeff = friction[geom_ids, 0].unsqueeze(0).expand(env.num_envs, -1)
    return coeff.mean(dim=1, keepdim=True)


_G_MS2 = 9.81  # m/s² — used to normalise accelerometer output to ~[-1, 1]


def imu_lin_acc(
    env: ManagerBasedRlEnv,
    sensor_name: str = "robot/imu_lin_acc",
) -> torch.Tensor:
    """IMU linear acceleration (specific force) normalised by g [B, 3].

    MuJoCo's `accelerometer` sensor measures **specific force** in the sensor
    (body) frame:  f_specific = a_kinematic + R_body^T · g_world.
    This is exactly what a real MEMS IMU outputs — no post-processing needed.

    Key properties:
    - At rest, upright:  f ≈ [0, 0, +g] in body frame → normalised ≈ [0, 0, +1].
    - At rest, supine:   f ≈ [-g, 0, 0]  → normalised ≈ [-1, 0, 0].
    - The static component equals -proj_grav_b * g (redundant with projected_gravity_b).
    - The **dynamic** component (acceleration during push-off, landing, etc.) is
      new information not captured by the gyro or projected gravity alone.

    Why normalise by g:
    - The static component is bounded by ±g; dynamic peaks reach ±3–5 g in practice.
    - Dividing by g centres the static range on [-1, 1] and keeps dynamic values
      in a consistent, moderately bounded range for the network.

    Deployability: the real G1 IMU (pelvis) outputs specific force in the same
    frame and convention.  Add Gaussian noise (≥0.1 per axis) and optionally
    a slow bias drift via a DR event to close the sim-to-real gap.

    Note: there is also an `imu_in_torso` site in g1.xml but NO sensor is
    defined for it in the `<sensor>` block — only the pelvis IMU is functional.
    """
    from mjlab.envs.mdp.observations import builtin_sensor
    raw = builtin_sensor(env, sensor_name)  # [B, 3]  m/s²
    return raw / _G_MS2  # [B, 3]  dimensionless specific-force / g


class ImuLinAcc:
    """Stateful accelerometer model for the ACTOR observation.

    Wraps the clean ``imu_lin_acc`` ground-truth reading (specific force / g) with
    the deterministic, on-board parts of a real MEMS accelerometer so the policy
    learns on a deployable signal:

        out = bias + EMA_lp( specific_force / g )

    Pipeline placement (see ObservationTermCfg docstring): this function runs first,
    then the manager applies, in order, ``noise`` (white sensor noise), ``clip``
    (saturation), ``delay`` (latency) and ``history``. So the full effective model is

        history( delay( clip( bias + LP(true/g) + white_noise ) ) )

    which mirrors a hardware IMU: anti-alias low-pass on the true signal, a slowly
    varying bias, additive white noise, ADC saturation and bus/read latency.

    Statefulness (per-env, lives only on the actor's instance):
      - ``_ema``  : low-pass filter memory, re-seeded with the raw reading on reset.
      - ``_bias`` : per-axis offset, resampled every episode reset to model drift.

    The CRITIC must NOT use this class: it should observe the clean ground-truth
    accelerometer (plain :func:`imu_lin_acc`) with no bias/LP/noise. The env config
    wires that override in ``_get_critic_obs_terms``.

    Config via ``ObservationTermCfg.params`` (all optional):
      sensor_name : accelerometer sensor (default "robot/imu_lin_acc").
      lp_alpha    : EMA coefficient in (0, 1]; lower = more smoothing (default 0.3).
      bias_std    : per-axis bias 1-sigma in g (default 0.02).
      bias_clip   : hard clip on the sampled bias in g (default 0.05).
    """

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        self._env = env
        self._device = env.device
        self._num_envs = env.num_envs
        p = cfg.params or {}
        self._sensor_name = p.get("sensor_name", "robot/imu_lin_acc")
        self._alpha = float(p.get("lp_alpha", 0.3))
        self._bias_std = float(p.get("bias_std", 0.02))
        self._bias_clip = float(p.get("bias_clip", 0.05))
        self._ema = torch.zeros(self._num_envs, 3, device=self._device)
        self._bias = torch.zeros(self._num_envs, 3, device=self._device)
        # Per-env flag: re-seed the EMA with the first post-reset reading so the
        # filter starts on the true value instead of decaying up from zero.
        self._needs_seed = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._resample_bias(slice(None))

    def _resample_bias(self, env_ids) -> None:
        if isinstance(env_ids, slice):
            n = self._num_envs
        else:
            n = len(env_ids)
        b = torch.randn(n, 3, device=self._device) * self._bias_std
        b = b.clamp(-self._bias_clip, self._bias_clip)
        self._bias[env_ids] = b

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        # Re-seed the LP filter on the next call; resample bias to model drift.
        self._needs_seed[env_ids] = True
        self._resample_bias(env_ids)

    def __call__(self, env: "ManagerBasedRlEnv", **kwargs) -> torch.Tensor:
        raw = imu_lin_acc(env, sensor_name=self._sensor_name)  # [B, 3] clean, in g
        # Seed EMA with the raw reading for freshly-reset envs.
        if bool(self._needs_seed.any()):
            seed = self._needs_seed
            self._ema = self._ema.clone()
            self._ema[seed] = raw[seed]
            self._needs_seed = self._needs_seed.clone()
            self._needs_seed[seed] = False
        self._ema = self._alpha * raw + (1.0 - self._alpha) * self._ema
        return self._ema + self._bias
