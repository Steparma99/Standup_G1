"""Robot asset exports.

This repository only vendors a subset of the upstream robot definitions.
Import optional robots lazily so task registration for G1 does not fail when
unrelated robot packages are absent.
"""

from importlib import import_module

from .unitree_g1.g1_constants import (
    G1_ACTION_SCALE,
    get_g1_prone_robot_cfg,
    get_g1_robot_cfg,
    get_g1_supine_robot_cfg,
    get_g1_supine_robot_cfg_host,
)


def _export(module_name: str, names: list[str]) -> None:
    try:
        module = import_module(module_name, package=__name__)
    except (ImportError, ModuleNotFoundError):
        return
    for name in names:
        globals()[name] = getattr(module, name)


_export(".unitree_g1.g1_23dof_constants", [
    "G1_23DOF_ACTION_SCALE",
    "get_g1_23dof_robot_cfg",
])

_export(".unitree_go2.go2_constants", [
    "get_go2_robot_cfg",
])

_export(".unitree_a2.a2_constants", [
    "get_a2_robot_cfg",
])

_export(".unitree_as2.as2_constants", [
    "get_as2_robot_cfg",
])

_export(".unitree_r1.r1_constants", [
    "R1_ACTION_SCALE",
    "get_r1_robot_cfg",
])

_export(".unitree_h1_2.h1_2_constants", [
    "H1_2_ACTION_SCALE",
    "get_h1_2_robot_cfg",
])

_export(".unitree_h2.h2_constants", [
    "H2_ACTION_SCALE",
    "get_h2_robot_cfg",
])
