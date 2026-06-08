"""Task registry bootstrap.

This repository is centered on the custom G1 get-up task. Upstream velocity
and tracking task trees may reference robot assets that are not vendored here,
so keep them optional during package import.
"""

import importlib


def _import_optional(module_name: str) -> None:
    try:
        importlib.import_module(module_name, package=__name__)
    except (ImportError, ModuleNotFoundError):
        return


importlib.import_module(".getup", package=__name__)
_import_optional(".velocity")
_import_optional(".tracking")

