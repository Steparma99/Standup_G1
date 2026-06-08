"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Core dependencies (backend-agnostic)
INSTALL_REQUIRES = [
    "mjlab>=1.4.0,<1.5",
]

# Optional extras:
#   pip install -e ".[cpu]"   → AMD / CPU-only (standard mujoco via mjlab)
#   pip install -e ".[cuda]"  → NVIDIA GPU (adds mujoco-warp)
EXTRAS_REQUIRE = {
    "cpu": [],                       # mjlab already pulls standard mujoco
    "cuda": ["mujoco-warp>=3.8.0.3,<3.9"],  # Match mjlab 1.4.x expectations
}

setup(
    name="unitree_rl_mjlab",
    packages=find_packages(include=["src", "src.*"]),
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
