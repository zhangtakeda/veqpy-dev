"""
Module: model.__init__

Role:
- 负责导出 model 层的公开类型与包级入口.

Public API:
- Grid
- Profile
- Geometry
- Equilibrium
- Reactive
- Serial

Notes:
- 这里只做包级导出.
- 不负责 packed runtime ownership, solver policy, 或 backend 选择.
"""

from veqpy.model.equilibrium import Equilibrium
from veqpy.model.geometry import Geometry
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile
from veqpy.model.reactive import Reactive
from veqpy.model.serial import Serial

__all__ = [
    "Equilibrium",
    "Geometry",
    "Grid",
    "Profile",
    "Reactive",
    "Serial",
]
