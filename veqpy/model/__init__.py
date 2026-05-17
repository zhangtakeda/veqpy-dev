"""
Module: model.__init__

Role:
- Export public model-layer types and package-level entrypoints.

Public API:
- Boundary
- Grid
- Profile
- Geometry
- Equilibrium
- Reactive
- Serial

Notes:
- This module only provides package-level exports.
- Does not own packed runtime state, solver policy, or backend selection.
"""

from __future__ import annotations

from veqpy.base import Reactive, Serial
from veqpy.model.boundary import Boundary
from veqpy.model.equilibrium import Equilibrium
from veqpy.model.geometry import Geometry
from veqpy.model.geqdsk import Geqdsk
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile

__all__ = [
    Equilibrium,
    "Geometry",
    Grid,
    "Geqdsk",
    "Boundary",
    Profile,
    "Reactive",
    "Serial",
]
