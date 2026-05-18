"""Public workspace interfaces used outside ``veqpy.workspace``.

Private block packers and stage-local workspace internals stay in their owning
modules. The package root exposes only the construction entrypoint and static grid
workspace consumed by operator/layout/engine wiring.
"""

from __future__ import annotations

from veqpy.workspace.allocation import allocate_runtime_state
from veqpy.workspace.grid_workspace import GridWorkspace

__all__ = [
    "GridWorkspace",
    "allocate_runtime_state",
]
