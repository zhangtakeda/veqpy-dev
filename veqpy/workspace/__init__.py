"""Public workspace interfaces used outside ``veqpy.workspace``.

Private block packers and stage-local workspace internals stay in their owning
modules. The package root exposes only the construction entrypoint and aggregate
contracts consumed by operator/layout/engine wiring.
"""

from __future__ import annotations

from veqpy.workspace.allocation import allocate_runtime_state
from veqpy.workspace.grid_workspace import GridWorkspace
from veqpy.workspace.operator_workspace import (
    BackendState,
    OperatorWorkspace,
    RuntimeAllocationBundle,
)

__all__ = [
    "BackendState",
    "GridWorkspace",
    "OperatorWorkspace",
    "RuntimeAllocationBundle",
    "allocate_runtime_state",
]
