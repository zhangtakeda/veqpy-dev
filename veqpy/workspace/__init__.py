"""Runtime workspace memory ownership and hot-path array views."""

from veqpy.workspace.runtime import (
    BackendState,
    FieldRuntimeState,
    GeometryWorkspace,
    OperatorWorkspace,
    ProfileWorkspace,
    ResidualWorkspace,
    RuntimeAllocationBundle,
    SourceAuxState,
    SourceConstState,
    SourceRuntimeState,
    SourceWorkspace,
    SourceWorkState,
    allocate_runtime_state,
)

__all__ = [
    "BackendState",
    "FieldRuntimeState",
    "GeometryWorkspace",
    "ProfileWorkspace",
    "ResidualWorkspace",
    "OperatorWorkspace",
    "RuntimeAllocationBundle",
    "SourceWorkspace",
    "SourceAuxState",
    "SourceConstState",
    "SourceRuntimeState",
    "SourceWorkState",
    "allocate_runtime_state",
]
