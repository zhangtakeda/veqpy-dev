"""Runtime workspace memory ownership and hot-path array views."""

from veqpy.workspace.runtime import (
    BackendState,
    FieldRuntimeState,
    GeometryWorkspace,
    OperatorWorkspace,
    ProfileWorkspace,
    ResidualWorkspace,
    RuntimeAllocationBundle,
    RuntimeLayout,
    RuntimeWorkspace,
    SourceAuxState,
    SourceConstState,
    SourceRuntimeState,
    SourceWorkspace,
    SourceWorkState,
    allocate_runtime_state,
    build_runtime_layout_view,
)

__all__ = [
    "BackendState",
    "FieldRuntimeState",
    "GeometryWorkspace",
    "ProfileWorkspace",
    "ResidualWorkspace",
    "OperatorWorkspace",
    "RuntimeAllocationBundle",
    "RuntimeLayout",
    "RuntimeWorkspace",
    "SourceWorkspace",
    "SourceAuxState",
    "SourceConstState",
    "SourceRuntimeState",
    "SourceWorkState",
    "allocate_runtime_state",
    "build_runtime_layout_view",
]
