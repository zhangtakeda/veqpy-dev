"""Backend binding state for engine/layout construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veqpy.operator.residual_binding import ResidualBindingLayout
    from veqpy.workspace.geometry_workspace import GeometryWorkspace
    from veqpy.workspace.grid_workspace import GridWorkspace
    from veqpy.workspace.profile_workspace import ProfileWorkspace
    from veqpy.workspace.residual_workspace import ResidualWorkspace
    from veqpy.workspace.source_workspace import SourceWorkspace


@dataclass(slots=True)
class BackendState:
    """Arrays-only backend ABI aggregate."""

    static_layout: GridWorkspace
    residual_binding_layout: ResidualBindingLayout
    profile_workspace: ProfileWorkspace
    geometry_workspace: GeometryWorkspace
    source_workspace: SourceWorkspace
    residual_workspace: ResidualWorkspace
