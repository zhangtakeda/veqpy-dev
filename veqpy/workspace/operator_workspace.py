"""Top-level operator workspace aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from veqpy.workspace.geometry_workspace import GeometryWorkspace
from veqpy.workspace.profile_workspace import ProfileWorkspace
from veqpy.workspace.residual_workspace import ResidualWorkspace
from veqpy.workspace.source_workspace import SourceWorkspace

if TYPE_CHECKING:
    from veqpy.model.profile import Profile
    from veqpy.operator.residual_binding import ResidualBindingLayout
    from veqpy.workspace.grid_workspace import GridWorkspace


@dataclass(slots=True)
class OperatorWorkspace:
    """Aggregate runtime memory owner for operator evaluation."""

    profile: ProfileWorkspace
    geometry: GeometryWorkspace
    source: SourceWorkspace
    residual: ResidualWorkspace

    def bind_profile_views(
        self,
        *,
        h_profile: Profile,
        v_profile: Profile,
        k_profile: Profile,
        F_profile: Profile,
        psin_profile: Profile,
    ) -> None:
        """Bind borrowed model profile arrays to their consuming stage workspaces."""

        self.geometry.bind_shape_profile_views(
            h_profile=h_profile,
            v_profile=v_profile,
            k_profile=k_profile,
        )
        self.source.bind_profile_views(
            F_profile=F_profile,
            psin_profile=psin_profile,
        )


@dataclass(slots=True)
class BackendState:
    """Arrays-only backend ABI aggregate."""

    static_layout: GridWorkspace
    residual_binding_layout: ResidualBindingLayout
    workspace: OperatorWorkspace


@dataclass(slots=True)
class RuntimeAllocationBundle:
    """Operator runtime allocation result."""

    profiles_by_name: dict[str, Profile]
    workspace: OperatorWorkspace
    profile_workspace: ProfileWorkspace
    geometry_workspace: GeometryWorkspace
    source_workspace: SourceWorkspace
    residual_workspace: ResidualWorkspace
