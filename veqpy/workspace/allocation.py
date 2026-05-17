"""Operator workspace allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from veqpy.workspace.geometry_workspace import GeometryWorkspace
from veqpy.workspace.profile_workspace import ProfileWorkspace
from veqpy.workspace.residual_workspace import ResidualWorkspace
from veqpy.workspace.source_workspace import SourceWorkspace

if TYPE_CHECKING:
    from veqpy.engine.backend_abi import SourceExecutionABI
    from veqpy.model.profile import Profile
    from veqpy.workspace.grid_workspace import GridWorkspace


@dataclass(slots=True)
class RuntimeAllocationBundle:
    """Operator runtime allocation result."""

    profiles_by_name: dict[str, Profile]
    profile_workspace: ProfileWorkspace
    geometry_workspace: GeometryWorkspace
    source_workspace: SourceWorkspace
    residual_workspace: ResidualWorkspace


def allocate_runtime_state(
    *,
    static_layout: GridWorkspace,
    source_execution: SourceExecutionABI,
    profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    active_profile_ids: np.ndarray,
    profile_L: np.ndarray,
    x_size: int,
    make_profile: Callable[[str], Profile],
) -> RuntimeAllocationBundle:
    """Build operator runtime state through stage workspace constructors."""

    nr = static_layout.Nr
    nt = static_layout.Nt
    m_max = static_layout.M_max

    profiles_by_name = {name: make_profile(name) for name in profile_names}

    profile_workspace = ProfileWorkspace(
        nr=nr,
        m_max=m_max,
        profile_names=profile_names,
        profile_index=profile_index,
        active_profile_ids=active_profile_ids,
        profile_L=profile_L,
    )
    geometry_workspace = GeometryWorkspace(
        nr=nr,
        nt=nt,
    )
    source_workspace = SourceWorkspace(
        nr=nr,
        nt=nt,
        source_execution=source_execution,
    )
    residual_workspace = ResidualWorkspace(
        nr=nr,
        nt=nt,
        x_size=x_size,
        radial_weights=np.asarray(static_layout.weights, dtype=np.float64),
    )
    return RuntimeAllocationBundle(
        profiles_by_name=profiles_by_name,
        profile_workspace=profile_workspace,
        geometry_workspace=geometry_workspace,
        source_workspace=source_workspace,
        residual_workspace=residual_workspace,
    )
