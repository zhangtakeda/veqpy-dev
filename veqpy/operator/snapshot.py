"""
Module: operator.snapshot

Role:
- Materialize immutable model snapshots from refreshed operator runtime arrays.

Notes:
- Snapshot helpers copy runtime-owned arrays before returning model objects.
- They do not run stages, allocate runtime memory, or bind executable layouts.
"""

from __future__ import annotations

import numpy as np

from veqpy.model.equilibrium import Equilibrium
from veqpy.model.geometry import Geometry
from veqpy.model.grid import Grid
from veqpy.operator.operator_case import OperatorCase


def snapshot_equilibrium_from_runtime(
    *,
    case: OperatorCase,
    grid: Grid,
    h_fields: np.ndarray,
    v_fields: np.ndarray,
    k_fields: np.ndarray,
    c_family_fields: np.ndarray,
    s_family_fields: np.ndarray,
    c_active_order: int,
    s_active_order: int,
    psin: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
    alpha1: float,
    alpha2: float,
) -> Equilibrium:
    """Materialize an Equilibrium snapshot from current Operator runtime arrays."""

    geometry = snapshot_geometry_from_runtime(
        case=case,
        grid=grid,
        h_fields=h_fields,
        v_fields=v_fields,
        k_fields=k_fields,
        c_family_fields=c_family_fields,
        s_family_fields=s_family_fields,
        c_active_order=c_active_order,
        s_active_order=s_active_order,
    )
    return Equilibrium(
        R0=case.R0,
        Z0=case.Z0,
        B0=case.B0,
        a=case.a,
        grid=grid,
        geometry=geometry,
        psin=psin.copy(),
        FFn_psin=np.asarray(FFn_psin, dtype=np.float64).copy(),
        Pn_psin=Pn_psin.copy(),
        psin_r=psin_r.copy(),
        psin_rr=psin_rr.copy(),
        alpha1=float(alpha1),
        alpha2=float(alpha2),
    )


def snapshot_geometry_from_runtime(
    *,
    case: OperatorCase,
    grid: Grid,
    h_fields: np.ndarray,
    v_fields: np.ndarray,
    k_fields: np.ndarray,
    c_family_fields: np.ndarray,
    s_family_fields: np.ndarray,
    c_active_order: int,
    s_active_order: int,
) -> Geometry:
    """Materialize Geometry directly from refreshed operator profile fields."""

    geometry = Geometry(grid=grid)
    geometry.update(
        case.a,
        case.R0,
        case.Z0,
        grid,
        np.asarray(h_fields, dtype=np.float64),
        np.asarray(v_fields, dtype=np.float64),
        np.asarray(k_fields, dtype=np.float64),
        np.asarray(c_family_fields, dtype=np.float64),
        np.asarray(s_family_fields, dtype=np.float64),
        c_active_order=int(c_active_order),
        s_active_order=int(s_active_order),
    )
    return geometry
