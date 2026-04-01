"""
Module: operator.execution_helpers

Role:
- 承载 operator 到 Equilibrium snapshot 的共享 helper.
- 避免 operator.py 混入 packed-state 解码与 profile 拷贝细节.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from veqpy.model import Equilibrium, Profile
from veqpy.operator.codec import decode_packed_blocks

if TYPE_CHECKING:
    from veqpy.model.grid import Grid
    from veqpy.operator.operator_case import OperatorCase


def snapshot_equilibrium_from_runtime(
    x: np.ndarray,
    *,
    case: "OperatorCase",
    grid: "Grid",
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    profile_names: tuple[str, ...],
    shape_profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    profiles_by_name: dict[str, Profile],
    psin: np.ndarray,
    FFn_psin: np.ndarray,
    Pn_psin: np.ndarray,
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
    alpha1: float,
    alpha2: float,
) -> Equilibrium:
    coeff_blocks = decode_packed_blocks(x, profile_L, coeff_index, profile_names=profile_names)
    active_profiles = snapshot_equilibrium_profiles(
        coeff_blocks,
        shape_profile_names=shape_profile_names,
        profile_index=profile_index,
        profiles_by_name=profiles_by_name,
    )
    return Equilibrium(
        R0=case.R0,
        Z0=case.Z0,
        B0=case.B0,
        a=case.a,
        grid=grid,
        active_profiles=active_profiles,
        psin=psin.copy(),
        FFn_psin=FFn_psin.copy(),
        Pn_psin=Pn_psin.copy(),
        psin_r=psin_r.copy(),
        psin_rr=psin_rr.copy(),
        alpha1=float(alpha1),
        alpha2=float(alpha2),
    )


def snapshot_equilibrium_profiles(
    coeff_blocks: tuple[np.ndarray | None, ...],
    *,
    shape_profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    profiles_by_name: dict[str, Profile],
) -> dict[str, Profile]:
    return {
        name: snapshot_profile(profiles_by_name[name], coeff_blocks[profile_index[name]])
        for name in shape_profile_names
    }


def snapshot_profile(profile: Profile, coeff_block: np.ndarray | None) -> Profile:
    copied = profile.copy()
    copied.coeff = None if coeff_block is None else coeff_block.copy()
    return copied
