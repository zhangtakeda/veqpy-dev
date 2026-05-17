"""Geometry-stage runtime memory ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from veqpy.model.profile import Profile


@dataclass(init=False, slots=True)
class GeometryWorkspace:
    """Geometry stage memory owner.

    ``surface_fields`` shape: ``(9, Nr, Nt)`` with rows:
    sin_tb, R, R_t, Z_t, J, JdivR, grtdivJR_t, gttdivJR, gttdivJR_r.

    ``radial_fields`` shape: ``(5, Nr)`` with rows:
    S_r, V_r, Kn, Kn_r, Ln_r.

    ``h/v/k_fields`` are borrowed model profile field arrays consumed only by the
    geometry stage. The owning ``Profile`` objects remain in the operator model state;
    this workspace owns the runtime binding responsibility.
    """

    surface_fields: np.ndarray
    radial_fields: np.ndarray
    h_fields: np.ndarray
    v_fields: np.ndarray
    k_fields: np.ndarray

    def __init__(self, *, nr: int, nt: int) -> None:
        """Allocate geometry-stage runtime memory."""

        self.surface_fields = np.empty((9, nr, nt), dtype=np.float64)
        self.radial_fields = np.empty((5, nr), dtype=np.float64)
        self.h_fields = np.empty((0, nr), dtype=np.float64)
        self.v_fields = np.empty((0, nr), dtype=np.float64)
        self.k_fields = np.empty((0, nr), dtype=np.float64)

    def bind_shape_profile_views(
        self, *, h_profile: Profile, v_profile: Profile, k_profile: Profile
    ) -> None:
        """Bind borrowed shape profile fields used by geometry kernels."""

        self.h_fields = h_profile.u_fields
        self.v_fields = v_profile.u_fields
        self.k_fields = k_profile.u_fields
