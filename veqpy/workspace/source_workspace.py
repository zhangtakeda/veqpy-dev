"""Source-stage runtime memory ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from veqpy.model.profile import Profile


@dataclass(slots=True)
class SourceWorkspace:
    """Source stage memory owner.

    The source stage owns route/interpolation cache arrays, materialization scratch,
    source-produced outputs, source scale factors, and borrowed F/psin profile inputs.
    """

    cache_key: tuple[str, str, int, str] | None

    barycentric_weights: np.ndarray
    fixed_remap_matrix: np.ndarray
    endpoint_blend: np.ndarray
    heat_spline_coeff: np.ndarray
    current_spline_coeff: np.ndarray

    psin_query: np.ndarray
    parameter_query: np.ndarray
    materialized_heat_input: np.ndarray
    materialized_current_input: np.ndarray
    scratch_1d: np.ndarray
    scratch_2d: np.ndarray

    target_root_fields: np.ndarray
    alpha_state: np.ndarray

    f_u: np.ndarray
    f_fields: np.ndarray
    psin_u: np.ndarray
    psin_fields: np.ndarray

    def bind_profile_views(self, *, F_profile: Profile, psin_profile: Profile) -> None:
        """Bind borrowed profile arrays used by source evaluation."""

        self.f_u = F_profile.u
        self.f_fields = F_profile.u_fields
        self.psin_u = psin_profile.u
        self.psin_fields = psin_profile.u_fields
