"""Profile-stage runtime memory ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from veqpy.model.profile import Profile


@dataclass(slots=True)
class ProfileWorkspace:
    """Profile stage memory owner.

    ``active_profile_slab`` shape: ``(3, n_active, 3, Nr)``.
    Rows are active profile fields, radial-prefactor fields, and envelope fields.
    Derivative axis ``0/1/2`` means value, radial first derivative, radial second derivative.
    The slab is the owning allocation; ``active_u_fields``, ``active_rp_fields``,
    and ``active_env_fields`` are semantic views into its first axis.

    ``family_field_slab`` shape: ``(4, M_max + 1, 3, Nr)``.
    Rows are c family fields, s family fields, c base fields, and s base fields.
    The named ``*_fields`` attributes are semantic views into this slab.

    A slab semantic view is a named slice into one owning array. The slab
    preserves one physical allocation for hot-path locality, while each view
    documents the row meaning expected by stage kernels and call sites.
    """

    active_profile_slab: np.ndarray
    family_field_slab: np.ndarray
    active_u_fields: np.ndarray
    active_rp_fields: np.ndarray
    active_env_fields: np.ndarray
    active_offsets: np.ndarray
    active_scales: np.ndarray
    active_lengths: np.ndarray
    active_coeff_index_rows: np.ndarray
    c_family_fields: np.ndarray
    s_family_fields: np.ndarray
    c_family_base_fields: np.ndarray
    s_family_base_fields: np.ndarray
    active_slot_by_profile_id: np.ndarray
    c_family_source_slots: np.ndarray
    s_family_source_slots: np.ndarray

    def residual_block_lengths(self) -> np.ndarray:
        """Return a copy of active residual block lengths for solver normalization."""

        return self.active_lengths.copy()

    def active_profile_blocks(
        self, *, active_profile_ids: np.ndarray, profile_names: tuple[str, ...]
    ) -> tuple[tuple[int, str, np.ndarray, float, float], ...]:
        """Return copy-based packed-profile metadata for solver scaling."""

        blocks: list[tuple[int, str, np.ndarray, float, float]] = []
        for slot, profile_id in enumerate(active_profile_ids):
            length = int(self.active_lengths[slot])
            if length <= 0:
                continue
            p = int(profile_id)
            blocks.append(
                (
                    p,
                    profile_names[p],
                    self.active_coeff_index_rows[slot, :length].copy(),
                    float(self.active_offsets[slot]),
                    float(self.active_scales[slot]),
                )
            )
        return tuple(blocks)

    def build_boundary_slope_initial_state(
        self,
        *,
        x_size: int,
        profile_names: tuple[str, ...],
        profiles_by_name: dict[str, Profile],
        boundary_slope_factor: float = 1.0,
    ) -> np.ndarray:
        """Build a boundary-scaled packed x0 for active c/s Fourier profiles."""

        x = np.zeros(x_size, dtype=np.float64)
        target_factor = float(boundary_slope_factor)
        for profile_id, name in enumerate(profile_names):
            if not (name.startswith("c") or name.startswith("s")):
                continue
            slot = int(self.active_slot_by_profile_id[int(profile_id)])
            if slot < 0 or int(self.active_lengths[slot]) <= 0:
                continue
            profile = profiles_by_name[name]
            power = int(profile.power)
            offset = float(profile.offset)
            if power <= 0 or abs(offset) <= 1.0e-14:
                continue
            coeff_index = int(self.active_coeff_index_rows[slot, 0])
            x[coeff_index] = 0.5 * (float(power) - target_factor) * offset
        return x
