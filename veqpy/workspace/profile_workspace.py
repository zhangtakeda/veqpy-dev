"""Profile-stage runtime memory ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from veqpy.model.profile import Profile


@dataclass(init=False, slots=True)
class ProfileWorkspace:
    """Profile stage memory owner.

    Active profile field arrays have shape ``(n_active, 3, Nr)``.
    Derivative axis ``0/1/2`` means value, radial first derivative, radial second derivative.

    Fourier family field arrays have shape ``(M_max + 1, 3, Nr)``.
    Each named ``*_fields`` array is directly owned by this workspace; there is
    no hidden packing axis or first-axis row contract.
    """

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

    def __init__(
        self,
        *,
        nr: int,
        m_max: int,
        profile_names: tuple[str, ...],
        profile_index: dict[str, int],
        active_profile_ids: np.ndarray,
        profile_L: np.ndarray,
    ) -> None:
        """Allocate profile-stage runtime memory and profile-slot metadata."""

        n_active = int(active_profile_ids.size)
        max_active_len = 0
        if n_active > 0:
            max_active_len = max(int(profile_L[int(p)]) + 1 for p in active_profile_ids)

        self.active_u_fields = np.empty((n_active, 3, nr), dtype=np.float64)
        self.active_rp_fields = np.empty((n_active, 3, nr), dtype=np.float64)
        self.active_env_fields = np.empty((n_active, 3, nr), dtype=np.float64)
        self.active_offsets = np.empty(n_active, dtype=np.float64)
        self.active_scales = np.empty(n_active, dtype=np.float64)
        self.active_lengths = np.empty(n_active, dtype=np.int64)
        self.active_coeff_index_rows = np.full((n_active, max_active_len), -1, dtype=np.int64)

        self.c_family_fields = np.empty((m_max + 1, 3, nr), dtype=np.float64)
        self.s_family_fields = np.zeros((m_max + 1, 3, nr), dtype=np.float64)
        self.c_family_base_fields = np.zeros((m_max + 1, 3, nr), dtype=np.float64)
        self.s_family_base_fields = np.zeros((m_max + 1, 3, nr), dtype=np.float64)

        self.active_slot_by_profile_id = np.full(len(profile_names), -1, dtype=np.int64)
        for slot, profile_id in enumerate(active_profile_ids):
            self.active_slot_by_profile_id[int(profile_id)] = int(slot)

        self.c_family_source_slots = np.full(m_max + 1, -1, dtype=np.int64)
        self.s_family_source_slots = np.full(m_max + 1, -1, dtype=np.int64)
        for order in range(m_max + 1):
            c_name = f"c{order}"
            if c_name in profile_index:
                self.c_family_source_slots[order] = self.active_slot_by_profile_id[
                    profile_index[c_name]
                ]
            if order == 0:
                continue
            s_name = f"s{order}"
            if s_name in profile_index:
                self.s_family_source_slots[order] = self.active_slot_by_profile_id[
                    profile_index[s_name]
                ]

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
