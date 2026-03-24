from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from veqpy.engine import (
    assemble_c0_residual_block,
    assemble_c1_residual_block,
    assemble_F_residual_block,
    assemble_h_residual_block,
    assemble_k_residual_block,
    assemble_psin_residual_block,
    assemble_s1_residual_block,
    assemble_s2_residual_block,
    assemble_v_residual_block,
    bind_runner,
    update_residual,
    validate_operator,
)
from veqpy.model import Equilibrium, Geometry, Grid, Profile
from veqpy.operator.codec import (
    decode_packed_blocks,
    decode_packed_state_inplace,
    encode_packed_residual,
    encode_packed_state,
)
from veqpy.operator.layout import (
    PROFILE_INDEX,
    PROFILE_NAMES,
    build_active_profile_metadata,
    build_profile_layout,
    packed_size,
)
from veqpy.operator.operator_case import OperatorCase


@dataclass(slots=True, frozen=True)
class ProfileFillSlot:
    fill: Callable[[], None]


@dataclass(slots=True, frozen=True)
class ResidualAssembleSlot:
    coeff_row: np.ndarray
    coeff_indices: np.ndarray
    assemble: Callable[[], None]


@dataclass(slots=True, frozen=True)
class HomotopyStageGroup:
    order: int
    indices: np.ndarray
    shape_profile_ids: np.ndarray


@dataclass(slots=True)
class Operator:
    name: str
    derivative: str
    grid: Grid = field(repr=False)
    case: OperatorCase = field(repr=False)
    geometry: Geometry = field(init=False)

    coeff_matrix: np.ndarray = field(init=False)
    h_profile: Profile = field(init=False)
    v_profile: Profile = field(init=False)
    k_profile: Profile = field(init=False)
    c0_profile: Profile = field(init=False)
    c1_profile: Profile = field(init=False)
    s1_profile: Profile = field(init=False)
    s2_profile: Profile = field(init=False)
    psin_profile: Profile = field(init=False)
    F_profile: Profile = field(init=False)

    psin_R: np.ndarray = field(init=False)
    psin_Z: np.ndarray = field(init=False)
    G: np.ndarray = field(init=False)
    psin_r: np.ndarray = field(init=False)
    psin_rr: np.ndarray = field(init=False)
    FFn_r: np.ndarray = field(init=False)
    Pn_r: np.ndarray = field(init=False)
    alpha1: float = field(init=False)
    alpha2: float = field(init=False)

    profile_L: np.ndarray = field(init=False, repr=False)
    coeff_index: np.ndarray = field(init=False, repr=False)
    order_offsets: np.ndarray = field(init=False, repr=False)
    active_profile_mask: np.ndarray = field(init=False, repr=False)
    active_profile_ids: np.ndarray = field(init=False, repr=False)
    x_size: int = field(init=False, repr=False)
    profile_fill_slots: tuple[ProfileFillSlot, ...] = field(init=False, repr=False)
    residual_slots: tuple[ResidualAssembleSlot, ...] = field(init=False, repr=False)
    source_runner: Callable = field(init=False, repr=False)

    def __post_init__(self) -> None:
        spec = validate_operator(self.name, self.derivative)

        self.name = spec.name
        self.geometry = Geometry(grid=self.grid)

        self.profile_L, self.coeff_index, self.order_offsets = build_profile_layout(self.case.coeffs_by_name)
        self.active_profile_mask, self.active_profile_ids = build_active_profile_metadata(self.profile_L)
        self.x_size = packed_size(self.coeff_index)

        self._allocate_runtime_arrays()
        self.source_runner = bind_runner(spec.name, self.derivative)
        self.residual_slots = ()
        self._refresh_case_runtime()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.residual(x)

    def replace_case(self, case: OperatorCase) -> None:
        self._validate_case_compatibility(case)
        self.case = case
        self._refresh_case_runtime()

    def encode_initial_state(self) -> np.ndarray:
        return encode_packed_state(self.case.coeffs_by_name, self.profile_L, self.coeff_index)

    def residual(self, x: np.ndarray) -> np.ndarray:
        x_eval = self.coerce_x(x)
        self.stage_a_profile(x_eval)
        self.stage_b_geometry()
        self.stage_c_source()
        return self.stage_d_residual()

    def residual_prefix(
        self,
        x_prefix: np.ndarray,
        *,
        active_len: int,
        x_template: np.ndarray,
    ) -> np.ndarray:
        x_full = self._compose_active_state(x_prefix, active_len=active_len, x_template=x_template)
        return self.residual(x_full)[:active_len].copy()

    def residual_masked(
        self,
        x_active: np.ndarray,
        *,
        active_indices: np.ndarray,
        x_template: np.ndarray,
    ) -> np.ndarray:
        x_full = self._compose_masked_state(x_active, active_indices=active_indices, x_template=x_template)
        return self.residual(x_full)[active_indices].copy()

    def homotopy_frontiers(self) -> np.ndarray:
        if self.x_size == 0:
            return np.zeros(0, dtype=np.int64)
        frontiers = np.asarray(self.order_offsets[1:], dtype=np.int64)
        frontiers = np.unique(frontiers)
        frontiers = frontiers[(frontiers > 0) & (frontiers <= self.x_size)]
        return frontiers.astype(np.int64, copy=False)

    def homotopy_stage_groups(self) -> tuple[HomotopyStageGroup, ...]:
        if self.x_size == 0:
            return ()

        prefix_indices: list[int] = []
        for name in ("F", "psin"):
            p = PROFILE_INDEX[name]
            L = int(self.profile_L[p])
            if L < 0:
                continue
            for k in range(L + 1):
                idx = int(self.coeff_index[p, k])
                if idx >= 0:
                    prefix_indices.append(idx)

        shape_profile_ids = [int(PROFILE_INDEX[name]) for name in ("h", "v", "k", "c0", "c1", "s1", "s2")]
        active_shape_ids = [p for p in shape_profile_ids if int(self.profile_L[p]) >= 0]
        if not active_shape_ids:
            if prefix_indices:
                return (
                    HomotopyStageGroup(
                        order=0,
                        indices=np.asarray(prefix_indices, dtype=np.int64),
                        shape_profile_ids=np.zeros(0, dtype=np.int64),
                    ),
                )
            return ()

        max_shape_order = max(int(self.profile_L[p]) for p in active_shape_ids)
        groups: list[HomotopyStageGroup] = []
        for order in range(max_shape_order + 1):
            shape_ids = [p for p in active_shape_ids if int(self.profile_L[p]) >= order]
            indices = list(prefix_indices) if order == 0 else []
            for p in shape_ids:
                idx = int(self.coeff_index[p, order])
                if idx >= 0:
                    indices.append(idx)
            if not indices:
                continue
            groups.append(
                HomotopyStageGroup(
                    order=order,
                    indices=np.asarray(indices, dtype=np.int64),
                    shape_profile_ids=np.asarray(shape_ids, dtype=np.int64),
                )
            )
        return tuple(groups)

    def homotopy_truncation_profile_ids(self) -> np.ndarray:
        profile_ids = [
            int(PROFILE_INDEX[name]) for name in ("h", "v", "k", "c0", "c1", "s1", "s2") if int(self.profile_L[PROFILE_INDEX[name]]) >= 0
        ]
        return np.asarray(profile_ids, dtype=np.int64)

    def build_coeffs(self, x: np.ndarray, *, include_none: bool = True) -> dict[str, list[float] | None]:
        blocks = decode_packed_blocks(x, self.profile_L, self.coeff_index)
        coeffs: dict[str, list[float] | None] = {}
        for name, block in zip(PROFILE_NAMES, blocks, strict=True):
            if include_none or block is not None:
                coeffs[name] = None if block is None else block.tolist()
        return coeffs

    def build_equilibrium(self, x: np.ndarray) -> Equilibrium:
        x_eval = self.coerce_x(x)
        self.stage_a_profile(x_eval)
        self.stage_b_geometry()
        self.stage_c_source()
        return self._build_equilibrium_from_runtime()

    def stage_a_profile(self, x: np.ndarray) -> None:
        decode_packed_state_inplace(x, self.profile_L, self.coeff_index, self.coeff_matrix)
        self._fill_profile_rows(self.active_profile_ids)

    def stage_b_geometry(self) -> None:
        self.geometry.update(
            float(self.case.a),
            float(self.case.R0),
            float(self.case.Z0),
            self.grid,
            self.h_profile,
            self.v_profile,
            self.k_profile,
            self.c0_profile,
            self.c1_profile,
            self.s1_profile,
            self.s2_profile,
        )

    def stage_c_source(self) -> None:
        alpha1, alpha2 = self.source_runner(
            self.psin_r,
            self.psin_rr,
            self.FFn_r,
            self.Pn_r,
            self.case.heat_input,
            self.case.current_input,
            float(self.case.R0),
            float(self.case.B0),
            self.grid.weights,
            self.grid.differentiation_matrix,
            self.grid.integration_matrix,
            self.grid.rho,
            self.geometry.V_r,
            self.geometry.Kn,
            self.geometry.Kn_r,
            self.geometry.Ln_r,
            self.geometry.S_r,
            self.geometry.R,
            self.geometry.JdivR,
            self.F_profile.u,
            float(self.case.Ip),
            float(self.case.beta),
        )
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)

    def stage_d_residual(self) -> np.ndarray:
        self._build_G_inplace()
        return self._assemble_residual()

    def coerce_x(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != self.x_size:
            raise ValueError(f"Expected x to have shape ({self.x_size},), got {arr.shape}")
        return arr

    def _coerce_active_x(self, x: np.ndarray, active_len: int) -> np.ndarray:
        active_len = self._validate_active_len(active_len)
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != active_len:
            raise ValueError(f"Expected active x to have shape ({active_len},), got {arr.shape}")
        return arr

    def _validate_active_len(self, active_len: int) -> int:
        length = int(active_len)
        if length < 0 or length > self.x_size:
            raise ValueError(f"active_len must be in [0, {self.x_size}], got {active_len!r}")
        return length

    def _coerce_active_indices(self, active_indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(active_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(f"Expected active indices to be 1D, got {indices.shape}")
        if np.any(indices < 0) or np.any(indices >= self.x_size):
            raise ValueError(f"Active indices must be within [0, {self.x_size})")
        if indices.size != np.unique(indices).size:
            raise ValueError("Active indices must be unique")
        return indices

    def _coerce_masked_x(self, x: np.ndarray, active_indices: np.ndarray) -> np.ndarray:
        indices = self._coerce_active_indices(active_indices)
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != indices.shape[0]:
            raise ValueError(f"Expected masked x to have shape ({indices.shape[0]},), got {arr.shape}")
        return arr

    def _compose_active_state(
        self,
        x_prefix: np.ndarray,
        *,
        active_len: int,
        x_template: np.ndarray,
    ) -> np.ndarray:
        active_len = self._validate_active_len(active_len)
        x_active = self._coerce_active_x(x_prefix, active_len)
        x_full = self.coerce_x(x_template).copy()
        x_full[:active_len] = x_active
        return x_full

    def _compose_masked_state(
        self,
        x_active: np.ndarray,
        *,
        active_indices: np.ndarray,
        x_template: np.ndarray,
    ) -> np.ndarray:
        indices = self._coerce_active_indices(active_indices)
        x_values = self._coerce_masked_x(x_active, indices)
        x_full = self.coerce_x(x_template).copy()
        x_full[indices] = x_values
        return x_full

    def _allocate_runtime_arrays(self) -> None:
        nr = self.grid.Nr
        nt = self.grid.Nt

        self.coeff_matrix = np.zeros_like(self.coeff_index, dtype=np.float64)
        self.h_profile = Profile(offset=0.0)
        self.v_profile = Profile(offset=0.0)
        self.k_profile = Profile(offset=float(self.case.ka))
        self.c0_profile = Profile(offset=float(self.case.c0a))
        self.c1_profile = Profile(power=1, offset=float(self.case.c1a))
        self.s1_profile = Profile(power=1, offset=float(self.case.s1a))
        self.s2_profile = Profile(power=2, offset=float(self.case.s2a))
        self.psin_profile = Profile(power=2, offset=1.0)
        self.F_profile = Profile(scale=self.case.R0 * self.case.B0, envelope_power=2, offset=1.0)

        self.psin_R = np.zeros((nr, nt), dtype=np.float64)
        self.psin_Z = np.zeros((nr, nt), dtype=np.float64)
        self.G = np.zeros((nr, nt), dtype=np.float64)
        self.psin_r = np.empty(nr, dtype=np.float64)
        self.psin_rr = np.empty(nr, dtype=np.float64)
        self.FFn_r = np.empty(nr, dtype=np.float64)
        self.Pn_r = np.empty(nr, dtype=np.float64)
        self.alpha1 = 0.0
        self.alpha2 = 0.0

    def _refresh_case_runtime(self) -> None:
        self._sync_profile_specs()
        self._load_case_coeff_matrix()
        self.profile_fill_slots = self._build_profile_fill_slots()
        self._rebind_runtime()

    def _sync_profile_specs(self) -> None:
        self.h_profile.offset = 0.0
        self.v_profile.offset = 0.0
        self.k_profile.offset = float(self.case.ka)
        self.c0_profile.offset = float(self.case.c0a)
        self.c1_profile.offset = float(self.case.c1a)
        self.s1_profile.offset = float(self.case.s1a)
        self.s2_profile.offset = float(self.case.s2a)
        self.psin_profile.offset = 1.0
        self.F_profile.offset = 1.0
        self.F_profile.scale = self.case.R0 * self.case.B0

        for profile in (
            self.h_profile,
            self.v_profile,
            self.k_profile,
            self.c0_profile,
            self.c1_profile,
            self.s1_profile,
            self.s2_profile,
            self.psin_profile,
            self.F_profile,
        ):
            profile._prepare_runtime_cache(self.grid)

    def _load_case_coeff_matrix(self) -> None:
        self.coeff_matrix.fill(0.0)
        coeff_dict = self.case.coeffs_by_name
        for p, name in enumerate(PROFILE_NAMES):
            L = int(self.profile_L[p])
            if L < 0:
                continue
            coeff = coeff_dict.get(name)
            if coeff is None:
                continue
            self.coeff_matrix[p, : L + 1] = np.asarray(coeff, dtype=np.float64)

    def _validate_case_compatibility(self, case: OperatorCase) -> None:
        profile_L, coeff_index, order_offsets = build_profile_layout(case.coeffs_by_name)
        if not np.array_equal(profile_L, self.profile_L):
            raise ValueError("Replacement case changes the active profile layout")
        if not np.array_equal(coeff_index, self.coeff_index):
            raise ValueError("Replacement case changes the packed coefficient layout")
        if not np.array_equal(order_offsets, self.order_offsets):
            raise ValueError("Replacement case changes the degree ordering layout")
        if case.heat_input.shape != (self.grid.Nr,) or case.current_input.shape != (self.grid.Nr,):
            raise ValueError(f"Expected heat_input/current_input to have shape ({self.grid.Nr},)")

    def _rebind_runtime(self) -> None:
        self.residual_slots = self._build_residual_slots()
        fixed_profile_ids = np.flatnonzero(~self.active_profile_mask).astype(np.int64, copy=False)
        self._fill_profile_rows(fixed_profile_ids)

    def _build_profile_fill_slots(self) -> tuple[ProfileFillSlot, ...]:
        slots: list[ProfileFillSlot] = []
        for p, profile_name in enumerate(PROFILE_NAMES):
            slots.append(self._build_profile_fill_slot(p, profile_name))
        return tuple(slots)

    def _build_profile_fill_slot(self, p: int, profile_name: str) -> ProfileFillSlot:
        L = int(self.profile_L[p])
        coeff_row = None if L < 0 else self.coeff_matrix[p, : L + 1]
        profile = getattr(self, f"{profile_name}_profile")
        if profile_name not in _PROFILE_OFFSET_SOURCES:
            raise ValueError(f"Unsupported profile {profile_name!r}")

        def fill() -> None:
            profile.update(coeff_row)

        return ProfileFillSlot(fill=fill)

    def _build_residual_slots(self) -> tuple[ResidualAssembleSlot, ...]:
        slots: list[ResidualAssembleSlot] = []
        for p in self.active_profile_ids:
            slots.append(self._build_residual_slot(int(p)))
        return tuple(slots)

    def _build_residual_slot(self, p: int) -> ResidualAssembleSlot:
        L = int(self.profile_L[p])
        profile_name = PROFILE_NAMES[p]
        coeff_row = self.coeff_matrix[p, : L + 1]
        coeff_indices = self.coeff_index[p, : L + 1]
        a = float(self.case.a)
        R0 = float(self.case.R0)
        B0 = float(self.case.B0)

        match profile_name:
            case "h":

                def assemble() -> None:
                    assemble_h_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_R,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "v":

                def assemble() -> None:
                    assemble_v_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_Z,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "k":

                def assemble() -> None:
                    assemble_k_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_Z,
                        self.grid.sin_theta,
                        self.grid.rho,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "c0":

                def assemble() -> None:
                    assemble_c0_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_R,
                        self.geometry.sin_tb,
                        self.grid.rho,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "c1":

                def assemble() -> None:
                    assemble_c1_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_R,
                        self.geometry.sin_tb,
                        self.grid.cos_theta,
                        self.grid.rho2,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "s1":

                def assemble() -> None:
                    assemble_s1_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_R,
                        self.geometry.sin_tb,
                        self.grid.sin_theta,
                        self.grid.rho2,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "s2":

                def assemble() -> None:
                    assemble_s2_residual_block(
                        coeff_row,
                        self.G,
                        self.psin_R,
                        self.geometry.sin_tb,
                        self.grid.sin_2theta,
                        self.grid.rho,
                        self.grid.rho2,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        a,
                    )
            case "psin":

                def assemble() -> None:
                    assemble_psin_residual_block(
                        coeff_row,
                        self.G,
                        self.grid.rho2,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                    )
            case "F":

                def assemble() -> None:
                    assemble_F_residual_block(
                        coeff_row,
                        self.G,
                        self.grid.y,
                        self.grid.T,
                        self.grid.weights,
                        R0,
                        B0,
                    )
            case _:
                raise ValueError(f"Unsupported active profile {profile_name!r}")

        return ResidualAssembleSlot(
            coeff_row=coeff_row,
            coeff_indices=coeff_indices,
            assemble=assemble,
        )

    def _fill_profile_rows(self, profile_ids: np.ndarray) -> None:
        for p in profile_ids:
            self.profile_fill_slots[int(p)].fill()

    def _build_G_inplace(self) -> None:
        update_residual(
            self.psin_R,
            self.psin_Z,
            self.G,
            self.alpha1,
            self.alpha2,
            self.psin_r,
            self.psin_rr,
            self.FFn_r,
            self.Pn_r,
            self.geometry.R,
            self.geometry.R_t,
            self.geometry.Z_t,
            self.geometry.J,
            self.geometry.JdivR,
            self.geometry.gttdivJR,
            self.geometry.grtdivJR_t,
            self.geometry.gttdivJR_r,
        )

    def _assemble_residual(self) -> np.ndarray:
        return encode_packed_residual(self.residual_slots, self.x_size)

    def _build_equilibrium_from_runtime(self) -> Equilibrium:
        case = self.case
        return Equilibrium(
            R0=case.R0,
            Z0=case.Z0,
            B0=case.B0,
            a=case.a,
            grid=self.grid,
            active_profiles=[
                name for name in ("h", "v", "k", "c0", "c1", "s1", "s2") if case.coeffs_by_name[name] is not None
            ],
            h_profile=self.h_profile.copy(),
            v_profile=self.v_profile.copy(),
            k_profile=self.k_profile.copy(),
            c0_profile=self.c0_profile.copy(),
            c1_profile=self.c1_profile.copy(),
            s1_profile=self.s1_profile.copy(),
            s2_profile=self.s2_profile.copy(),
            FFn_r=self.FFn_r.copy(),
            Pn_r=self.Pn_r.copy(),
            psin_r=self.psin_r.copy(),
            psin_rr=self.psin_rr.copy(),
            alpha1=float(self.alpha1),
            alpha2=float(self.alpha2),
        )


_PROFILE_OFFSET_SOURCES = {
    "h": ("const", 0.0),
    "v": ("const", 0.0),
    "k": ("attr", "ka"),
    "c0": ("attr", "c0a"),
    "c1": ("attr", "c1a"),
    "s1": ("attr", "s1a"),
    "s2": ("attr", "s2a"),
    "psin": ("const", 1.0),
    "F": ("const", 1.0),
}
