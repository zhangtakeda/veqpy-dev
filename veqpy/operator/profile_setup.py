"""
Module: operator.profile_setup

Role:
- 收敛 profile/case setup 与 profile-stage 装配的共享 Python 规则.
- 避免 operator.py 混入过多 profile 参数解析, stage A 绑定与 Fourier family 细节.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from veqpy.engine import update_profiles_packed_bulk, validate_route
from veqpy.model import Grid, Profile
from veqpy.operator.layout import build_profile_layout
from veqpy.operator.operator_case import OperatorCase


def profile_static_kwargs(
    name: str,
    *,
    profile_static_kwargs_by_name: dict[str, dict[str, int]],
) -> dict[str, int]:
    if name in profile_static_kwargs_by_name:
        return profile_static_kwargs_by_name[name]
    if name.startswith(("c", "s")) and name[1:].isdigit():
        order = int(name[1:])
        return {} if order == 0 else {"power": order}
    return {}


def profile_offset_from_case(
    case: OperatorCase,
    name: str,
    *,
    profile_offset_specs: dict[str, float | str],
) -> float:
    if name.startswith("c") and name[1:].isdigit():
        return offset_from_array(case.c_offsets, int(name[1:]))
    if name.startswith("s") and name[1:].isdigit():
        return offset_from_array(case.s_offsets, int(name[1:]))
    try:
        spec = profile_offset_specs[name]
    except KeyError as exc:
        raise KeyError(f"Unknown profile name {name!r}") from exc
    if isinstance(spec, str):
        return float(getattr(case, spec))
    return float(spec)


def profile_scale_from_case(
    case: OperatorCase,
    name: str,
    *,
    profile_scale_specs: dict[str, tuple[str, ...]],
) -> float:
    attrs = profile_scale_specs.get(name)
    if attrs is None:
        return 1.0
    scale = 1.0
    for attr in attrs:
        scale *= float(getattr(case, attr))
    return scale


def profile_coeff_from_case(
    case: OperatorCase,
    *,
    p: int,
    profile_L: np.ndarray,
    profile_names: tuple[str, ...],
) -> np.ndarray | None:
    L = int(profile_L[p])
    if L < 0:
        return None
    coeff = case.profile_coeffs.get(profile_names[p])
    if coeff is None:
        return None
    arr = np.asarray(coeff, dtype=np.float64)
    return arr[: L + 1].copy()


def make_profile(
    *,
    case: OperatorCase,
    name: str,
    profile_L: np.ndarray,
    profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    profile_static_kwargs_by_name: dict[str, dict[str, int]],
    profile_offset_specs: dict[str, float | str],
    profile_scale_specs: dict[str, tuple[str, ...]],
) -> Profile:
    kwargs: dict[str, float | int | np.ndarray | None] = dict(
        profile_static_kwargs(name, profile_static_kwargs_by_name=profile_static_kwargs_by_name)
    )
    kwargs["offset"] = profile_offset_from_case(case, name, profile_offset_specs=profile_offset_specs)
    kwargs["coeff"] = profile_coeff_from_case(
        case,
        p=profile_index[name],
        profile_L=profile_L,
        profile_names=profile_names,
    )
    kwargs["scale"] = profile_scale_from_case(case, name, profile_scale_specs=profile_scale_specs)
    return Profile(**kwargs)


def refresh_profile_runtime(
    *,
    case: OperatorCase,
    grid: Grid,
    profile_names: tuple[str, ...],
    profile_index: dict[str, int],
    profile_L: np.ndarray,
    profiles_by_name: dict[str, Profile],
    profile_static_kwargs_by_name: dict[str, dict[str, int]],
    profile_offset_specs: dict[str, float | str],
    profile_scale_specs: dict[str, tuple[str, ...]],
    refresh_fourier_family_base_fields: Callable[[], None],
) -> None:
    for name in profile_names:
        profile = profiles_by_name[name]
        static_kwargs = profile_static_kwargs(name, profile_static_kwargs_by_name=profile_static_kwargs_by_name)
        profile.power = int(static_kwargs.get("power", 0))
        profile.envelope_power = int(static_kwargs.get("envelope_power", 1))
        profile.offset = profile_offset_from_case(case, name, profile_offset_specs=profile_offset_specs)
        profile.scale = profile_scale_from_case(case, name, profile_scale_specs=profile_scale_specs)
        profile.coeff = profile_coeff_from_case(
            case,
            p=profile_index[name],
            profile_L=profile_L,
            profile_names=profile_names,
        )
        profile._prepare_runtime_cache(grid)
        profile.update()
    refresh_fourier_family_base_fields()


def refresh_stage_a_runtime(
    *,
    active_profile_ids: np.ndarray,
    profile_names: tuple[str, ...],
    profiles_by_name: dict[str, Profile],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    active_u_fields: np.ndarray,
    active_rp_fields: np.ndarray,
    active_env_fields: np.ndarray,
    active_offsets: np.ndarray,
    active_scales: np.ndarray,
    active_lengths: np.ndarray,
    active_coeff_index_rows: np.ndarray,
) -> None:
    if active_profile_ids.size == 0:
        return

    for slot, p in enumerate(active_profile_ids):
        p_int = int(p)
        profile_name = profile_names[p_int]
        profile = profiles_by_name[profile_name]
        L = int(profile_L[p_int])
        coeff_indices = coeff_index[p_int, : L + 1]

        profile.u_fields = active_u_fields[slot]
        active_rp_fields[slot] = profile.rp_fields
        active_env_fields[slot] = profile.env_fields
        active_offsets[slot] = profile.offset
        active_scales[slot] = profile.scale
        active_lengths[slot] = coeff_indices.size
        if active_coeff_index_rows.shape[1] > 0:
            active_coeff_index_rows[slot].fill(-1)
            active_coeff_index_rows[slot, : coeff_indices.size] = coeff_indices


def build_profile_stage_runner(
    *,
    active_profile_ids: np.ndarray,
    active_profile_slab: np.ndarray,
    T_fields: np.ndarray,
    active_offsets: np.ndarray,
    active_scales: np.ndarray,
    active_coeff_index_rows: np.ndarray,
    active_lengths: np.ndarray,
) -> Callable[[np.ndarray], None]:
    if active_profile_ids.size == 0:
        return lambda x: None

    def runner(x: np.ndarray) -> None:
        update_profiles_packed_bulk(
            active_profile_slab,
            T_fields,
            active_offsets,
            active_scales,
            x,
            active_coeff_index_rows,
            active_lengths,
        )

    return runner


def refresh_fourier_family_base_fields(
    *,
    M_max: int,
    profile_index: dict[str, int],
    profiles_by_name: dict[str, Profile],
    c_family_base_fields: np.ndarray,
    s_family_base_fields: np.ndarray,
) -> None:
    c_family_base_fields.fill(0.0)
    s_family_base_fields.fill(0.0)
    for order in range(int(M_max) + 1):
        c_name = f"c{order}"
        if c_name in profile_index:
            np.copyto(c_family_base_fields[order], profiles_by_name[c_name].u_fields)
        if order == 0:
            continue
        s_name = f"s{order}"
        if s_name in profile_index:
            np.copyto(s_family_base_fields[order], profiles_by_name[s_name].u_fields)


def refresh_fourier_family_metadata(
    *,
    c_profile_names: tuple[str, ...],
    s_profile_names: tuple[str, ...],
    profile_coeffs: dict[str, list[float] | None],
    c_offsets: np.ndarray | None,
    s_offsets: np.ndarray | None,
    c_family_fields: np.ndarray,
    s_family_fields: np.ndarray,
) -> tuple[int, int]:
    c_effective_order = effective_family_order(c_profile_names, profile_coeffs, c_offsets, minimum_order=0)
    s_effective_order = effective_family_order(s_profile_names, profile_coeffs, s_offsets, minimum_order=0)

    if c_effective_order + 1 < c_family_fields.shape[0]:
        c_family_fields[c_effective_order + 1 :].fill(0.0)
    if s_effective_order + 1 < s_family_fields.shape[0]:
        s_family_fields[s_effective_order + 1 :].fill(0.0)
    return c_effective_order, s_effective_order


def effective_family_order(
    profile_names: tuple[str, ...],
    profile_coeffs: dict[str, list[float] | None],
    offsets: np.ndarray | None,
    *,
    minimum_order: int,
) -> int:
    effective_order = int(minimum_order)
    for name in profile_names:
        order = int(name[1:])
        if profile_coeffs.get(name) is not None:
            effective_order = max(effective_order, order)
            continue
        if abs(offset_from_array(offsets, order)) > 1e-14:
            effective_order = max(effective_order, order)
    return effective_order


def validate_case_compatibility(
    case: OperatorCase,
    *,
    profile_names: tuple[str, ...],
    prefix_profile_names: tuple[str, ...],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    order_offsets: np.ndarray,
    validate_source_inputs: Callable[[OperatorCase], None],
) -> None:
    validate_route(case.route, case.coordinate, case.nodes)
    next_profile_L, next_coeff_index, next_order_offsets = build_profile_layout(
        case.profile_coeffs,
        profile_names=profile_names,
        prefix_profile_names=prefix_profile_names,
    )
    if not np.array_equal(next_profile_L, profile_L):
        raise ValueError("Replacement case changes the active profile layout")
    if not np.array_equal(next_coeff_index, coeff_index):
        raise ValueError("Replacement case changes the packed coefficient layout")
    if not np.array_equal(next_order_offsets, order_offsets):
        raise ValueError("Replacement case changes the degree ordering layout")
    validate_source_inputs(case)


def offset_from_array(offsets: np.ndarray | None, order: int) -> float:
    if offsets is None or order >= offsets.shape[0]:
        return 0.0
    return float(offsets[order])
