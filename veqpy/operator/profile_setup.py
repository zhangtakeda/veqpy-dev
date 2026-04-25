"""
Module: operator.profile_setup

Role:
- 收敛 profile/case setup 与 profile-stage 装配的共享 Python 规则.
- 避免 operator.py 混入过多 profile 参数解析, stage A 绑定与 Fourier family 细节.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from veqpy.engine.numba_source import validate_route
from veqpy.engine.profile_regularization import resolve_fourier_power
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile
from veqpy.operator.layout import build_profile_layout
from veqpy.operator.operator_case import OperatorCase


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
    kwargs: dict[str, float | int | np.ndarray | None] = {}
    static_kwargs = profile_static_kwargs_by_name.get(name)
    if static_kwargs is None and name.startswith(("c", "s")) and name[1:].isdigit():
        order = int(name[1:])
        static_kwargs = {} if order == 0 else {"power": resolve_fourier_power(order)}
    if static_kwargs is not None:
        kwargs.update(static_kwargs)

    if name.startswith("c") and name[1:].isdigit():
        order = int(name[1:])
        kwargs["offset"] = 0.0 if order >= case.c_offsets.shape[0] else float(case.c_offsets[order])
    elif name.startswith("s") and name[1:].isdigit():
        order = int(name[1:])
        kwargs["offset"] = 0.0 if order >= case.s_offsets.shape[0] else float(case.s_offsets[order])
    else:
        try:
            offset_spec = profile_offset_specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown profile name {name!r}") from exc
        kwargs["offset"] = float(getattr(case, offset_spec)) if isinstance(offset_spec, str) else float(offset_spec)

    attrs = profile_scale_specs.get(name)
    scale = 1.0
    if attrs is not None:
        for attr in attrs:
            scale *= float(getattr(case, attr))
    kwargs["scale"] = scale

    p = profile_index[name]
    L = int(profile_L[p])
    coeff = case.profile_coeffs.get(name)
    kwargs["coeff"] = None if L < 0 or coeff is None else np.asarray(coeff, dtype=np.float64)[: L + 1].copy()
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
        static_kwargs = profile_static_kwargs_by_name.get(name)
        if static_kwargs is None and name.startswith(("c", "s")) and name[1:].isdigit():
            order = int(name[1:])
            static_kwargs = {} if order == 0 else {"power": resolve_fourier_power(order)}
        elif static_kwargs is None:
            static_kwargs = {}
        profile.power = int(static_kwargs.get("power", 0))
        profile.envelope_power = int(static_kwargs.get("envelope_power", 1))
        if name.startswith("c") and name[1:].isdigit():
            order = int(name[1:])
            profile.offset = 0.0 if order >= case.c_offsets.shape[0] else float(case.c_offsets[order])
        elif name.startswith("s") and name[1:].isdigit():
            order = int(name[1:])
            profile.offset = 0.0 if order >= case.s_offsets.shape[0] else float(case.s_offsets[order])
        else:
            offset_spec = profile_offset_specs[name]
            profile.offset = (
                float(getattr(case, offset_spec)) if isinstance(offset_spec, str) else float(offset_spec)
            )
        attrs = profile_scale_specs.get(name)
        profile.scale = 1.0
        if attrs is not None:
            for attr in attrs:
                profile.scale *= float(getattr(case, attr))
        p = profile_index[name]
        L = int(profile_L[p])
        coeff = case.profile_coeffs.get(name)
        profile.coeff = None if L < 0 or coeff is None else np.asarray(coeff, dtype=np.float64)[: L + 1].copy()
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
    update_profiles_packed_bulk: Callable,
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
    c_effective_order = 0
    for name in c_profile_names:
        order = int(name[1:])
        if profile_coeffs.get(name) is not None:
            c_effective_order = max(c_effective_order, order)
            continue
        if c_offsets is not None and order < c_offsets.shape[0] and abs(float(c_offsets[order])) > 1e-14:
            c_effective_order = max(c_effective_order, order)

    s_effective_order = 0
    for name in s_profile_names:
        order = int(name[1:])
        if profile_coeffs.get(name) is not None:
            s_effective_order = max(s_effective_order, order)
            continue
        if s_offsets is not None and order < s_offsets.shape[0] and abs(float(s_offsets[order])) > 1e-14:
            s_effective_order = max(s_effective_order, order)

    if c_effective_order + 1 < c_family_fields.shape[0]:
        c_family_fields[c_effective_order + 1 :].fill(0.0)
    if s_effective_order + 1 < s_family_fields.shape[0]:
        s_family_fields[s_effective_order + 1 :].fill(0.0)
    return c_effective_order, s_effective_order


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
