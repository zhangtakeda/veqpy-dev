"""
Module: operator.profile_setup

Role:
- 收敛 profile/case setup 的共享 Python 规则.
- 避免 operator.py 混入过多 profile 参数解析与 case 兼容性校验细节.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from veqpy.engine import validate_route
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
    profile_offset_specs: dict[str, float | str],
    profile_scale_specs: dict[str, tuple[str, ...]],
    refresh_fourier_family_base_fields: Callable[[], None],
) -> None:
    for name in profile_names:
        profile = profiles_by_name[name]
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
