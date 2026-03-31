"""
Module: operator.layout

Role:
- 负责构造 packed layout 与 profile 元数据.
- 负责提供 packed state 形状校验工具.

Public API:
- PACKED_PROFILE_FAMILY_ORDER
- INTERLEAVE_SHAPE_COEFFS_BY_ORDER
- get_prefix_profile_names
- build_fourier_profile_names
- build_shape_profile_names
- build_profile_names
- build_profile_index
- build_profile_layout
- build_active_profile_metadata
- packed_size
- validate_packed_state
- coeff_array_from_list

Notes:
- 这里定义的是 packed layout 规则.
- 不负责数值核计算, residual 组装, 或 solver 迭代控制.
"""

import numpy as np

from veqpy.residual_blocks import BLOCK_CODE_BY_NAME

_BLOCK_CODE_FAMILY_ORDER = tuple(
    "c0" if name == "c0" else "c" if name == "c_family" else "s" if name == "s_family" else name
    for name in BLOCK_CODE_BY_NAME
)
PACKED_PROFILE_FAMILY_ORDER = _BLOCK_CODE_FAMILY_ORDER
INTERLEAVE_SHAPE_COEFFS_BY_ORDER = True

_PREFIX_PROFILE_FAMILIES = ("psin", "F")
_SHAPE_PROFILE_FAMILIES = ("h", "v", "k", "c0", "c", "s")
_ALL_PROFILE_FAMILIES = _SHAPE_PROFILE_FAMILIES + _PREFIX_PROFILE_FAMILIES


def _validated_profile_family_order() -> tuple[str, ...]:
    family_order = tuple(PACKED_PROFILE_FAMILY_ORDER)
    if len(family_order) != len(_ALL_PROFILE_FAMILIES) or set(family_order) != set(_ALL_PROFILE_FAMILIES):
        raise ValueError(
            "PACKED_PROFILE_FAMILY_ORDER must contain each family exactly once: "
            f"{_ALL_PROFILE_FAMILIES!r}, got {family_order!r}"
        )
    return family_order


def get_prefix_profile_names() -> tuple[str, ...]:
    return tuple(family for family in _validated_profile_family_order() if family in _PREFIX_PROFILE_FAMILIES)


def _expand_profile_family(family: str, M_max: int) -> tuple[str, ...]:
    if family == "psin":
        return ("psin",)
    if family == "F":
        return ("F",)
    if family == "h":
        return ("h",)
    if family == "v":
        return ("v",)
    if family == "k":
        return ("k",)
    if family == "c0":
        return ("c0",)
    if family == "c":
        return tuple(f"c{k}" for k in range(1, M_max + 1))
    if family == "s":
        return tuple(f"s{k}" for k in range(1, M_max + 1))
    raise KeyError(f"Unknown profile family {family!r}")


def build_fourier_profile_names(M_max: int) -> tuple[str, ...]:
    M_max = int(M_max)
    if M_max < 0:
        raise ValueError(f"M_max must be non-negative, got {M_max}")

    fourier_names: list[str] = []
    for family in _validated_profile_family_order():
        if family not in ("c0", "c", "s"):
            continue
        fourier_names.extend(_expand_profile_family(family, M_max))
    return tuple(fourier_names)


def build_shape_profile_names(M_max: int) -> tuple[str, ...]:
    shape_profile_names: list[str] = []
    for family in _validated_profile_family_order():
        if family in _PREFIX_PROFILE_FAMILIES:
            continue
        shape_profile_names.extend(_expand_profile_family(family, int(M_max)))
    return tuple(shape_profile_names)


def build_profile_names(M_max: int) -> tuple[str, ...]:
    M_max = int(M_max)
    profile_names: list[str] = []
    for family in _validated_profile_family_order():
        profile_names.extend(_expand_profile_family(family, M_max))
    return tuple(profile_names)


def build_profile_index(profile_names: tuple[str, ...]) -> dict[str, int]:
    return {name: i for i, name in enumerate(profile_names)}


def build_profile_layout(
    profile_coeffs: dict[str, list[float] | None],
    *,
    profile_names: tuple[str, ...],
    prefix_profile_names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据 profile 系数字典构造 packed 状态布局."""
    profile_names = tuple(profile_names)
    if prefix_profile_names is None:
        prefix_profile_names = get_prefix_profile_names()
    prefix_profile_names = tuple(prefix_profile_names)
    profile_index = build_profile_index(profile_names)
    profile_count = len(profile_names)
    profile_L = np.full(profile_count, -1, dtype=np.int64)

    shape_profile_names = tuple(name for name in profile_names if name not in prefix_profile_names)

    unknown = set(profile_coeffs) - set(profile_index)
    if unknown:
        raise KeyError(f"Unknown profile names: {sorted(unknown)}")

    for name, coeff in profile_coeffs.items():
        if coeff is None:
            continue
        coeff_arr = coeff_array_from_list(name, coeff)
        profile_L[profile_index[name]] = coeff_arr.size - 1

    max_L = int(np.max(profile_L))
    if max_L < 0:
        raise ValueError("At least one active profile is required")

    coeff_index = -np.ones((profile_count, max_L + 1), dtype=np.int64)
    order_offsets = np.full(max_L + 2, -1, dtype=np.int64)

    x_pos = 0
    if INTERLEAVE_SHAPE_COEFFS_BY_ORDER:
        for k in range(max_L + 1):
            order_offsets[k] = x_pos
            for name in profile_names:
                p = profile_index[name]
                if profile_L[p] >= k:
                    coeff_index[p, k] = x_pos
                    x_pos += 1
    else:
        for name in profile_names:
            p = profile_index[name]
            L = int(profile_L[p])
            if L < 0:
                continue
            for k in range(L + 1):
                if order_offsets[k] < 0:
                    order_offsets[k] = x_pos
                coeff_index[p, k] = x_pos
                x_pos += 1
        for k in range(max_L + 1):
            if order_offsets[k] < 0:
                order_offsets[k] = x_pos
    order_offsets[max_L + 1] = x_pos

    return profile_L, coeff_index, order_offsets


def build_active_profile_metadata(
    profile_L: np.ndarray,
    *,
    profile_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """从 profile 阶数向量提取 active profile 元数据."""
    profile_L = np.asarray(profile_L, dtype=np.int64)
    expected_size = len(tuple(profile_names))
    if profile_L.ndim != 1 or profile_L.shape[0] != expected_size:
        raise ValueError(f"Expected profile_L to have shape ({expected_size},), got {profile_L.shape}")

    active_profile_mask = profile_L >= 0
    active_profile_ids = np.flatnonzero(active_profile_mask).astype(np.int64, copy=False)
    return active_profile_mask, active_profile_ids


def packed_size(coeff_index: np.ndarray) -> int:
    """统计 packed 向量的有效长度."""
    if coeff_index.size == 0:
        return 0
    return int(np.count_nonzero(coeff_index >= 0))


def validate_packed_state(x: np.ndarray, coeff_index: np.ndarray) -> np.ndarray:
    """校验 packed 状态向量形状并返回 float64 视图."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected x to be 1D, got {x.shape}")

    x_size = packed_size(coeff_index)
    if x.shape[0] != x_size:
        raise ValueError(f"Expected x to have shape ({x_size},), got {x.shape}")
    return x


def coeff_array_from_list(name: str, coeff: list[float]) -> np.ndarray:
    """把 profile 系数列表转成受约束的一维数组."""
    if not isinstance(coeff, list):
        raise TypeError(f"{name} coeff must be list[float] or None, got {type(coeff).__name__}")
    coeff_arr = np.asarray(coeff, dtype=np.float64)
    if coeff_arr.ndim != 1:
        raise ValueError(f"{name} coeff must be 1D, got {coeff_arr.shape}")
    if coeff_arr.size == 0:
        raise ValueError(f"{name} coeff must be non-empty or None")
    return coeff_arr
