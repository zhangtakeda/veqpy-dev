"""
operator 层 packed layout helper.
负责从 profile 系数字典构造稳定 packed 排布, 并提供 active profile 元数据与形状校验工具.
不负责数值核计算, residual 组装, solver 迭代控制.
"""

import numpy as np

PREFIX_PROFILE_NAMES = ("psin", "F")
SHAPE_PROFILE_NAMES = ("h", "v", "k", "c0", "c1", "s1", "s2")
PROFILE_NAMES = PREFIX_PROFILE_NAMES + SHAPE_PROFILE_NAMES
PROFILE_INDEX = {name: i for i, name in enumerate(PROFILE_NAMES)}
PROFILE_COUNT = len(PROFILE_NAMES)


def build_profile_layout(
    coeffs_by_name: dict[str, list[float] | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据 profile 系数字典构造 packed 状态布局.

    Args:
        coeffs_by_name: profile 名到系数列表的映射. None 表示该 profile 不参与当前 layout.

    Returns:
        返回 profile_L, coeff_index, order_offsets 三元组.
        profile_L 记录各 profile 的最高阶数, coeff_index 给出 packed 向量索引, order_offsets 给出 homotopy 分阶边界.
    """
    profile_L = np.full(PROFILE_COUNT, -1, dtype=np.int64)

    unknown = set(coeffs_by_name) - set(PROFILE_INDEX)
    if unknown:
        raise KeyError(f"Unknown profile names: {sorted(unknown)}")

    for name, coeff in coeffs_by_name.items():
        if coeff is None:
            continue
        coeff_arr = coeff_array_from_list(name, coeff)
        profile_L[PROFILE_INDEX[name]] = coeff_arr.size - 1

    max_L = int(np.max(profile_L))
    if max_L < 0:
        raise ValueError("At least one active profile is required")

    coeff_index = -np.ones((PROFILE_COUNT, max_L + 1), dtype=np.int64)
    order_offsets = np.zeros(max_L + 2, dtype=np.int64)

    x_pos = 0
    for name in PREFIX_PROFILE_NAMES:
        p = PROFILE_INDEX[name]
        L = int(profile_L[p])
        if L < 0:
            continue
        for k in range(L + 1):
            coeff_index[p, k] = x_pos
            x_pos += 1

    for k in range(max_L + 1):
        order_offsets[k] = x_pos
        for name in SHAPE_PROFILE_NAMES:
            p = PROFILE_INDEX[name]
            if profile_L[p] >= k:
                coeff_index[p, k] = x_pos
                x_pos += 1
    order_offsets[max_L + 1] = x_pos

    return profile_L, coeff_index, order_offsets


def build_active_profile_metadata(profile_L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    从 profile 阶数向量提取 active profile 元数据.

    Args:
        profile_L: 各 profile 的最高阶数向量, shape=(PROFILE_COUNT,).

    Returns:
        返回 active_profile_mask 和 active_profile_ids.
        前者是布尔掩码, 后者是升序 profile 编号数组.
    """
    profile_L = np.asarray(profile_L, dtype=np.int64)
    if profile_L.ndim != 1 or profile_L.shape[0] != PROFILE_COUNT:
        raise ValueError(f"Expected profile_L to have shape ({PROFILE_COUNT},), got {profile_L.shape}")

    active_profile_mask = profile_L >= 0
    active_profile_ids = np.flatnonzero(active_profile_mask).astype(np.int64, copy=False)
    return active_profile_mask, active_profile_ids


def packed_size(coeff_index: np.ndarray) -> int:
    """
    统计 packed 向量的有效长度.

    Args:
        coeff_index: profile 到 packed 向量的索引矩阵.

    Returns:
        返回非负索引的个数, 即 packed 状态向量长度.
    """
    if coeff_index.size == 0:
        return 0
    return int(np.count_nonzero(coeff_index >= 0))


def validate_packed_state(x: np.ndarray, coeff_index: np.ndarray) -> np.ndarray:
    """
    校验 packed 状态向量形状并返回 float64 视图.

    Args:
        x: 待校验的一维 packed 状态向量.
        coeff_index: 当前 layout 的索引矩阵.

    Returns:
        返回通过校验后的 float64 一维数组视图.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected x to be 1D, got {x.shape}")

    x_size = packed_size(coeff_index)
    if x.shape[0] != x_size:
        raise ValueError(f"Expected x to have shape ({x_size},), got {x.shape}")
    return x


def coeff_array_from_list(name: str, coeff: list[float]) -> np.ndarray:
    """
    把 profile 系数列表转成受约束的一维数组.

    Args:
        name: profile 名, 仅用于报错上下文.
        coeff: Python 系数列表.

    Returns:
        返回 float64 一维数组. 该数组非空, 并保持原顺序.
    """
    if not isinstance(coeff, list):
        raise TypeError(f"{name} coeff must be list[float] or None, got {type(coeff).__name__}")
    coeff_arr = np.asarray(coeff, dtype=np.float64)
    if coeff_arr.ndim != 1:
        raise ValueError(f"{name} coeff must be 1D, got {coeff_arr.shape}")
    if coeff_arr.size == 0:
        raise ValueError(f"{name} coeff must be non-empty or None")
    return coeff_arr
