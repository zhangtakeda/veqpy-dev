"""
Module: operator.codec

Role:
- 负责在 operator 边界编码和解码 packed 状态向量.

Public API:
- encode_packed_state
- decode_packed_blocks

Notes:
- 这里只处理 packed codec.
- 不负责 layout 定义, source 路由, 或 solver 终止准则.
"""

import numpy as np

from veqpy.operator.layout import (
    PROFILE_NAMES,
    coeff_array_from_list,
    packed_size,
    validate_packed_state,
)


def encode_packed_state(
    coeffs_by_name: dict[str, list[float] | None],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
) -> np.ndarray:
    """按 layout 把 profile 系数字典编码成 packed 状态向量."""
    x = np.empty(packed_size(coeff_index), dtype=np.float64)

    for p, name in enumerate(PROFILE_NAMES):
        L = int(profile_L[p])
        coeff = coeffs_by_name.get(name)

        if L < 0:
            continue
        if coeff is None:
            raise ValueError(f"{name} is active in layout but coeff is None")

        coeff_arr = coeff_array_from_list(name, coeff)
        if coeff_arr.size != L + 1:
            raise ValueError(f"{name} coeff shape mismatch: expected {(L + 1,)}, got {coeff_arr.shape}")

        for k in range(L + 1):
            x[coeff_index[p, k]] = coeff_arr[k]

    return x


def decode_packed_blocks(
    x: np.ndarray,
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
) -> tuple[np.ndarray | None, ...]:
    """把 packed 状态向量解码成按 profile 分块的系数副本."""
    x = validate_packed_state(x, coeff_index)

    blocks: list[np.ndarray | None] = []
    for p, _ in enumerate(PROFILE_NAMES):
        L = int(profile_L[p])
        if L < 0:
            blocks.append(None)
        else:
            block = np.empty(L + 1, dtype=np.float64)
            for k in range(L + 1):
                block[k] = x[coeff_index[p, k]]
            blocks.append(block)
    return tuple(blocks)
