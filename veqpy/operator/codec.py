"""
operator 层 packed codec.
负责在 operator 边界编码和解码 packed 状态向量, 并在 residual 组装后回收 packed 残差.
不负责 layout 定义, source 路由, solver 终止准则.
"""

from typing import Protocol

import numpy as np

from veqpy.operator.layout import (
    PROFILE_COUNT,
    PROFILE_NAMES,
    coeff_array_from_list,
    packed_size,
    validate_packed_state,
)


class ResidualAssembleSlot(Protocol):
    """描述一个 residual block 写回 packed 向量所需的最小接口."""

    coeff_row: np.ndarray
    coeff_indices: np.ndarray

    def assemble(self) -> None: ...


def encode_packed_state(
    coeffs_by_name: dict[str, list[float] | None],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
) -> np.ndarray:
    """
    按 layout 把 profile 系数字典编码成 packed 状态向量.

    Args:
        coeffs_by_name: profile 名到系数列表的映射.
        profile_L: 各 profile 的最高阶数向量.
        coeff_index: 当前 layout 的 packed 索引矩阵.

    Returns:
        返回一维 packed 状态向量, shape=(packed_size(coeff_index),).
    """
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


def encode_packed_residual(
    residual_slots: tuple[ResidualAssembleSlot, ...],
    residual_size: int,
) -> np.ndarray:
    """
    调用各 residual slot 并收集 packed 残差向量.

    Args:
        residual_slots: 各 active profile 对应的 residual 组装槽位.
        residual_size: packed 残差向量长度.

    Returns:
        返回 packed residual 向量. 各 slot 会先原地更新 coeff_row, 再写回对应索引.
    """
    out = np.zeros(residual_size, dtype=np.float64)
    for slot in residual_slots:
        slot.assemble()
        out[slot.coeff_indices] = slot.coeff_row
    return out


def decode_packed_state_inplace(
    x: np.ndarray,
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    coeff_matrix: np.ndarray,
) -> None:
    """
    把 packed 状态向量解码到 profile 系数矩阵中.

    Args:
        x: 一维 packed 状态向量.
        profile_L: 各 profile 的最高阶数向量.
        coeff_index: 当前 layout 的 packed 索引矩阵.
        coeff_matrix: 调用方持有的输出系数矩阵, shape 必须与 coeff_index 一致.

    Returns:
        返回 None. 解码结果会原地写入 coeff_matrix.
    """
    x = validate_packed_state(x, coeff_index)
    if coeff_matrix.shape != coeff_index.shape:
        raise ValueError(f"Expected coeff_matrix shape {coeff_index.shape}, got {coeff_matrix.shape}")

    coeff_matrix.fill(0.0)
    for p in range(PROFILE_COUNT):
        L = int(profile_L[p])
        if L < 0:
            continue
        for k in range(L + 1):
            coeff_matrix[p, k] = x[coeff_index[p, k]]


def decode_packed_blocks(
    x: np.ndarray,
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
) -> tuple[np.ndarray | None, ...]:
    """
    把 packed 状态向量解码成按 profile 分块的系数副本.

    Args:
        x: 一维 packed 状态向量.
        profile_L: 各 profile 的最高阶数向量.
        coeff_index: 当前 layout 的 packed 索引矩阵.

    Returns:
        返回长度为 PROFILE_COUNT 的 tuple.
        inactive profile 位置为 None, active profile 位置为对应系数块副本.
    """
    x = validate_packed_state(x, coeff_index)

    coeff_matrix = np.zeros_like(coeff_index, dtype=np.float64)
    decode_packed_state_inplace(x, profile_L, coeff_index, coeff_matrix)

    blocks: list[np.ndarray | None] = []
    for p in range(PROFILE_COUNT):
        L = int(profile_L[p])
        if L < 0:
            blocks.append(None)
        else:
            blocks.append(coeff_matrix[p, : L + 1].copy())
    return tuple(blocks)
