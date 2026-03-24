"""
operator 层 packed codec.
负责在 operator 边界编码和解码 packed 状态向量, 并在 residual 组装后回收 packed 残差.
不负责 layout 定义, source 路由, solver 终止准则.
"""

from typing import Protocol

import numpy as np

from veqpy.operator.layout import (
    PROFILE_NAMES,
    coeff_array_from_list,
    packed_size,
    validate_packed_state,
)


class ResidualAssembleSlot(Protocol):
    """描述一个 residual block 写回 packed 向量所需的最小接口."""

    coeff_indices: np.ndarray
    kernel: object


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
    G: np.ndarray,
    psin_R: np.ndarray,
    psin_Z: np.ndarray,
    sin_tb: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_2theta: np.ndarray,
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
    R0: float,
    B0: float,
) -> np.ndarray:
    """
    调用各 residual slot 并收集 packed 残差向量.

    Args:
        residual_slots: 各 active profile 对应的 residual 组装槽位.
        residual_size: packed 残差向量长度.
        G, psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta:
            当前 residual 与几何相关场.
        rho, rho2, y, T, weights:
            当前 grid 上的谱投影输入.
        a, R0, B0:
            当前 case 的标量尺度.

    Returns:
        返回 packed residual 向量. 各 slot 会直接按 coeff_indices 原地写入对应 packed 位置.
    """
    out = np.zeros(residual_size, dtype=np.float64)
    for slot in residual_slots:
        slot.kernel(
            out,
            slot.coeff_indices,
            G,
            psin_R,
            psin_Z,
            sin_tb,
            sin_theta,
            cos_theta,
            sin_2theta,
            rho,
            rho2,
            y,
            T,
            weights,
            a,
            R0,
            B0,
        )
    return out

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
