"""
engine 层 Numba residual 核.
负责计算 Grad-Shafranov residual 相关场, 并把二维残差投影到一维基函数系数空间.
不负责算子路由, packed layout/codec, solver 收敛控制.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numba import njit


@dataclass(frozen=True, slots=True)
class _ResidualBlockSpec:
    name: str
    implementation: Callable


RESIDUAL_BLOCK_REGISTRY: dict[str, _ResidualBlockSpec] = {}


def register_residual_block(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        existing = RESIDUAL_BLOCK_REGISTRY.get(name)
        if existing is not None:
            raise ValueError(f"Residual block {name!r} is already registered")
        RESIDUAL_BLOCK_REGISTRY[name] = _ResidualBlockSpec(name=name, implementation=func)
        return func

    return decorator


def bind_residual_block(name: str) -> Callable:
    try:
        spec = RESIDUAL_BLOCK_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(RESIDUAL_BLOCK_REGISTRY)
        raise KeyError(f"Unknown residual block {name!r}. Supported blocks: {supported}") from exc
    return spec.implementation


_BLOCK_CODE_BY_NAME = {
    "h": 0,
    "v": 1,
    "k": 2,
    "c0": 3,
    "c1": 4,
    "s1": 5,
    "s2": 6,
    "psin": 7,
    "F": 8,
}


def bind_residual_runner(
    profile_names: tuple[str, ...],
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_size: int,
) -> Callable:
    block_codes = np.empty(len(profile_names), dtype=np.int64)
    for i, name in enumerate(profile_names):
        bind_residual_block(name)
        try:
            block_codes[i] = _BLOCK_CODE_BY_NAME[name]
        except KeyError as exc:
            supported = ", ".join(_BLOCK_CODE_BY_NAME)
            raise KeyError(f"Unknown residual block {name!r}. Supported blocks: {supported}") from exc

    def runner(
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
        out = np.zeros(residual_size, dtype=np.float64)
        _run_residual_blocks_packed(
            out,
            block_codes,
            coeff_index_rows,
            lengths,
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

    return runner


@njit(cache=True, fastmath=True, nogil=True)
def update_residual(
    out_fields: np.ndarray,
    alpha1: float,
    alpha2: float,
    root_fields: np.ndarray,
    R_fields: np.ndarray,
    Z_fields: np.ndarray,
    J_fields: np.ndarray,
    g_fields: np.ndarray,
) -> None:
    """
    原地更新 residual 相关二维场.

    Args:
        out_fields: 调用方持有的二维输出 fields, shape=(3, nr, nt).
        alpha1, alpha2: source 与几何项的归一化系数.
        root_fields: 当前 grid 上的一维 root fields, shape=(4, nr).
        R_fields, Z_fields, J_fields, g_fields: 当前几何 packed fields.

    Returns:
        返回 None. 所有 residual 相关二维场都会原地写入 out_fields.
    """
    out_psin_R = out_fields[0]
    out_psin_Z = out_fields[1]
    out_G = out_fields[2]

    psin_r = root_fields[0]
    psin_rr = root_fields[1]
    FFn_r = root_fields[2]
    Pn_r = root_fields[3]

    R = R_fields[0]
    R_t = R_fields[2]
    Z_t = Z_fields[2]
    J = J_fields[0]
    JdivR = J_fields[6]
    grtdivJR_t = g_fields[2]
    gttdivJR = g_fields[5]
    gttdivJR_r = g_fields[6]

    nr, nt = out_G.shape
    for i in range(nr):
        psin_r_safe = psin_r[i]
        if psin_r_safe < 1e-10:
            psin_r_safe = 1e-10
        for j in range(nt):
            inv_J = 1.0 / J[i, j]
            psin_R = -Z_t[i, j] * inv_J * psin_r[i]
            psin_Z = R_t[i, j] * inv_J * psin_r[i]
            out_psin_R[i, j] = psin_R
            out_psin_Z[i, j] = psin_Z

            G1n = JdivR[i, j] * (FFn_r[i] + R[i, j] * R[i, j] * Pn_r[i]) / psin_r_safe
            G2n = gttdivJR[i, j] * psin_rr[i] + (gttdivJR_r[i, j] - grtdivJR_t[i, j]) * psin_r[i]
            out_G[i, j] = alpha1 * G1n + alpha2 * G2n


@register_residual_block("h")
@njit(cache=True, fastmath=True, nogil=True)
def assemble_h_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
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
) -> None:
    """
    组装 h 通道的一维 residual 投影块.

    Args:
        out: 输出系数缓冲区, shape=(n_basis,).
        G, psin_R: 二维 residual 相关场, shape=(nr, nt).
        y, weights: 径向权重项, shape=(nr,).
        T: 基函数矩阵, shape=(n_basis, nr).
        a: 小半径尺度, 与几何单位一致.

    Returns:
        返回 None. 组装后的 h 通道投影会原地写入 out.
    """
    _assemble_weighted_projection(out_packed, coeff_indices, G, psin_R, y, T, weights, (2.0 * np.pi / G.shape[1]) * a)


@register_residual_block("v")
@njit(cache=True, fastmath=True, nogil=True)
def assemble_v_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
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
) -> None:
    """
    组装 v 通道的一维 residual 投影块.

    Args:
        out: 输出系数缓冲区, shape=(n_basis,).
        G, psin_Z: 二维 residual 相关场, shape=(nr, nt).
        y, weights: 径向权重项, shape=(nr,).
        T: 基函数矩阵, shape=(n_basis, nr).
        a: 小半径尺度, 与几何单位一致.

    Returns:
        返回 None. 组装后的 v 通道投影会原地写入 out.
    """
    _assemble_weighted_projection(out_packed, coeff_indices, G, psin_Z, y, T, weights, (2.0 * np.pi / G.shape[1]) * a)


@register_residual_block("k")
@njit(cache=True, fastmath=True, nogil=True)
def assemble_k_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
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
) -> None:
    """
    组装 k 通道的一维 residual 投影块.

    Args:
        out: 输出系数缓冲区, shape=(n_basis,).
        G, psin_Z: 二维 residual 相关场, shape=(nr, nt).
        sin_theta: theta 基函数取值, shape=(nt,).
        rho, y, weights: 径向权重项, shape=(nr,).
        T: 基函数矩阵, shape=(n_basis, nr).
        a: 小半径尺度, 与几何单位一致.

    Returns:
        返回 None. 组装后的 k 通道投影会原地写入 out.
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = (2.0 * np.pi / nt) * (-a)
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * psin_Z[i, j] * sin_theta[j]
        weighted_rho[i] = collapsed * rho[i] * y[i] * weights[i] * scale
    _project_rows_to_packed(out_packed, coeff_indices, T, weighted_rho)


@register_residual_block("c0")
@njit(cache=True, fastmath=True, nogil=True)
def assemble_c0_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
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
) -> None:
    """
    组装 c0 通道的一维 residual 投影块.

    Args:
        out: 输出系数缓冲区, shape=(n_basis,).
        G, psin_R, sin_tb: 二维 residual 相关场, shape=(nr, nt).
        rho, y, weights: 径向权重项, shape=(nr,).
        T: 基函数矩阵, shape=(n_basis, nr).
        a: 小半径尺度, 与几何单位一致.

    Returns:
        返回 None. 组装后的 c0 通道投影会原地写入 out.
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = (2.0 * np.pi / nt) * (-a)
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * psin_R[i, j] * sin_tb[i, j]
        weighted_rho[i] = collapsed * rho[i] * y[i] * weights[i] * scale
    _project_rows_to_packed(out_packed, coeff_indices, T, weighted_rho)


@register_residual_block("c1")
@njit(cache=True, fastmath=True, nogil=True)
def assemble_c1_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
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
) -> None:
    """
    组装 c1 通道的一维 residual 投影块.

    Args:
        out: 输出系数缓冲区, shape=(n_basis,).
        G, psin_R, sin_tb: 二维 residual 相关场, shape=(nr, nt).
        cos_theta: theta 基函数取值, shape=(nt,).
        rho2, y, weights: 径向权重项, shape=(nr,).
        T: 基函数矩阵, shape=(n_basis, nr).
        a: 小半径尺度, 与几何单位一致.

    Returns:
        返回 None. 组装后的 c1 通道投影会原地写入 out.
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = (2.0 * np.pi / nt) * (-a)
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * psin_R[i, j] * sin_tb[i, j] * cos_theta[j]
        weighted_rho[i] = collapsed * rho2[i] * y[i] * weights[i] * scale
    _project_rows_to_packed(out_packed, coeff_indices, T, weighted_rho)


@register_residual_block("s1")
@njit(cache=True, fastmath=True, nogil=True)
def assemble_s1_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
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
) -> None:
    """
    组装 s1 通道的一维 residual 投影块.

    Args:
        out: 输出系数缓冲区, shape=(n_basis,).
        G, psin_R, sin_tb: 二维 residual 相关场, shape=(nr, nt).
        sin_theta: theta 基函数取值, shape=(nt,).
        rho2, y, weights: 径向权重项, shape=(nr,).
        T: 基函数矩阵, shape=(n_basis, nr).
        a: 小半径尺度, 与几何单位一致.

    Returns:
        返回 None. 组装后的 s1 通道投影会原地写入 out.
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = (2.0 * np.pi / nt) * (-a)
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * psin_R[i, j] * sin_tb[i, j] * sin_theta[j]
        weighted_rho[i] = collapsed * rho2[i] * y[i] * weights[i] * scale
    _project_rows_to_packed(out_packed, coeff_indices, T, weighted_rho)


@register_residual_block("s2")
@njit(cache=True, fastmath=True, nogil=True)
def assemble_s2_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
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
) -> None:
    """
    组装 s2 通道的一维 residual 投影块.

    Args:
        out: 输出系数缓冲区, shape=(n_basis,).
        G, psin_R, sin_tb: 二维 residual 相关场, shape=(nr, nt).
        sin_2theta: 二倍角 theta 基函数取值, shape=(nt,).
        rho, rho2, y, weights: 径向权重项, shape=(nr,).
        T: 基函数矩阵, shape=(n_basis, nr).
        a: 小半径尺度, 与几何单位一致.

    Returns:
        返回 None. 组装后的 s2 通道投影会原地写入 out.
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = (2.0 * np.pi / nt) * (-a)
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * psin_R[i, j] * sin_tb[i, j] * sin_2theta[j]
        weighted_rho[i] = collapsed * rho[i] * rho2[i] * y[i] * weights[i] * scale
    _project_rows_to_packed(out_packed, coeff_indices, T, weighted_rho)


@register_residual_block("psin")
@njit(cache=True, fastmath=True, nogil=True)
def assemble_psin_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
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
) -> None:
    """
    组装 psin 通道的一维 residual 投影块.

    Args:
        out: 输出系数缓冲区, shape=(n_basis,).
        G: 二维 residual 场, shape=(nr, nt).
        rho2, y, weights: 径向权重项, shape=(nr,).
        T: 基函数矩阵, shape=(n_basis, nr).

    Returns:
        返回 None. 组装后的 psin 通道投影会原地写入 out.
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = 2.0 * np.pi / nt
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j]
        weighted_rho[i] = collapsed * rho2[i] * y[i] * weights[i] * scale
    _project_rows_to_packed(out_packed, coeff_indices, T, weighted_rho)


@register_residual_block("F")
@njit(cache=True, fastmath=True, nogil=True)
def assemble_F_residual_block(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
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
) -> None:
    """
    组装 F 通道的一维 residual 投影块.

    Args:
        out: 输出系数缓冲区, shape=(n_basis,).
        G: 二维 residual 场, shape=(nr, nt).
        y, weights: 径向权重项, shape=(nr,).
        T: 基函数矩阵, shape=(n_basis, nr).
        R0, B0: 参考磁轴半径与磁场强度, 用于 F 通道归一化.

    Returns:
        返回 None. 组装后的 F 通道投影会原地写入 out.
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = (2.0 * np.pi / nt) * (R0 * B0)
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j]
        weighted_rho[i] = collapsed * y[i] * y[i] * weights[i] * scale
    _project_rows_to_packed(out_packed, coeff_indices, T, weighted_rho)


@njit(cache=True, fastmath=True, nogil=True)
def _assemble_weighted_projection(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    G: np.ndarray,
    field: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    scale: float,
) -> None:
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * field[i, j]
        weighted_rho[i] = collapsed * y[i] * weights[i] * scale
    _project_rows_to_packed(out_packed, coeff_indices, T, weighted_rho)


@njit(cache=True, fastmath=True, nogil=True)
def _project_rows_to_packed(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    T: np.ndarray,
    weighted_rho: np.ndarray,
) -> None:
    rows = coeff_indices.shape[0]
    cols = weighted_rho.shape[0]
    for i in range(rows):
        total = 0.0
        for j in range(cols):
            total += T[i, j] * weighted_rho[j]
        out_packed[coeff_indices[i]] = total


@njit(cache=True, fastmath=True, nogil=True)
def _run_residual_blocks_packed(
    out_packed: np.ndarray,
    block_codes: np.ndarray,
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
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
) -> None:
    for slot in range(block_codes.shape[0]):
        coeff_indices = coeff_index_rows[slot, : lengths[slot]]
        code = block_codes[slot]
        if code == 0:
            assemble_h_residual_block(out_packed, coeff_indices, G, psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, y, T, weights, a, R0, B0)
        elif code == 1:
            assemble_v_residual_block(out_packed, coeff_indices, G, psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, y, T, weights, a, R0, B0)
        elif code == 2:
            assemble_k_residual_block(out_packed, coeff_indices, G, psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, y, T, weights, a, R0, B0)
        elif code == 3:
            assemble_c0_residual_block(out_packed, coeff_indices, G, psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, y, T, weights, a, R0, B0)
        elif code == 4:
            assemble_c1_residual_block(out_packed, coeff_indices, G, psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, y, T, weights, a, R0, B0)
        elif code == 5:
            assemble_s1_residual_block(out_packed, coeff_indices, G, psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, y, T, weights, a, R0, B0)
        elif code == 6:
            assemble_s2_residual_block(out_packed, coeff_indices, G, psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, y, T, weights, a, R0, B0)
        elif code == 7:
            assemble_psin_residual_block(out_packed, coeff_indices, G, psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, y, T, weights, a, R0, B0)
        else:
            assemble_F_residual_block(out_packed, coeff_indices, G, psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, y, T, weights, a, R0, B0)
