"""
engine 层 Numba residual 核.
负责计算 Grad-Shafranov residual 相关场, 并把二维残差投影到一维基函数系数空间.
不负责算子路由, packed layout/codec, solver 收敛控制.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def update_residual(
    out_psin_R: np.ndarray,
    out_psin_Z: np.ndarray,
    out_G: np.ndarray,
    alpha1: float,
    alpha2: float,
    psin_r: np.ndarray,
    psin_rr: np.ndarray,
    FFn_r: np.ndarray,
    Pn_r: np.ndarray,
    R: np.ndarray,
    R_t: np.ndarray,
    Z_t: np.ndarray,
    J: np.ndarray,
    JdivR: np.ndarray,
    gttdivJR: np.ndarray,
    grtdivJR_t: np.ndarray,
    gttdivJR_r: np.ndarray,
) -> None:
    """
    原地更新 residual 相关二维场.

    Args:
        out_psin_R, out_psin_Z, out_G: 调用方持有的二维输出缓冲区, shape=(nr, nt).
        alpha1, alpha2: source 与几何项的归一化系数.
        psin_r, psin_rr, FFn_r, Pn_r: 当前 grid 上的一维 root fields, shape=(nr,).
        R, R_t, Z_t, J, JdivR, gttdivJR, grtdivJR_t, gttdivJR_r: 当前几何场及其组合量, shape=(nr, nt).
    """
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


@njit(cache=True, fastmath=True, nogil=True)
def assemble_h_residual_block(
    out: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
) -> None:
    """
    组装 h 通道的一维 residual 投影块.

    Args:
        out: 输出系数缓冲区, shape=(n_basis,).
        G, psin_R: 二维 residual 相关场, shape=(nr, nt).
        y, weights: 径向权重项, shape=(nr,).
        T: 基函数矩阵, shape=(n_basis, nr).
        a: 小半径尺度, 与几何单位一致.
    """
    _assemble_weighted_projection(out, G, psin_R, y, T, weights, (2.0 * np.pi / G.shape[1]) * a)


@njit(cache=True, fastmath=True, nogil=True)
def assemble_v_residual_block(
    out: np.ndarray,
    G: np.ndarray,
    psin_Z: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
) -> None:
    """
    组装 v 通道的一维 residual 投影块.

    Args:
        out: 输出系数缓冲区, shape=(n_basis,).
        G, psin_Z: 二维 residual 相关场, shape=(nr, nt).
        y, weights: 径向权重项, shape=(nr,).
        T: 基函数矩阵, shape=(n_basis, nr).
        a: 小半径尺度, 与几何单位一致.
    """
    _assemble_weighted_projection(out, G, psin_Z, y, T, weights, (2.0 * np.pi / G.shape[1]) * a)


@njit(cache=True, fastmath=True, nogil=True)
def assemble_k_residual_block(
    out: np.ndarray,
    G: np.ndarray,
    psin_Z: np.ndarray,
    sin_theta: np.ndarray,
    rho: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
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
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = (2.0 * np.pi / nt) * (-a)
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * psin_Z[i, j] * sin_theta[j]
        weighted_rho[i] = collapsed * rho[i] * y[i] * weights[i] * scale
    _project_rows(out, T, weighted_rho)


@njit(cache=True, fastmath=True, nogil=True)
def assemble_c0_residual_block(
    out: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    sin_tb: np.ndarray,
    rho: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
) -> None:
    """
    组装 c0 通道的一维 residual 投影块.

    Args:
        out: 输出系数缓冲区, shape=(n_basis,).
        G, psin_R, sin_tb: 二维 residual 相关场, shape=(nr, nt).
        rho, y, weights: 径向权重项, shape=(nr,).
        T: 基函数矩阵, shape=(n_basis, nr).
        a: 小半径尺度, 与几何单位一致.
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = (2.0 * np.pi / nt) * (-a)
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * psin_R[i, j] * sin_tb[i, j]
        weighted_rho[i] = collapsed * rho[i] * y[i] * weights[i] * scale
    _project_rows(out, T, weighted_rho)


@njit(cache=True, fastmath=True, nogil=True)
def assemble_c1_residual_block(
    out: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    sin_tb: np.ndarray,
    cos_theta: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
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
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = (2.0 * np.pi / nt) * (-a)
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * psin_R[i, j] * sin_tb[i, j] * cos_theta[j]
        weighted_rho[i] = collapsed * rho2[i] * y[i] * weights[i] * scale
    _project_rows(out, T, weighted_rho)


@njit(cache=True, fastmath=True, nogil=True)
def assemble_s1_residual_block(
    out: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    sin_tb: np.ndarray,
    sin_theta: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
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
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = (2.0 * np.pi / nt) * (-a)
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * psin_R[i, j] * sin_tb[i, j] * sin_theta[j]
        weighted_rho[i] = collapsed * rho2[i] * y[i] * weights[i] * scale
    _project_rows(out, T, weighted_rho)


@njit(cache=True, fastmath=True, nogil=True)
def assemble_s2_residual_block(
    out: np.ndarray,
    G: np.ndarray,
    psin_R: np.ndarray,
    sin_tb: np.ndarray,
    sin_2theta: np.ndarray,
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
    a: float,
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
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = (2.0 * np.pi / nt) * (-a)
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j] * psin_R[i, j] * sin_tb[i, j] * sin_2theta[j]
        weighted_rho[i] = collapsed * rho[i] * rho2[i] * y[i] * weights[i] * scale
    _project_rows(out, T, weighted_rho)


@njit(cache=True, fastmath=True, nogil=True)
def assemble_psin_residual_block(
    out: np.ndarray,
    G: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
) -> None:
    """
    组装 psin 通道的一维 residual 投影块.

    Args:
        out: 输出系数缓冲区, shape=(n_basis,).
        G: 二维 residual 场, shape=(nr, nt).
        rho2, y, weights: 径向权重项, shape=(nr,).
        T: 基函数矩阵, shape=(n_basis, nr).
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = 2.0 * np.pi / nt
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j]
        weighted_rho[i] = collapsed * rho2[i] * y[i] * weights[i] * scale
    _project_rows(out, T, weighted_rho)


@njit(cache=True, fastmath=True, nogil=True)
def assemble_F_residual_block(
    out: np.ndarray,
    G: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    weights: np.ndarray,
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
    """
    nr, nt = G.shape
    weighted_rho = np.empty(nr, dtype=G.dtype)
    scale = (2.0 * np.pi / nt) * (R0 * B0)
    for i in range(nr):
        collapsed = 0.0
        for j in range(nt):
            collapsed += G[i, j]
        weighted_rho[i] = collapsed * y[i] * y[i] * weights[i] * scale
    _project_rows(out, T, weighted_rho)


@njit(cache=True, fastmath=True, nogil=True)
def _assemble_weighted_projection(
    out: np.ndarray,
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
    _project_rows(out, T, weighted_rho)


@njit(cache=True, fastmath=True, nogil=True)
def _project_rows(out: np.ndarray, T: np.ndarray, weighted_rho: np.ndarray) -> None:
    rows = out.shape[0]
    cols = weighted_rho.shape[0]
    for i in range(rows):
        total = 0.0
        for j in range(cols):
            total += T[i, j] * weighted_rho[j]
        out[i] = total
