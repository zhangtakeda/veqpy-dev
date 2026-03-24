"""
engine 层 NumPy residual 核.
负责计算 Grad-Shafranov residual 相关场, 并把二维残差投影到一维基函数系数空间.
不负责算子路由, packed layout/codec, solver 收敛控制.
"""

import numpy as np


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

    Returns:
        返回 None. 所有 residual 相关二维场都会原地写入 out_psin_R, out_psin_Z, out_G.
    """
    np.divide(Z_t, J, out=out_psin_R)
    out_psin_R *= -1.0
    out_psin_R *= psin_r[:, None]

    np.divide(R_t, J, out=out_psin_Z)
    out_psin_Z *= psin_r[:, None]

    psin_r_safe = np.maximum(psin_r, 1e-10)

    G1n = JdivR * (FFn_r[:, None] + R**2 * Pn_r[:, None]) / psin_r_safe[:, None]
    G2n = gttdivJR * psin_rr[:, None] + (gttdivJR_r - grtdivJR_t) * psin_r[:, None]

    out_G[:] = alpha1 * G1n + alpha2 * G2n


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

    Returns:
        返回 None. 组装后的 h 通道投影会原地写入 out.
    """
    collapsed_rho = np.einsum("ij,ij->i", G, psin_R)
    weighted_rho = collapsed_rho * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * a
    out[:] = T[: out.shape[0]] @ weighted_rho


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

    Returns:
        返回 None. 组装后的 v 通道投影会原地写入 out.
    """
    collapsed_rho = np.einsum("ij,ij->i", G, psin_Z)
    weighted_rho = collapsed_rho * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * a
    out[:] = T[: out.shape[0]] @ weighted_rho


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

    Returns:
        返回 None. 组装后的 k 通道投影会原地写入 out.
    """
    collapsed_rho = np.einsum("ij,ij,j->i", G, psin_Z, sin_theta)
    weighted_rho = collapsed_rho * rho * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * (-a)
    out[:] = T[: out.shape[0]] @ weighted_rho


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

    Returns:
        返回 None. 组装后的 c0 通道投影会原地写入 out.
    """
    collapsed_rho = np.einsum("ij,ij,ij->i", G, psin_R, sin_tb)
    weighted_rho = collapsed_rho * rho * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * (-a)
    out[:] = T[: out.shape[0]] @ weighted_rho


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

    Returns:
        返回 None. 组装后的 c1 通道投影会原地写入 out.
    """
    collapsed_rho = np.einsum("ij,ij,ij,j->i", G, psin_R, sin_tb, cos_theta)
    weighted_rho = collapsed_rho * rho2 * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * (-a)
    out[:] = T[: out.shape[0]] @ weighted_rho


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

    Returns:
        返回 None. 组装后的 s1 通道投影会原地写入 out.
    """
    collapsed_rho = np.einsum("ij,ij,ij,j->i", G, psin_R, sin_tb, sin_theta)
    weighted_rho = collapsed_rho * rho2 * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * (-a)
    out[:] = T[: out.shape[0]] @ weighted_rho


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

    Returns:
        返回 None. 组装后的 s2 通道投影会原地写入 out.
    """
    collapsed_rho = np.einsum("ij,ij,ij,j->i", G, psin_R, sin_tb, sin_2theta)
    weighted_rho = collapsed_rho * rho * rho2 * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * (-a)
    out[:] = T[: out.shape[0]] @ weighted_rho


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

    Returns:
        返回 None. 组装后的 psin 通道投影会原地写入 out.
    """
    collapsed_rho = np.sum(G, axis=1)
    weighted_rho = collapsed_rho * rho2 * y
    weighted_rho *= weights
    weighted_rho *= 2.0 * np.pi / G.shape[1]
    out[:] = T[: out.shape[0]] @ weighted_rho


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

    Returns:
        返回 None. 组装后的 F 通道投影会原地写入 out.
    """
    collapsed_rho = np.sum(G, axis=1)
    weighted_rho = collapsed_rho * y * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * (R0 * B0)
    out[:] = T[: out.shape[0]] @ weighted_rho
