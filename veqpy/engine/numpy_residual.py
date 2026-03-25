"""
engine 层 NumPy residual 核.
负责计算 Grad-Shafranov residual 相关场, 并把二维残差投影到一维基函数系数空间.
不负责算子路由, packed layout/codec, solver 收敛控制.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


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


def bind_residual_runner(
    profile_names: tuple[str, ...],
    coeff_index_rows: np.ndarray,
    lengths: np.ndarray,
    residual_size: int,
) -> Callable:
    kernels: list[Callable] = []
    for name in profile_names:
        kernels.append(bind_residual_block(name))
    bound_kernels = tuple(kernels)

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
        for slot, kernel in enumerate(bound_kernels):
            coeff_size = int(lengths[slot])
            kernel(
                out,
                coeff_index_rows[slot, :coeff_size],
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

    np.divide(Z_t, J, out=out_psin_R)
    out_psin_R *= -1.0
    out_psin_R *= psin_r[:, None]

    np.divide(R_t, J, out=out_psin_Z)
    out_psin_Z *= psin_r[:, None]

    psin_r_safe = np.maximum(psin_r, 1e-10)

    G1n = JdivR * (FFn_r[:, None] + R**2 * Pn_r[:, None]) / psin_r_safe[:, None]
    G2n = gttdivJR * psin_rr[:, None] + (gttdivJR_r - grtdivJR_t) * psin_r[:, None]

    out_G[:] = alpha1 * G1n + alpha2 * G2n


@register_residual_block("h")
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
    del psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, R0, B0
    collapsed_rho = np.einsum("ij,ij->i", G, psin_R)
    weighted_rho = collapsed_rho * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * a
    out_packed[coeff_indices] = T[: coeff_indices.shape[0]] @ weighted_rho


@register_residual_block("v")
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
    del psin_R, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, R0, B0
    collapsed_rho = np.einsum("ij,ij->i", G, psin_Z)
    weighted_rho = collapsed_rho * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * a
    out_packed[coeff_indices] = T[: coeff_indices.shape[0]] @ weighted_rho


@register_residual_block("k")
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
    del psin_R, sin_tb, cos_theta, sin_2theta, rho2, R0, B0
    collapsed_rho = np.einsum("ij,ij,j->i", G, psin_Z, sin_theta)
    weighted_rho = collapsed_rho * rho * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * (-a)
    out_packed[coeff_indices] = T[: coeff_indices.shape[0]] @ weighted_rho


@register_residual_block("c0")
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
    del psin_Z, sin_theta, cos_theta, sin_2theta, rho2, R0, B0
    collapsed_rho = np.einsum("ij,ij,ij->i", G, psin_R, sin_tb)
    weighted_rho = collapsed_rho * rho * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * (-a)
    out_packed[coeff_indices] = T[: coeff_indices.shape[0]] @ weighted_rho


@register_residual_block("c1")
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
    del psin_Z, sin_theta, sin_2theta, rho, R0, B0
    collapsed_rho = np.einsum("ij,ij,ij,j->i", G, psin_R, sin_tb, cos_theta)
    weighted_rho = collapsed_rho * rho2 * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * (-a)
    out_packed[coeff_indices] = T[: coeff_indices.shape[0]] @ weighted_rho


@register_residual_block("s1")
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
    del psin_Z, cos_theta, sin_2theta, rho, R0, B0
    collapsed_rho = np.einsum("ij,ij,ij,j->i", G, psin_R, sin_tb, sin_theta)
    weighted_rho = collapsed_rho * rho2 * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * (-a)
    out_packed[coeff_indices] = T[: coeff_indices.shape[0]] @ weighted_rho


@register_residual_block("s2")
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
    del psin_Z, sin_theta, cos_theta, R0, B0
    collapsed_rho = np.einsum("ij,ij,ij,j->i", G, psin_R, sin_tb, sin_2theta)
    weighted_rho = collapsed_rho * rho * rho2 * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * (-a)
    out_packed[coeff_indices] = T[: coeff_indices.shape[0]] @ weighted_rho


@register_residual_block("psin")
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
    del psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, a, R0, B0
    collapsed_rho = np.sum(G, axis=1)
    weighted_rho = collapsed_rho * rho2 * y
    weighted_rho *= weights
    weighted_rho *= 2.0 * np.pi / G.shape[1]
    out_packed[coeff_indices] = T[: coeff_indices.shape[0]] @ weighted_rho


@register_residual_block("F")
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
    del psin_R, psin_Z, sin_tb, sin_theta, cos_theta, sin_2theta, rho, rho2, a
    collapsed_rho = np.sum(G, axis=1)
    weighted_rho = collapsed_rho * y * y
    weighted_rho *= weights
    weighted_rho *= (2.0 * np.pi / G.shape[1]) * (R0 * B0)
    out_packed[coeff_indices] = T[: coeff_indices.shape[0]] @ weighted_rho
