"""
Module: operator.runtime_layout

Role:
- 定义 operator 运行期使用的三层 layout 容器.
- 定义 operator 静态 grid layout 与 residual binding layout.
- 兼容导出 workspace runtime memory/view 符号.

Public API:
- StaticLayout
- ResidualBindingLayout
- BackendState (compat re-export from veqpy.workspace)
- FieldRuntimeState (compat re-export from veqpy.workspace)
- SourceRuntimeState (compat re-export from veqpy.workspace)
- SourceConstState (compat re-export from veqpy.workspace)
- SourceWorkState (compat re-export from veqpy.workspace)
- SourceAuxState (compat re-export from veqpy.workspace)

Notes:
- 运行期内存 ownership 和分配逻辑在 ``veqpy.workspace``.
- 不负责具体数值核计算或 solver 迭代控制.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from veqpy.workspace.runtime import (
    BackendState,
    FieldRuntimeState,
    GeometryWorkspace,
    OperatorWorkspace,
    ProfileWorkspace,
    ResidualWorkspace,
    RuntimeAllocationBundle,
    SourceAuxState,
    SourceConstState,
    SourceRuntimeState,
    SourceWorkspace,
    SourceWorkState,
    allocate_runtime_state,
)

if TYPE_CHECKING:
    from veqpy.model.grid import Grid

__all__ = [
    "BackendState",
    "FieldRuntimeState",
    "GeometryWorkspace",
    "ProfileWorkspace",
    "ResidualBindingLayout",
    "ResidualWorkspace",
    "OperatorWorkspace",
    "RuntimeAllocationBundle",
    "SourceWorkspace",
    "SourceAuxState",
    "SourceConstState",
    "SourceRuntimeState",
    "SourceWorkState",
    "StaticLayout",
    "_pack_poloidal_block",
    "_pack_radial_block",
    "allocate_runtime_state",
]


# ---- block layout convention -----------------------------------------------
#   radial_block (8+K_max+3*L_max, Nr)
#     rho           (Nr,)
#     x             (Nr,)
#     y             (Nr,)
#     rho_powers    (K_max+2, Nr)
#     T             (L_max+1, Nr)
#     T_r           (L_max+1, Nr)
#     T_rr          (L_max+1, Nr)
#
#   poloidal_block (7+6*M_max, Nt)
#     theta         (Nt,)
#     cos_mtheta    (M_max+1, Nt)
#     sin_mtheta    (M_max+1, Nt)
#     m_cos_mtheta  (M_max+1, Nt)
#     m_sin_mtheta  (M_max+1, Nt)
#     m2_cos_mtheta (M_max+1, Nt)
#     m2_sin_mtheta (M_max+1, Nt)
#


def _pack_radial_block(
    rho: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    rho_powers: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
) -> np.ndarray:
    """按规范压合径向场为 (R, Nr) 2D 只读数组."""
    if abs(rho[0]) < 1e-10:
        raise ValueError("rho[0] is too close to zero")

    Nr = rho.shape[0]
    K_max = rho_powers.shape[0] - 2
    L_max = T.shape[0] - 1

    block = np.empty((8 + K_max + 3 * L_max, Nr), dtype=np.float64)
    block[0] = rho
    block[1] = x
    block[2] = y
    block[3 : 5 + K_max] = rho_powers
    block[5 + K_max : 6 + K_max + L_max] = T
    block[6 + K_max + L_max : 7 + K_max + 2 * L_max] = T_r
    block[7 + K_max + 2 * L_max : 8 + K_max + 3 * L_max] = T_rr
    block.flags.writeable = False
    return block


def _pack_poloidal_block(
    theta: np.ndarray,
    cos_mtheta: np.ndarray,
    sin_mtheta: np.ndarray,
    m_cos_mtheta: np.ndarray,
    m_sin_mtheta: np.ndarray,
    m2_cos_mtheta: np.ndarray,
    m2_sin_mtheta: np.ndarray,
) -> np.ndarray:
    """按规范压合极向场为 (P, Nt) 2D 只读数组."""
    Nt = theta.shape[0]
    M_max = cos_mtheta.shape[0] - 1

    block = np.empty((7 + 6 * M_max, Nt), dtype=np.float64)
    block[0] = theta
    block[1 : 2 + M_max] = cos_mtheta
    block[2 + M_max : 3 + 2 * M_max] = sin_mtheta
    block[3 + 2 * M_max : 4 + 3 * M_max] = m_cos_mtheta
    block[4 + 3 * M_max : 5 + 4 * M_max] = m_sin_mtheta
    block[5 + 4 * M_max : 6 + 5 * M_max] = m2_cos_mtheta
    block[6 + 5 * M_max : 7 + 6 * M_max] = m2_sin_mtheta
    block.flags.writeable = False
    return block


@dataclass(frozen=True, slots=True)
class StaticLayout:
    """operator 层 Grid 快照 — engine ABI 所需的数组+元数据.

    K_max 已规范化：Grid.K_max=None 映射为 M_max.
    """

    Nr: int
    Nt: int
    M_max: int
    L_max: int
    K_max: int
    quadrature_scheme: str
    calculus_scheme: str
    K_values: np.ndarray
    weights: np.ndarray
    differentiator: np.ndarray
    accumulator: np.ndarray
    radial_block: np.ndarray  # (8+K_max+3*L_max, Nr)
    poloidal_block: np.ndarray  # (7+6*M_max, Nt)

    @property
    def rho(self) -> np.ndarray:
        return self.radial_block[0]

    @property
    def x(self) -> np.ndarray:
        return self.radial_block[1]

    @property
    def y(self) -> np.ndarray:
        return self.radial_block[2]

    @property
    def rho_powers(self) -> np.ndarray:
        return self.radial_block[3 : 5 + self.K_max]

    @property
    def T(self) -> np.ndarray:
        return self.radial_block[5 + self.K_max : 6 + self.K_max + self.L_max]

    @property
    def T_r(self) -> np.ndarray:
        return self.radial_block[6 + self.K_max + self.L_max : 7 + self.K_max + 2 * self.L_max]

    @property
    def T_rr(self) -> np.ndarray:
        return self.radial_block[7 + self.K_max + 2 * self.L_max : 8 + self.K_max + 3 * self.L_max]

    @property
    def theta(self) -> np.ndarray:
        return self.poloidal_block[0]

    @property
    def cos_mtheta(self) -> np.ndarray:
        return self.poloidal_block[1 : 2 + self.M_max]

    @property
    def sin_mtheta(self) -> np.ndarray:
        return self.poloidal_block[2 + self.M_max : 3 + 2 * self.M_max]

    @property
    def m_cos_mtheta(self) -> np.ndarray:
        return self.poloidal_block[3 + 2 * self.M_max : 4 + 3 * self.M_max]

    @property
    def m_sin_mtheta(self) -> np.ndarray:
        return self.poloidal_block[4 + 3 * self.M_max : 5 + 4 * self.M_max]

    @property
    def m2_cos_mtheta(self) -> np.ndarray:
        return self.poloidal_block[5 + 4 * self.M_max : 6 + 5 * self.M_max]

    @property
    def m2_sin_mtheta(self) -> np.ndarray:
        return self.poloidal_block[6 + 5 * self.M_max : 7 + 6 * self.M_max]

    def to_grid(self) -> Grid:
        """从快照重建完整 Grid（用于 Equilibrium 物化等需要真实 Grid 的场景）."""
        from veqpy.model.grid import Grid

        return Grid(
            Nr=self.Nr,
            Nt=self.Nt,
            L_max=self.L_max,
            M_max=self.M_max,
            K_max=self.K_max,
            quadrature_scheme=self.quadrature_scheme,
            calculus_scheme=self.calculus_scheme,
        )


@dataclass(frozen=True, slots=True)
class ResidualBindingLayout:
    """绑定到 residual binder 的只读 metadata."""

    active_profile_names: tuple[str, ...]
    active_residual_block_codes: np.ndarray
    active_residual_block_orders: np.ndarray
    active_residual_block_radial_powers: np.ndarray
