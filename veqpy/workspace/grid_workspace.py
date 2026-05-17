"""Static grid memory snapshot used by operator hot paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

if TYPE_CHECKING:
    from veqpy.model.grid import Grid


@dataclass(frozen=True, slots=True)
class GridWorkspace:
    """Operator-level Grid snapshot: arrays and metadata required by the engine ABI.

    K_max is normalized: Grid.K_max=None maps to M_max.
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

    # rho           (Nr,)
    # x             (Nr,)
    # y             (Nr,)
    # rho_powers    (K_max+2, Nr)
    # T             (L_max+1, Nr)
    # T_r           (L_max+1, Nr)
    # T_rr          (L_max+1, Nr)
    radial_block: np.ndarray  # (8+K_max+3*L_max, Nr)

    # theta         (Nt,)
    # cos_mtheta    (M_max+1, Nt)
    # sin_mtheta    (M_max+1, Nt)
    # m_cos_mtheta  (M_max+1, Nt)
    # m_sin_mtheta  (M_max+1, Nt)
    # m2_cos_mtheta (M_max+1, Nt)
    # m2_sin_mtheta (M_max+1, Nt)
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

    @classmethod
    def from_grid(cls, grid: Grid) -> Self:
        """Lower ``Grid`` into static arrays consumed by runtime binding."""

        return cls(
            Nr=int(grid.Nr),
            Nt=int(grid.Nt),
            M_max=int(grid.M_max),
            L_max=int(grid.L_max),
            K_max=grid.K_max or grid.M_max,
            quadrature_scheme=grid.quadrature_scheme,
            calculus_scheme=grid.calculus_scheme,
            K_values=grid.K_values.copy(),
            weights=grid.weights.copy(),
            differentiator=grid.differentiator.copy(),
            accumulator=grid.accumulator.copy(),
            radial_block=_pack_radial_block(
                grid.rho, grid.x, grid.y, grid.rho_powers, grid.T, grid.T_r, grid.T_rr
            ),
            poloidal_block=_pack_poloidal_block(
                grid.theta,
                grid.cos_mtheta,
                grid.sin_mtheta,
                grid.m_cos_mtheta,
                grid.m_sin_mtheta,
                grid.m2_cos_mtheta,
                grid.m2_sin_mtheta,
            ),
        )

    def to_grid(self) -> Grid:
        """Rebuild a full Grid from the snapshot for Equilibrium materialization and
        other callers that need a real Grid."""
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


def _pack_radial_block(
    rho: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    rho_powers: np.ndarray,
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
) -> np.ndarray:
    """Pack radial fields into a read-only (R, Nr) 2D array according to the layout contract."""
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
    """Pack poloidal fields into a read-only (P, Nt) 2D array according to the layout contract."""
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
