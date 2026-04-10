"""
Module: model.geometry

Role:
- 负责持有单个 Grid 上物化后的 geometry fields 与积分量.
- 负责把 shape profiles 映射成 geometry runtime fields.

Public API:
- Geometry

Notes:
- `Geometry` 属于 model 层 runtime 容器.
- 不负责 packed state ownership, source route, residual 组装, 或 solver policy.
"""

from dataclasses import InitVar, dataclass, field

import numpy as np
from numba import njit

from veqpy.model.grid import Grid


@dataclass(frozen=True, slots=True)
class Geometry:
    """单个 Grid 上的 geometry runtime 容器."""

    grid: InitVar[Grid]

    S_r: np.ndarray = field(init=False)
    V_r: np.ndarray = field(init=False)
    Kn: np.ndarray = field(init=False)
    Kn_r: np.ndarray = field(init=False)
    Ln_r: np.ndarray = field(init=False)

    tb_fields: np.ndarray = field(init=False, repr=False)
    R_fields: np.ndarray = field(init=False, repr=False)
    Z_fields: np.ndarray = field(init=False, repr=False)
    J_fields: np.ndarray = field(init=False, repr=False)
    g_fields: np.ndarray = field(init=False, repr=False)

    def __post_init__(self, grid: Grid):
        nr = grid.Nr
        nt = grid.Nt
        object.__setattr__(self, "S_r", np.empty(nr, dtype=np.float64))
        object.__setattr__(self, "V_r", np.empty(nr, dtype=np.float64))
        object.__setattr__(self, "Kn", np.empty(nr, dtype=np.float64))
        object.__setattr__(self, "Kn_r", np.empty(nr, dtype=np.float64))
        object.__setattr__(self, "Ln_r", np.empty(nr, dtype=np.float64))
        object.__setattr__(self, "tb_fields", np.empty((8, nr, nt), dtype=np.float64))
        object.__setattr__(self, "R_fields", np.empty((6, nr, nt), dtype=np.float64))
        object.__setattr__(self, "Z_fields", np.empty((6, nr, nt), dtype=np.float64))
        object.__setattr__(self, "J_fields", np.empty((8, nr, nt), dtype=np.float64))
        object.__setattr__(self, "g_fields", np.empty((7, nr, nt), dtype=np.float64))

    def update(
        self,
        a: float,
        R0: float,
        Z0: float,
        grid: Grid,
        h_fields: np.ndarray,
        v_fields: np.ndarray,
        k_fields: np.ndarray,
        c_fields: np.ndarray,
        s_fields: np.ndarray,
        *,
        c_active_order: int | None = None,
        s_active_order: int | None = None,
    ):
        """用当前 Grid 和 profile fields 刷新 geometry."""
        if self.R_fields.shape[1:] != (grid.Nr, grid.Nt):
            raise ValueError(
                f"Expected geometry shape {(self.R_fields.shape[1], self.R_fields.shape[2])}, got {(grid.Nr, grid.Nt)}"
            )
        if c_active_order is None:
            c_active_order = int(c_fields.shape[0] - 1)
        if s_active_order is None:
            s_active_order = int(s_fields.shape[0] - 1)

        _geometry_update(
            self.tb_fields,
            self.R_fields,
            self.Z_fields,
            self.J_fields,
            self.g_fields,
            self.S_r,
            self.V_r,
            self.Kn,
            self.Kn_r,
            self.Ln_r,
            float(a),
            float(R0),
            float(Z0),
            grid.rho,
            grid.theta,
            grid.cos_ktheta,
            grid.sin_ktheta,
            grid.k_cos_ktheta,
            grid.k_sin_ktheta,
            grid.k2_cos_ktheta,
            grid.k2_sin_ktheta,
            grid.weights,
            h_fields,
            v_fields,
            k_fields,
            c_fields,
            s_fields,
            int(c_active_order),
            int(s_active_order),
        )

    @property
    def tb(self) -> np.ndarray:
        return self.tb_fields[0]

    @property
    def tb_r(self) -> np.ndarray:
        return self.tb_fields[1]

    @property
    def tb_t(self) -> np.ndarray:
        return self.tb_fields[2]

    @property
    def tb_rr(self) -> np.ndarray:
        return self.tb_fields[3]

    @property
    def tb_rt(self) -> np.ndarray:
        return self.tb_fields[4]

    @property
    def tb_tt(self) -> np.ndarray:
        return self.tb_fields[5]

    @property
    def cos_tb(self) -> np.ndarray:
        return self.tb_fields[6]

    @property
    def sin_tb(self) -> np.ndarray:
        return self.tb_fields[7]

    @property
    def R(self) -> np.ndarray:
        return self.R_fields[0]

    @property
    def R_r(self) -> np.ndarray:
        return self.R_fields[1]

    @property
    def R_t(self) -> np.ndarray:
        return self.R_fields[2]

    @property
    def R_rr(self) -> np.ndarray:
        return self.R_fields[3]

    @property
    def R_rt(self) -> np.ndarray:
        return self.R_fields[4]

    @property
    def R_tt(self) -> np.ndarray:
        return self.R_fields[5]

    @property
    def Z(self) -> np.ndarray:
        return self.Z_fields[0]

    @property
    def Z_r(self) -> np.ndarray:
        return self.Z_fields[1]

    @property
    def Z_t(self) -> np.ndarray:
        return self.Z_fields[2]

    @property
    def Z_rr(self) -> np.ndarray:
        return self.Z_fields[3]

    @property
    def Z_rt(self) -> np.ndarray:
        return self.Z_fields[4]

    @property
    def Z_tt(self) -> np.ndarray:
        return self.Z_fields[5]

    @property
    def J(self) -> np.ndarray:
        return self.J_fields[0]

    @property
    def J_r(self) -> np.ndarray:
        return self.J_fields[1]

    @property
    def J_t(self) -> np.ndarray:
        return self.J_fields[2]

    @property
    def JR(self) -> np.ndarray:
        return self.J_fields[3]

    @property
    def JR_r(self) -> np.ndarray:
        return self.J_fields[4]

    @property
    def JR_t(self) -> np.ndarray:
        return self.J_fields[5]

    @property
    def JdivR(self) -> np.ndarray:
        return self.J_fields[6]

    @property
    def JdivR_r(self) -> np.ndarray:
        return self.J_fields[7]

    @property
    def grt(self) -> np.ndarray:
        return self.g_fields[0]

    @property
    def grt_t(self) -> np.ndarray:
        return self.g_fields[1]

    @property
    def gtt(self) -> np.ndarray:
        return self.g_fields[3]

    @property
    def gtt_r(self) -> np.ndarray:
        return self.g_fields[4]

    @property
    def gttdivJR(self) -> np.ndarray:
        return self.g_fields[5]

    @property
    def gttdivJR_r(self) -> np.ndarray:
        return self.g_fields[6]

    @property
    def grtdivJR_t(self) -> np.ndarray:
        return self.g_fields[2]


@njit(cache=True, fastmath=True, nogil=True)
def _geometry_update(
    tb_fields: np.ndarray,
    R_fields: np.ndarray,
    Z_fields: np.ndarray,
    J_fields: np.ndarray,
    g_fields: np.ndarray,
    S_r: np.ndarray,
    V_r: np.ndarray,
    Kn: np.ndarray,
    Kn_r: np.ndarray,
    Ln_r: np.ndarray,
    a: float,
    R0: float,
    Z0: float,
    rho: np.ndarray,
    theta: np.ndarray,
    cos_ktheta: np.ndarray,
    sin_ktheta: np.ndarray,
    k_cos_ktheta: np.ndarray,
    k_sin_ktheta: np.ndarray,
    k2_cos_ktheta: np.ndarray,
    k2_sin_ktheta: np.ndarray,
    weights: np.ndarray,
    h_fields: np.ndarray,
    v_fields: np.ndarray,
    k_fields: np.ndarray,
    c_fields: np.ndarray,
    s_fields: np.ndarray,
    c_active_order: int,
    s_active_order: int,
):
    """原地更新 geometry fields 与 geometry integrals."""
    nr = rho.shape[0]
    nt = theta.shape[0]
    theta_scale = 2.0 * np.pi / nt
    mean_scale = 1.0 / nt
    two_pi = 2.0 * np.pi
    c_limit = min(c_active_order + 1, c_fields.shape[0], cos_ktheta.shape[0])
    s_limit = min(s_active_order + 1, s_fields.shape[0], sin_ktheta.shape[0])

    for i in range(nr):
        rho_i = rho[i]
        h_i = h_fields[0, i]
        h_r_i = h_fields[1, i]
        h_rr_i = h_fields[2, i]
        v_i = v_fields[0, i]
        v_r_i = v_fields[1, i]
        v_rr_i = v_fields[2, i]
        k_i = k_fields[0, i]
        k_r_i = k_fields[1, i]
        k_rr_i = k_fields[2, i]
        c0_i = c_fields[0, 0, i]
        c0_r_i = c_fields[0, 1, i]
        c0_rr_i = c_fields[0, 2, i]

        sum_J = 0.0
        sum_JR = 0.0
        sum_gttdivJR = 0.0
        sum_gttdivJR_r = 0.0
        sum_JdivR = 0.0

        for j in range(nt):
            sin_t = sin_ktheta[1, j]
            cos_t = cos_ktheta[1, j]

            tb_ij = theta[j] + c0_i
            tb_r_ij = c0_r_i
            tb_t_ij = 1.0
            tb_rr_ij = c0_rr_i
            tb_rt_ij = 0.0
            tb_tt_ij = 0.0

            for order in range(1, c_limit):
                cos_kt = cos_ktheta[order, j]
                k_sin_kt = k_sin_ktheta[order, j]
                k2_cos_kt = k2_cos_ktheta[order, j]
                c_i = c_fields[order, 0, i]
                c_r_i = c_fields[order, 1, i]
                c_rr_i = c_fields[order, 2, i]

                tb_ij += c_i * cos_kt
                tb_r_ij += c_r_i * cos_kt
                tb_t_ij -= c_i * k_sin_kt
                tb_rr_ij += c_rr_i * cos_kt
                tb_rt_ij -= c_r_i * k_sin_kt
                tb_tt_ij -= c_i * k2_cos_kt

            for order in range(1, s_limit):
                sin_kt = sin_ktheta[order, j]
                k_cos_kt = k_cos_ktheta[order, j]
                k2_sin_kt = k2_sin_ktheta[order, j]
                s_i = s_fields[order, 0, i]
                s_r_i = s_fields[order, 1, i]
                s_rr_i = s_fields[order, 2, i]

                tb_ij += s_i * sin_kt
                tb_r_ij += s_r_i * sin_kt
                tb_t_ij += s_i * k_cos_kt
                tb_rr_ij += s_rr_i * sin_kt
                tb_rt_ij += s_r_i * k_cos_kt
                tb_tt_ij -= s_i * k2_sin_kt

            cos_tb_ij = np.cos(tb_ij)
            sin_tb_ij = np.sin(tb_ij)

            R_ij = R0 + a * (h_i + rho_i * cos_tb_ij)
            if R_ij < 1e-15:
                R_ij = 1e-15

            R_r_ij = a * (h_r_i + cos_tb_ij - rho_i * sin_tb_ij * tb_r_ij)
            R_t_ij = -a * rho_i * sin_tb_ij * tb_t_ij
            R_rr_ij = a * (
                h_rr_i - 2.0 * sin_tb_ij * tb_r_ij - rho_i * (cos_tb_ij * tb_r_ij * tb_r_ij + sin_tb_ij * tb_rr_ij)
            )
            R_rt_ij = -a * (sin_tb_ij * tb_t_ij + rho_i * (cos_tb_ij * tb_r_ij * tb_t_ij + sin_tb_ij * tb_rt_ij))
            R_tt_ij = -a * rho_i * (cos_tb_ij * tb_t_ij * tb_t_ij + sin_tb_ij * tb_tt_ij)

            Z_ij = Z0 + a * (v_i - rho_i * k_i * sin_t)
            Z_r_ij = a * (v_r_i - (k_i + rho_i * k_r_i) * sin_t)
            Z_t_ij = -a * rho_i * k_i * cos_t
            Z_rr_ij = a * (v_rr_i - (2.0 * k_r_i + rho_i * k_rr_i) * sin_t)
            Z_rt_ij = -a * (k_i + rho_i * k_r_i) * cos_t
            Z_tt_ij = a * rho_i * k_i * sin_t

            J_ij = R_t_ij * Z_r_ij - R_r_ij * Z_t_ij
            if J_ij < 1e-15:
                J_ij = 1e-15

            J_r_ij = -(R_rr_ij * Z_t_ij - R_rt_ij * Z_r_ij + R_r_ij * Z_rt_ij - R_t_ij * Z_rr_ij)
            J_t_ij = -(R_rt_ij * Z_t_ij - R_tt_ij * Z_r_ij + R_r_ij * Z_tt_ij - R_t_ij * Z_rt_ij)
            JR_ij = J_ij * R_ij
            JR_r_ij = J_r_ij * R_ij + J_ij * R_r_ij
            JR_t_ij = J_t_ij * R_ij + J_ij * R_t_ij
            inv_R = 1.0 / R_ij
            JdivR_ij = J_ij * inv_R
            JdivR_r_ij = (J_r_ij * R_ij - J_ij * R_r_ij) * inv_R * inv_R

            grt_ij = R_r_ij * R_t_ij + Z_r_ij * Z_t_ij
            grt_t_ij = R_rt_ij * R_t_ij + R_r_ij * R_tt_ij + Z_rt_ij * Z_t_ij + Z_r_ij * Z_tt_ij
            gtt_ij = R_t_ij * R_t_ij + Z_t_ij * Z_t_ij
            gtt_r_ij = 2.0 * (R_t_ij * R_rt_ij + Z_t_ij * Z_rt_ij)
            inv_JR = 1.0 / JR_ij
            grtdivJR_t_ij = (grt_t_ij - grt_ij * JR_t_ij * inv_JR) * inv_JR
            gttdivJR_ij = gtt_ij * inv_JR
            gttdivJR_r_ij = gtt_r_ij * inv_JR - gtt_ij * JR_r_ij * inv_JR * inv_JR

            tb_fields[0, i, j] = tb_ij
            tb_fields[1, i, j] = tb_r_ij
            tb_fields[2, i, j] = tb_t_ij
            tb_fields[3, i, j] = tb_rr_ij
            tb_fields[4, i, j] = tb_rt_ij
            tb_fields[5, i, j] = tb_tt_ij
            tb_fields[6, i, j] = cos_tb_ij
            tb_fields[7, i, j] = sin_tb_ij

            R_fields[0, i, j] = R_ij
            R_fields[1, i, j] = R_r_ij
            R_fields[2, i, j] = R_t_ij
            R_fields[3, i, j] = R_rr_ij
            R_fields[4, i, j] = R_rt_ij
            R_fields[5, i, j] = R_tt_ij

            Z_fields[0, i, j] = Z_ij
            Z_fields[1, i, j] = Z_r_ij
            Z_fields[2, i, j] = Z_t_ij
            Z_fields[3, i, j] = Z_rr_ij
            Z_fields[4, i, j] = Z_rt_ij
            Z_fields[5, i, j] = Z_tt_ij

            J_fields[0, i, j] = J_ij
            J_fields[1, i, j] = J_r_ij
            J_fields[2, i, j] = J_t_ij

            J_fields[3, i, j] = JR_ij
            J_fields[4, i, j] = JR_r_ij
            J_fields[5, i, j] = JR_t_ij
            J_fields[6, i, j] = JdivR_ij
            J_fields[7, i, j] = JdivR_r_ij

            g_fields[0, i, j] = grt_ij
            g_fields[1, i, j] = grt_t_ij
            g_fields[2, i, j] = grtdivJR_t_ij
            g_fields[3, i, j] = gtt_ij
            g_fields[4, i, j] = gtt_r_ij
            g_fields[5, i, j] = gttdivJR_ij
            g_fields[6, i, j] = gttdivJR_r_ij

            sum_J += J_ij
            sum_JR += JR_ij
            sum_gttdivJR += gttdivJR_ij
            sum_gttdivJR_r += gttdivJR_r_ij
            sum_JdivR += JdivR_ij

        S_r[i] = sum_J * theta_scale
        V_r[i] = sum_JR * theta_scale * two_pi
        Kn[i] = sum_gttdivJR * mean_scale
        Kn_r[i] = sum_gttdivJR_r * mean_scale
        Ln_r[i] = sum_JdivR * mean_scale
