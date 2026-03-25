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

from veqpy.engine import update_geometry
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile

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
        shape = (nr, nt)
        object.__setattr__(self, "S_r", np.empty(nr, dtype=np.float64))
        object.__setattr__(self, "V_r", np.empty(nr, dtype=np.float64))
        object.__setattr__(self, "Kn", np.empty(nr, dtype=np.float64))
        object.__setattr__(self, "Kn_r", np.empty(nr, dtype=np.float64))
        object.__setattr__(self, "Ln_r", np.empty(nr, dtype=np.float64))
        object.__setattr__(self, "tb_fields", np.empty((8, *shape), dtype=np.float64))
        object.__setattr__(self, "R_fields", np.empty((6, *shape), dtype=np.float64))
        object.__setattr__(self, "Z_fields", np.empty((6, *shape), dtype=np.float64))
        object.__setattr__(self, "J_fields", np.empty((8, *shape), dtype=np.float64))
        object.__setattr__(self, "g_fields", np.empty((7, *shape), dtype=np.float64))

    def update(
        self,
        a: float,
        R0: float,
        Z0: float,
        grid: Grid,
        h_profile: Profile,
        v_profile: Profile,
        k_profile: Profile,
        c0_profile: Profile,
        c1_profile: Profile,
        s1_profile: Profile,
        s2_profile: Profile,
    ):
        """用当前 Grid 和 profile fields 刷新 geometry."""
        if self.R_fields.shape[1:] != (grid.Nr, grid.Nt):
            raise ValueError(
                f"Expected geometry shape {(self.R_fields.shape[1], self.R_fields.shape[2])}, got {(grid.Nr, grid.Nt)}"
            )

        update_geometry(
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
            grid.cos_theta,
            grid.sin_theta,
            grid.cos_2theta,
            grid.sin_2theta,
            grid.weights,
            h_profile.u_fields,
            v_profile.u_fields,
            k_profile.u_fields,
            c0_profile.u_fields,
            c1_profile.u_fields,
            s1_profile.u_fields,
            s2_profile.u_fields,
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
