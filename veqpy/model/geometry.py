"""model 层的 Geometry 定义.

属于 model 层.
负责持有单个 Grid 上物化后的几何 runtime buffer, 以及由 profile 派生的一维几何积分量.
不负责 packed state ownership, source route, residual 组装, 或 solver policy.
"""

from dataclasses import InitVar, dataclass, field, fields

import numpy as np

from veqpy.engine import update_geometry
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile


def bufferclass(cls):
    annotations = dict(getattr(cls, "__annotations__", {}))
    cls.__annotations__ = {"grid": InitVar[Grid], **annotations}

    def _buffer_field_names():
        return tuple(f.name for f in fields(cls) if not f.init)

    def __post_init__(self, grid: Grid):
        shape = (grid.Nr, grid.Nt)
        for name in _buffer_field_names():
            object.__setattr__(self, name, np.empty(shape, dtype=np.float64))

    def __iter__(self):
        for name in _buffer_field_names():
            yield getattr(self, name)

    cls.__post_init__ = __post_init__
    cls.__iter__ = __iter__
    return cls


@dataclass(frozen=True, slots=True)
@bufferclass
class _TbFields:
    tb: np.ndarray = field(init=False)
    cos_tb: np.ndarray = field(init=False)
    sin_tb: np.ndarray = field(init=False)
    tb_r: np.ndarray = field(init=False)
    tb_t: np.ndarray = field(init=False)
    tb_rr: np.ndarray = field(init=False)
    tb_rt: np.ndarray = field(init=False)
    tb_tt: np.ndarray = field(init=False)


@dataclass(frozen=True, slots=True)
@bufferclass
class _RFields:
    R: np.ndarray = field(init=False)
    R_r: np.ndarray = field(init=False)
    R_t: np.ndarray = field(init=False)
    R_rr: np.ndarray = field(init=False)
    R_rt: np.ndarray = field(init=False)
    R_tt: np.ndarray = field(init=False)


@dataclass(frozen=True, slots=True)
@bufferclass
class _ZFields:
    Z: np.ndarray = field(init=False)
    Z_r: np.ndarray = field(init=False)
    Z_t: np.ndarray = field(init=False)
    Z_rr: np.ndarray = field(init=False)
    Z_rt: np.ndarray = field(init=False)
    Z_tt: np.ndarray = field(init=False)


@dataclass(frozen=True, slots=True)
@bufferclass
class _JFields:
    J: np.ndarray = field(init=False)
    J_r: np.ndarray = field(init=False)
    J_t: np.ndarray = field(init=False)
    JR: np.ndarray = field(init=False)
    JR_r: np.ndarray = field(init=False)
    JR_t: np.ndarray = field(init=False)
    JdivR: np.ndarray = field(init=False)
    JdivR_r: np.ndarray = field(init=False)


@dataclass(frozen=True, slots=True)
@bufferclass
class _MFields:
    grt: np.ndarray = field(init=False)
    grt_t: np.ndarray = field(init=False)
    gtt: np.ndarray = field(init=False)
    gtt_r: np.ndarray = field(init=False)
    gttdivJR: np.ndarray = field(init=False)
    gttdivJR_r: np.ndarray = field(init=False)
    grtdivJR_t: np.ndarray = field(init=False)


@dataclass(frozen=True, slots=True)
class Geometry:
    """单个 grid 上的几何 runtime 容器.

    Geometry 持有已经物化的 2D 几何场和 1D 几何积分量.
    它依赖 Grid 和各个 Profile 的当前值, 但不持有 profile 根参数的所有权.
    """

    grid: InitVar[Grid]

    S_r: np.ndarray = field(init=False)
    V_r: np.ndarray = field(init=False)
    Kn: np.ndarray = field(init=False)
    Kn_r: np.ndarray = field(init=False)
    Ln_r: np.ndarray = field(init=False)

    _tb_fields: _TbFields = field(init=False)
    _R_fields: _RFields = field(init=False)
    _Z_fields: _ZFields = field(init=False)
    _J_fields: _JFields = field(init=False)
    _M_fields: _MFields = field(init=False)

    def __post_init__(self, grid: Grid):
        """按当前 grid 形状分配 Geometry runtime buffer."""
        object.__setattr__(self, "S_r", np.empty((grid.Nr,), dtype=np.float64))
        object.__setattr__(self, "V_r", np.empty((grid.Nr,), dtype=np.float64))
        object.__setattr__(self, "Kn", np.empty((grid.Nr,), dtype=np.float64))
        object.__setattr__(self, "Kn_r", np.empty((grid.Nr,), dtype=np.float64))
        object.__setattr__(self, "Ln_r", np.empty((grid.Nr,), dtype=np.float64))

        object.__setattr__(self, "_tb_fields", _TbFields(grid=grid))
        object.__setattr__(self, "_R_fields", _RFields(grid=grid))
        object.__setattr__(self, "_Z_fields", _ZFields(grid=grid))
        object.__setattr__(self, "_J_fields", _JFields(grid=grid))
        object.__setattr__(self, "_M_fields", _MFields(grid=grid))

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
        """用当前 Grid 和 profile 值刷新几何场.

        Args:
            a: 小半径尺度, 单位 m.
            R0: 几何中心 R 坐标, 单位 m.
            Z0: 几何中心 Z 坐标, 单位 m.
            grid: 当前几何所属网格.
            h_profile, v_profile, k_profile, c0_profile, c1_profile, s1_profile, s2_profile:
                已经在当前 grid 上物化的 profile runtime 容器.

        Returns:
            无返回值. 当前对象的二维几何场和一维几何积分量会被原地更新.
        """
        if self.R.shape != (grid.Nr, grid.Nt):
            raise ValueError(f"Expected geometry shape {(self.R.shape[0], self.R.shape[1])}, got {(grid.Nr, grid.Nt)}")

        rho, theta = grid.rho, grid.theta
        cos_theta, sin_theta = grid.cos_theta, grid.sin_theta
        weights = grid.weights

        update_geometry(
            *self._tb_fields,
            *self._R_fields,
            *self._Z_fields,
            *self._J_fields,
            *self._M_fields,
            self.S_r,
            self.V_r,
            self.Kn,
            self.Kn_r,
            self.Ln_r,
            float(a),
            float(R0),
            float(Z0),
            rho,
            theta,
            cos_theta,
            sin_theta,
            weights,
            *h_profile,
            *v_profile,
            *k_profile,
            *c0_profile,
            *c1_profile,
            *s1_profile,
            *s2_profile,
        )

    @property
    def tb(self) -> np.ndarray:
        return self._tb_fields.tb

    @property
    def cos_tb(self) -> np.ndarray:
        return self._tb_fields.cos_tb

    @property
    def sin_tb(self) -> np.ndarray:
        return self._tb_fields.sin_tb

    @property
    def tb_r(self) -> np.ndarray:
        return self._tb_fields.tb_r

    @property
    def tb_t(self) -> np.ndarray:
        return self._tb_fields.tb_t

    @property
    def tb_rr(self) -> np.ndarray:
        return self._tb_fields.tb_rr

    @property
    def tb_rt(self) -> np.ndarray:
        return self._tb_fields.tb_rt

    @property
    def tb_tt(self) -> np.ndarray:
        return self._tb_fields.tb_tt

    @property
    def R(self) -> np.ndarray:
        return self._R_fields.R

    @property
    def R_r(self) -> np.ndarray:
        return self._R_fields.R_r

    @property
    def R_t(self) -> np.ndarray:
        return self._R_fields.R_t

    @property
    def R_rr(self) -> np.ndarray:
        return self._R_fields.R_rr

    @property
    def R_rt(self) -> np.ndarray:
        return self._R_fields.R_rt

    @property
    def R_tt(self) -> np.ndarray:
        return self._R_fields.R_tt

    @property
    def Z(self) -> np.ndarray:
        return self._Z_fields.Z

    @property
    def Z_r(self) -> np.ndarray:
        return self._Z_fields.Z_r

    @property
    def Z_t(self) -> np.ndarray:
        return self._Z_fields.Z_t

    @property
    def Z_rr(self) -> np.ndarray:
        return self._Z_fields.Z_rr

    @property
    def Z_rt(self) -> np.ndarray:
        return self._Z_fields.Z_rt

    @property
    def Z_tt(self) -> np.ndarray:
        return self._Z_fields.Z_tt

    @property
    def J(self) -> np.ndarray:
        return self._J_fields.J

    @property
    def J_r(self) -> np.ndarray:
        return self._J_fields.J_r

    @property
    def J_t(self) -> np.ndarray:
        return self._J_fields.J_t

    @property
    def JR(self) -> np.ndarray:
        return self._J_fields.JR

    @property
    def JR_r(self) -> np.ndarray:
        return self._J_fields.JR_r

    @property
    def JR_t(self) -> np.ndarray:
        return self._J_fields.JR_t

    @property
    def JdivR(self) -> np.ndarray:
        return self._J_fields.JdivR

    @property
    def JdivR_r(self) -> np.ndarray:
        return self._J_fields.JdivR_r

    @property
    def grt(self) -> np.ndarray:
        return self._M_fields.grt

    @property
    def grt_t(self) -> np.ndarray:
        return self._M_fields.grt_t

    @property
    def gtt(self) -> np.ndarray:
        return self._M_fields.gtt

    @property
    def gtt_r(self) -> np.ndarray:
        return self._M_fields.gtt_r

    @property
    def gttdivJR(self) -> np.ndarray:
        return self._M_fields.gttdivJR

    @property
    def gttdivJR_r(self) -> np.ndarray:
        return self._M_fields.gttdivJR_r

    @property
    def grtdivJR_t(self) -> np.ndarray:
        return self._M_fields.grtdivJR_t
