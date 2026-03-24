"""model 层的 Profile 定义.

属于 model 层.
负责持有单个 profile 的根参数, 以及绑定到某个 Grid 后的 u, u_r, u_rr runtime buffer.
不负责 packed state ownership, source scaling, 或 solver orchestration.
"""

from dataclasses import InitVar, dataclass, field

import numpy as np

from veqpy.engine import update_profile
from veqpy.model.grid import Grid
from veqpy.model.serial import Serial


@dataclass(slots=True)
class Profile(Serial):
    """单个一维 profile 的 runtime 容器.

    Profile 持有 scale, power, envelope_power, offset, coeff 等根参数.
    绑定到某个 Grid 后, 它会物化出 u, u_r, u_rr 三个 1D runtime buffer.
    """

    grid: InitVar[Grid | None] = None
    scale: float = 1.0
    power: int = 0
    envelope_power: int = 1
    offset: float = 0.0
    coeff: np.ndarray | None = None
    u: np.ndarray | None = field(init=False, default=None)
    u_r: np.ndarray | None = field(init=False, default=None)
    u_rr: np.ndarray | None = field(init=False, default=None)
    _coeff_topology: tuple[bool, int] | None = field(init=False, default=None, repr=False)
    _T: np.ndarray | None = field(init=False, default=None, repr=False)
    _T_r: np.ndarray | None = field(init=False, default=None, repr=False)
    _T_rr: np.ndarray | None = field(init=False, default=None, repr=False)
    _rp: np.ndarray | None = field(init=False, default=None, repr=False)
    _rp_r: np.ndarray | None = field(init=False, default=None, repr=False)
    _rp_rr: np.ndarray | None = field(init=False, default=None, repr=False)
    _env: np.ndarray | None = field(init=False, default=None, repr=False)
    _env_r: np.ndarray | None = field(init=False, default=None, repr=False)
    _env_rr: np.ndarray | None = field(init=False, default=None, repr=False)

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """声明可序列化的根属性.

        Returns:
            可用于重建 Profile 根参数的字段类型映射.
        """
        return {
            "scale": float,
            "power": int,
            "envelope_power": int,
            "offset": float,
            "coeff": np.ndarray | None,
        }

    def __post_init__(self, grid: Grid | None):
        self.scale = float(self.scale)
        self.power = int(self.power)
        self.envelope_power = int(self.envelope_power)
        self.offset = 0.0 if self.offset is None else float(self.offset)
        self.coeff = _coerce_optional_array(self.coeff, copy=False, name="coeff")
        if self.coeff is not None:
            self._coeff_topology = _coeff_topology(self.coeff)

        if grid is not None:
            self._prepare_runtime_cache(grid)

    def __iter__(self):
        """返回当前已经物化的 u, u_r, u_rr 三元组."""
        if self.u is None or self.u_r is None or self.u_rr is None:
            raise RuntimeError("Profile is not materialized; call update(..., grid=...) first")
        yield self.u
        yield self.u_r
        yield self.u_rr

    def check(self) -> None:
        """校验根参数与可序列化字段的基本类型约束."""
        for key, expected in type(self).serial_attributes().items():
            value = getattr(self, key)
            if value is None:
                continue
            if expected in {float, int} and not isinstance(value, (expected, np.generic)):
                raise TypeError(f"Attribute '{key}' must be {expected.__name__}, got {type(value).__name__}")
            if isinstance(value, np.ndarray) and value.ndim != 1:
                raise ValueError(f"Attribute '{key}' must be 1D, got {value.shape}")

    def copy(self) -> "Profile":
        """复制 Profile 根参数与已存在的 runtime buffer."""
        out = Profile(
            scale=self.scale,
            power=self.power,
            envelope_power=self.envelope_power,
            offset=self.offset,
            coeff=_copy_optional_array(self.coeff),
        )
        out.u = _copy_optional_array(self.u)
        out.u_r = _copy_optional_array(self.u_r)
        out.u_rr = _copy_optional_array(self.u_rr)
        out._coeff_topology = self._coeff_topology
        out._T = self._T
        out._T_r = self._T_r
        out._T_rr = self._T_rr
        out._rp = self._rp
        out._rp_r = self._rp_r
        out._rp_rr = self._rp_rr
        out._env = self._env
        out._env_r = self._env_r
        out._env_rr = self._env_rr
        return out

    def update(self, grid: Grid | None = None) -> None:
        """刷新当前 grid 上的 profile 值和导数.

        Args:
            grid: 可选的新 Grid. 首次调用时必须提供, 以建立 runtime cache.

        Returns:
            无返回值. 当前对象的 u, u_r, u_rr 会被原地更新.
        """
        if grid is not None:
            self._prepare_runtime_cache(grid)
        if self._T is None:
            raise RuntimeError("Profile runtime cache is not initialized; pass grid on first update().")
        if self.u is None or self.u_r is None or self.u_rr is None:
            raise RuntimeError("Profile output buffers are not initialized; pass grid on first update().")
        update_profile(
            self.u,
            self.u_r,
            self.u_rr,
            self._T,
            self._T_r,
            self._T_rr,
            self._rp,
            self._rp_r,
            self._rp_rr,
            self._env,
            self._env_r,
            self._env_rr,
            self.offset,
            self.coeff,
        )
        if self.scale != 1.0:
            np.multiply(self.u, self.scale, out=self.u)
            np.multiply(self.u_r, self.scale, out=self.u_r)
            np.multiply(self.u_rr, self.scale, out=self.u_rr)

    def _bind_coeff(self, coeff: np.ndarray | None) -> None:
        """绑定并锁定 coeff 拓扑.

        第一次绑定决定该 Profile 的 coeff 是否为 None 以及长度.
        之后只允许重绑到相同拓扑的数组视图.
        """
        coeff_arr = _coerce_optional_array(coeff, copy=False, name="coeff")
        topology = _coeff_topology(coeff_arr)
        if self._coeff_topology is None:
            self.coeff = coeff_arr
            self._coeff_topology = topology
            return
        if topology != self._coeff_topology:
            expected = _format_coeff_topology(self._coeff_topology)
            got = _format_coeff_topology(topology)
            raise ValueError(f"Profile coeff topology is immutable; expected {expected}, got {got}")
        self.coeff = coeff_arr

    def _prepare_runtime_cache(self, grid: Grid) -> None:
        """绑定 Grid 并准备 profile 计算所需的只读缓存."""
        self._T = grid.T
        self._T_r = grid.T_r
        self._T_rr = grid.T_rr

        rp, rp_r, rp_rr = _power_terms(grid.rho, self.power)
        env, env_r, env_rr = _envelope_terms(grid.rho, grid.rho2, grid.y, self.envelope_power)
        for arr in (rp, rp_r, rp_rr, env, env_r, env_rr):
            arr.flags.writeable = False

        self._rp = rp
        self._rp_r = rp_r
        self._rp_rr = rp_rr
        self._env = env
        self._env_r = env_r
        self._env_rr = env_rr

        if self.u is None or self.u.shape != (grid.Nr,):
            self.u = np.empty(grid.Nr, dtype=np.float64)
            self.u_r = np.empty(grid.Nr, dtype=np.float64)
            self.u_rr = np.empty(grid.Nr, dtype=np.float64)


def _coerce_optional_array(value, *, copy: bool, name: str = "array") -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        scalar = value.item()
        if scalar is None or (isinstance(scalar, float) and np.isnan(scalar)):
            return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr.copy() if copy else arr


def _copy_optional_array(value: np.ndarray | None) -> np.ndarray | None:
    return None if value is None else value.copy()


def _coeff_topology(value: np.ndarray | None) -> tuple[bool, int]:
    if value is None:
        return True, -1
    return False, int(value.shape[0])


def _format_coeff_topology(topology: tuple[bool, int]) -> str:
    is_none, length = topology
    if is_none:
        return "coeff=None"
    return f"coeff length {length}"


def _power_terms(rho: np.ndarray, a: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if a == 0:
        ones = np.ones_like(rho)
        zeros = np.zeros_like(rho)
        return ones, zeros, zeros

    rp = rho**a
    rp_r = a * rho ** (a - 1)
    if a == 1:
        rp_rr = np.zeros_like(rho)
    else:
        rp_rr = a * (a - 1) * rho ** (a - 2)
    return rp, rp_r, rp_rr


def _envelope_terms(
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    c: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if c == 0:
        ones = np.ones_like(rho)
        zeros = np.zeros_like(rho)
        return ones, zeros, zeros
    if c == 1:
        return y, -2.0 * rho, np.full_like(rho, -2.0)

    env = y**c
    env_r = -2.0 * c * rho * y ** (c - 1)
    env_rr = -2.0 * c * y ** (c - 1) + 4.0 * c * (c - 1) * rho2 * y ** (c - 2)
    return env, env_r, env_rr
