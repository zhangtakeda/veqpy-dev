"""
Module: model.profile

Role:
- 负责持有单个 profile 的根参数与 runtime fields.
- 负责把 coeff 在当前 Grid 上物化成 `u_fields`.

Public API:
- Profile

Notes:
- `Profile` 属于 model 层 runtime 容器.
- 不负责 packed state ownership, source scaling, 或 solver orchestration.
"""

from dataclasses import InitVar, dataclass, field

import numpy as np

from veqpy.engine import update_profile
from veqpy.model.grid import Grid
from veqpy.model.serial import Serial


@dataclass(slots=True)
class Profile(Serial):
    """单个一维 profile 的 runtime 容器."""

    grid: InitVar[Grid | None] = None
    scale: float = 1.0
    power: int = 0
    envelope_power: int = 1
    offset: float = 0.0
    coeff: np.ndarray | None = None
    u_fields: np.ndarray | None = field(init=False, default=None, repr=False)
    T_fields: np.ndarray | None = field(init=False, default=None, repr=False)
    rp_fields: np.ndarray | None = field(init=False, default=None, repr=False)
    env_fields: np.ndarray | None = field(init=False, default=None, repr=False)

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """声明可序列化的根属性."""
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

        if grid is not None:
            self._prepare_runtime_cache(grid)

    @property
    def u(self) -> np.ndarray | None:
        return _field_slice(self.u_fields, 0)

    @property
    def u_r(self) -> np.ndarray | None:
        return _field_slice(self.u_fields, 1)

    @property
    def u_rr(self) -> np.ndarray | None:
        return _field_slice(self.u_fields, 2)

    def check(self) -> None:
        """校验根参数与可序列化字段."""
        for key, expected in type(self).serial_attributes().items():
            value = getattr(self, key)
            if value is None:
                continue
            if expected in {float, int} and not isinstance(value, (expected, np.generic)):
                raise TypeError(f"Attribute '{key}' must be {expected.__name__}, got {type(value).__name__}")
            if isinstance(value, np.ndarray) and value.ndim != 1:
                raise ValueError(f"Attribute '{key}' must be 1D, got {value.shape}")

    def copy(self) -> "Profile":
        """复制根参数与已存在的 runtime buffer."""
        out = Profile(
            scale=self.scale,
            power=self.power,
            envelope_power=self.envelope_power,
            offset=self.offset,
            coeff=_copy_optional_array(self.coeff),
        )
        out.u_fields = _copy_optional_array(self.u_fields)
        out.T_fields = self.T_fields
        out.rp_fields = self.rp_fields
        out.env_fields = self.env_fields
        return out

    def update(self, grid: Grid | None = None) -> None:
        """刷新当前 Grid 上的 profile fields."""
        if grid is not None:
            self._prepare_runtime_cache(grid)
        if self.T_fields is None:
            raise RuntimeError("Profile runtime cache is not initialized; pass grid on first update().")
        if self.u_fields is None:
            raise RuntimeError("Profile output buffers are not initialized; pass grid on first update().")
        _fill_profile_outputs(
            self.u_fields,
            self.T_fields,
            self.rp_fields,
            self.env_fields,
            self.offset,
            self.coeff,
            self.scale,
        )

    def _prepare_runtime_cache(self, grid: Grid) -> None:
        """绑定 Grid 并准备 runtime 缓存."""
        self.T_fields = grid.T_fields
        self.rp_fields = _power_terms(grid.rho, self.power)
        self.env_fields = _envelope_terms(grid.rho, grid.rho2, grid.y, self.envelope_power)
        self.rp_fields.flags.writeable = False
        self.env_fields.flags.writeable = False

        if self.u_fields is None or self.u_fields.shape != (3, grid.Nr):
            self.u_fields = np.empty((3, grid.Nr), dtype=np.float64)


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
def _fill_profile_outputs(
    u_fields: np.ndarray,
    T_fields: np.ndarray,
    rp_fields: np.ndarray,
    env_fields: np.ndarray,
    offset: float,
    coeff: np.ndarray | None,
    scale: float,
) -> None:
    """根据 coeff 刷新单个 profile fields."""
    update_profile(
        u_fields,
        T_fields,
        rp_fields,
        env_fields,
        offset,
        coeff,
    )
    if scale != 1.0:
        np.multiply(u_fields, scale, out=u_fields)


def _power_terms(rho: np.ndarray, a: int) -> np.ndarray:
    if a == 0:
        ones = np.ones_like(rho)
        zeros = np.zeros_like(rho)
        return _stack_fields(ones, zeros, zeros)

    rp = rho**a
    rp_r = a * rho ** (a - 1)
    if a == 1:
        rp_rr = np.zeros_like(rho)
    else:
        rp_rr = a * (a - 1) * rho ** (a - 2)
    return _stack_fields(rp, rp_r, rp_rr)


def _envelope_terms(
    rho: np.ndarray,
    rho2: np.ndarray,
    y: np.ndarray,
    c: int,
) -> np.ndarray:
    if c == 0:
        ones = np.ones_like(rho)
        zeros = np.zeros_like(rho)
        return _stack_fields(ones, zeros, zeros)
    if c == 1:
        return _stack_fields(y, -2.0 * rho, np.full_like(rho, -2.0))

    env = y**c
    env_r = -2.0 * c * rho * y ** (c - 1)
    env_rr = -2.0 * c * y ** (c - 1) + 4.0 * c * (c - 1) * rho2 * y ** (c - 2)
    return _stack_fields(env, env_r, env_rr)


def _field_slice(fields: np.ndarray | None, index: int) -> np.ndarray | None:
    if fields is None:
        return None
    return fields[index]


def _stack_fields(a0: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    out = np.empty((3, a0.shape[0]), dtype=np.float64)
    out[0] = a0
    out[1] = a1
    out[2] = a2
    return out
