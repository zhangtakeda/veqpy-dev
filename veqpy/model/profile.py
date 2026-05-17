"""
Module: model.profile

Role:
- Hold root parameters and runtime fields for one profile.
- Materialize coefficients on the current Grid into `u_fields`.

Public API:
- Profile

Notes:
- `Profile` is a model-layer runtime container.
- Does not own packed state, source scaling, or solver orchestration.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Self

import numpy as np

from veqpy.base import Serial
from veqpy.engine.numba_profile import update_profile
from veqpy.model.grid import Grid


@dataclass(slots=True)
class Profile(Serial):
    """Runtime container for one one-dimensional profile."""

    grid: InitVar[Grid | None] = None
    scale: float = 1.0
    power: int = 0
    envelope_power: int = 1
    offset: float = 0.0
    coeff: np.ndarray | None = None
    u_fields: np.ndarray | None = field(init=False, default=None, repr=False)
    T: np.ndarray | None = field(init=False, default=None, repr=False)
    T_r: np.ndarray | None = field(init=False, default=None, repr=False)
    T_rr: np.ndarray | None = field(init=False, default=None, repr=False)
    rp_fields: np.ndarray | None = field(init=False, default=None, repr=False)
    env_fields: np.ndarray | None = field(init=False, default=None, repr=False)

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """Declare serializable root attributes."""
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
        """Validate root parameters and serializable fields."""
        for key, expected in type(self).serial_attributes().items():
            value = getattr(self, key)
            if value is None:
                continue
            if expected in {float, int} and not isinstance(value, (expected, np.generic)):
                raise TypeError(
                    f"Attribute '{key}' must be {expected.__name__}, got {type(value).__name__}"
                )
            if isinstance(value, np.ndarray) and value.ndim != 1:
                raise ValueError(f"Attribute '{key}' must be 1D, got {value.shape}")

    def copy(self) -> Self:
        """Copy root parameters and existing runtime buffers."""
        out = Profile(
            scale=self.scale,
            power=self.power,
            envelope_power=self.envelope_power,
            offset=self.offset,
            coeff=_copy_optional_array(self.coeff),
        )
        out.u_fields = _copy_optional_array(self.u_fields)
        out.T = self.T
        out.T_r = self.T_r
        out.T_rr = self.T_rr
        out.rp_fields = self.rp_fields
        out.env_fields = self.env_fields
        return out

    def update(self, grid: Grid | None = None) -> None:
        """Refresh profile fields on the current Grid."""
        if grid is not None:
            self._prepare_runtime_cache(grid)
        if self.T is None:
            raise RuntimeError(
                "Profile runtime cache is not initialized; pass grid on first update()."
            )
        if self.u_fields is None:
            raise RuntimeError("Profile buffers not initialized; pass grid first.")
        _fill_profile_outputs(
            self.u_fields,
            self.T,
            self.T_r,
            self.T_rr,
            self.rp_fields,
            self.env_fields,
            self.offset,
            self.coeff,
            self.scale,
        )

    def _prepare_runtime_cache(self, grid) -> None:
        """Bind a grid snapshot (Grid or GridWorkspace) and prepare runtime caches."""
        self.T = grid.T
        self.T_r = grid.T_r
        self.T_rr = grid.T_rr
        self.rp_fields = _power_terms(grid.rho, self.power)
        self.env_fields = _envelope_terms(grid.rho, grid.rho_powers[2], grid.y, self.envelope_power)
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
    T: np.ndarray,
    T_r: np.ndarray,
    T_rr: np.ndarray,
    rp_fields: np.ndarray,
    env_fields: np.ndarray,
    offset: float,
    coeff: np.ndarray | None,
    scale: float,
) -> None:
    """Refresh one profile field set from coefficients."""
    update_profile(u_fields, T, T_r, T_rr, rp_fields, env_fields, offset, coeff)
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
