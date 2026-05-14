"""
Module: model.grid

Role:
- 负责持有径向-极角网格配置及其派生 tables.
- 负责装配节点, 权重, 谱矩阵与 basis fields.

Public API:
- Grid

Notes:
- `Grid` 是不可变的 model 层配置对象.
- 纯数学矩阵构造委托给 `veqpy.math`.
- 不负责 source route, residual 组装, 或 solver runtime 状态.
"""

import numpy as np
from rich.console import Console
from rich.tree import Tree

from veqpy.base import Reactive, Serial
from veqpy.engine import (
    RHO_AXIS,
    THETA_AXIS,
    full_differentiation,
    full_integration,
)
from veqpy.math.calculus import make_calculus
from veqpy.math.fast import colwise_weighted_sum_into, dot, rowwise_sum_into
from veqpy.math.quadrature import make_quadrature


class Grid(Reactive, Serial):
    """径向-极角离散化的网格配置."""

    root_properties = {
        "Nr",
        "Nt",
        "L_max",
        "M_max",
        "K_max",
        "quadrature_scheme",
        "calculus_scheme",
    }

    def __init__(
        self,
        Nr: int,
        Nt: int,
        L_max: int = 20,
        M_max: int = 20,
        K_max: int | None = None,
        quadrature_scheme: str = "legendre",
        calculus_scheme: str = "spectral",
    ):
        super().__init__()

        self.Nr = Nr
        self.Nt = Nt
        self.L_max = L_max
        self.M_max = M_max
        self.K_max = K_max
        self.quadrature_scheme = quadrature_scheme
        self.calculus_scheme = calculus_scheme

    def __rich__(self):
        tree = Tree("[bold blue]Grid[/]")
        tree.add(f"Nr: {self.Nr}")
        tree.add(f"Nt: {self.Nt}")
        tree.add(f"quadrature_scheme: {self.quadrature_scheme}")
        tree.add(f"calculus_scheme: {self.calculus_scheme}")
        if self.K_max is not None:
            tree.add(f"K_max: {self.K_max}")
        return tree

    def __str__(self) -> str:
        console = Console(
            color_system=None, force_terminal=False, width=120, record=True, soft_wrap=False
        )
        with console.capture() as capture:
            console.print(self.__rich__())
        return capture.get().rstrip()

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def reactive_inspections(cls, name: str, value):
        match name:
            case "Nr":
                value = int(value)
                if value < 4:
                    raise ValueError("Nr must be at least 4 for stable spectral methods")
                return value

            case "Nt":
                value = int(value)
                if value < 1:
                    raise ValueError("Nt must be positive")
                return value

            case "L_max":
                value = int(value)
                if value < 0:
                    raise ValueError("L_max must be non-negative")
                return value

            case "M_max":
                value = int(value)
                if value < 2:
                    raise ValueError("M_max must be at least 2")
                return value

            case "K_max":
                if value is None:
                    return None
                value = int(value)
                if value < 2:
                    raise ValueError("K_max must be at least 2")
                return value

            case "quadrature_scheme" | "calculus_scheme":
                return str(value).lower()

        return value

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """声明可序列化的根属性."""

        return {
            "Nr": int,
            "Nt": int,
            "quadrature_scheme": str,
            "calculus_scheme": str,
            "L_max": int,
            "M_max": int,
            "K_max": int | None,
        }

    @property
    def quadrature(self) -> tuple[np.ndarray, np.ndarray]:
        rho, weights = make_quadrature(self.Nr, scheme=self.quadrature_scheme)
        return _const_array(rho), _const_array(weights)

    @property
    def rho(self) -> np.ndarray:
        return self.quadrature[0]

    @property
    def weights(self) -> np.ndarray:
        return self.quadrature[1]

    @property
    def theta(self) -> np.ndarray:
        theta = np.linspace(0.0, 2.0 * np.pi, self.Nt, endpoint=False)
        return _const_array(theta)

    @property
    def calculus(self) -> tuple[np.ndarray, np.ndarray]:
        accumulator, differentiator = make_calculus(self.rho, scheme=self.calculus_scheme)
        return _const_array(accumulator), _const_array(differentiator)

    @property
    def accumulator(self) -> np.ndarray:
        return self.calculus[0]

    @property
    def differentiator(self) -> np.ndarray:
        return self.calculus[1]

    def differentiate(
        self,
        f_1D: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """在当前 Grid 上对 1D 场做谱微分."""
        if out is None:
            out = np.empty_like(f_1D)
        return full_differentiation(out, f_1D, self.differentiator)

    def accumulate(
        self,
        f_1D: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """在当前 Grid 上对 1D 场做前缀积分。"""
        if out is None:
            out = np.empty_like(f_1D)
        return full_integration(out, f_1D, self.accumulator)

    def integrate(
        self,
        f: np.ndarray,
        *,
        axis: int | None = None,
        out: np.ndarray | None = None,
    ) -> float | np.ndarray:
        """在当前 Grid 上执行求积."""
        if out is None:
            if axis is None:
                if f.ndim == 1:
                    return dot(f, self.weights)
                if f.ndim != 2:
                    raise ValueError(f"Expected a 1D or 2D array, got shape {f.shape}")
                scratch = np.empty(f.shape[1], dtype=f.dtype)
                colwise_weighted_sum_into(scratch, f, self.weights)
                total = 0.0
                for j in range(f.shape[1]):
                    total += scratch[j]
                return (2.0 * np.pi / f.shape[1]) * total

            if f.ndim != 2:
                raise ValueError(f"Expected a 2D array when axis={axis}, got {f.shape}")
            if axis == RHO_AXIS:
                out = np.empty(f.shape[1], dtype=f.dtype)
            elif axis == THETA_AXIS:
                out = np.empty(f.shape[0], dtype=f.dtype)
            else:
                raise ValueError(f"Unsupported quadrature axis {axis}")

        if axis == RHO_AXIS:
            colwise_weighted_sum_into(out, f, self.weights)
        elif axis == THETA_AXIS:
            nt = f.shape[1]
            rowwise_sum_into(out, f)
            out *= 2.0 * np.pi / nt
        else:
            raise ValueError(f"Unsupported quadrature axis {axis}")
        return out

    @property
    def x(self) -> np.ndarray:
        return _const_array(2.0 * self.rho * self.rho - 1.0)

    @property
    def y(self) -> np.ndarray:
        return _const_array(1.0 - self.rho * self.rho)

    @property
    def T_fields(self) -> np.ndarray:
        return _const_array(_build_chebyshev_tables(self.rho, self.x, self.L_max))

    @property
    def T(self) -> np.ndarray:
        return self.T_fields[0]

    @property
    def T_r(self) -> np.ndarray:
        return self.T_fields[1]

    @property
    def T_rr(self) -> np.ndarray:
        return self.T_fields[2]

    @property
    def _trig_tables(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        harmonics = np.arange(self.M_max + 1, dtype=np.float64)[:, None]
        ktheta = harmonics * self.theta[None, :]
        cos_mtheta = np.cos(ktheta)
        sin_mtheta = np.sin(ktheta)
        m_cos_mtheta = harmonics * cos_mtheta
        m_sin_mtheta = harmonics * sin_mtheta
        k2 = harmonics * harmonics
        m2_cos_mtheta = k2 * cos_mtheta
        m2_sin_mtheta = k2 * sin_mtheta
        return (
            _const_array(cos_mtheta),
            _const_array(sin_mtheta),
            _const_array(m_cos_mtheta),
            _const_array(m_sin_mtheta),
            _const_array(m2_cos_mtheta),
            _const_array(m2_sin_mtheta),
        )

    @property
    def cos_mtheta(self) -> np.ndarray:
        return self._trig_tables[0]

    @property
    def sin_mtheta(self) -> np.ndarray:
        return self._trig_tables[1]

    @property
    def m_cos_mtheta(self) -> np.ndarray:
        return self._trig_tables[2]

    @property
    def m_sin_mtheta(self) -> np.ndarray:
        return self._trig_tables[3]

    @property
    def m2_cos_mtheta(self) -> np.ndarray:
        return self._trig_tables[4]

    @property
    def m2_sin_mtheta(self) -> np.ndarray:
        return self._trig_tables[5]

    @property
    def K_values(self) -> np.ndarray:
        return _const_array(_build_K_values(self.M_max, self.K_max))

    @property
    def rho_powers(self) -> np.ndarray:
        k_values = self.K_values
        max_rho_power = max(2, int(np.max(k_values)) + 1)
        rho_powers = np.empty((max_rho_power + 1, self.Nr), dtype=np.float64)
        rho_powers[0].fill(1.0)
        for power in range(1, max_rho_power + 1):
            rho_powers[power] = self.rho**power
        return _const_array(rho_powers)


def _const_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    array.flags.writeable = False
    return array


def _normalize_optional_int(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _build_K_values(M_max: int, K_max: int | None) -> np.ndarray:
    powers = np.arange(int(M_max) + 1, dtype=np.int64)
    powers[0] = 0
    if K_max is not None:
        powers[1:] = np.minimum(powers[1:], int(K_max))
    return powers


def _build_chebyshev_tables(
    rho: np.ndarray,
    x: np.ndarray,
    L_max: int,
) -> np.ndarray:
    Nr = len(rho)
    T = np.zeros((L_max + 1, Nr), dtype=np.float64)
    Tx = np.zeros((L_max + 1, Nr), dtype=np.float64)
    Txx = np.zeros((L_max + 1, Nr), dtype=np.float64)
    T[0, :] = 1.0
    if L_max >= 1:
        T[1, :] = x
        Tx[1, :] = 1.0

    for k in range(1, L_max):
        T[k + 1, :] = 2.0 * x * T[k, :] - T[k - 1, :]
        Tx[k + 1, :] = 2.0 * T[k, :] + 2.0 * x * Tx[k, :] - Tx[k - 1, :]
        Txx[k + 1, :] = 4.0 * Tx[k, :] + 2.0 * x * Txx[k, :] - Txx[k - 1, :]

    dx_dr = 4.0 * rho
    d2x_dr2 = 4.0
    T_r = Tx * dx_dr[None, :]
    T_rr = Txx * (dx_dr[None, :] ** 2) + Tx * d2x_dr2
    out = np.empty((3, L_max + 1, Nr), dtype=np.float64)
    out[0] = T
    out[1] = T_r
    out[2] = T_rr
    return out
