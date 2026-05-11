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

import warnings
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.tree import Tree

from veqpy.base import Serial
from veqpy.engine import (
    RHO_AXIS,
    THETA_AXIS,
    full_differentiation,
    full_integration,
)
from veqpy.math.calculus import (
    corrected_differentiation_matrix,
    corrected_integration_matrix,
    make_calculus,
)
from veqpy.math.fast import colwise_weighted_sum_into, dot, rowwise_sum_into
from veqpy.math.quadrature import available_quadrature_schemes, make_quadrature


@dataclass(frozen=True, slots=True)
class Grid(Serial):
    """径向-极角离散化的网格配置."""

    Nr: int
    Nt: int
    L_max: int = 20
    M_max: int = 20
    K_max: int | None = None
    scheme: str = "legendre"
    calculus: str = "spectral"

    rho: np.ndarray = field(init=False)
    rho_powers: np.ndarray = field(init=False)
    fourier_radial_powers: np.ndarray = field(init=False)
    theta: np.ndarray = field(init=False)
    cos_ktheta: np.ndarray = field(init=False)
    sin_ktheta: np.ndarray = field(init=False)
    k_cos_ktheta: np.ndarray = field(init=False)
    k_sin_ktheta: np.ndarray = field(init=False)
    k2_cos_ktheta: np.ndarray = field(init=False)
    k2_sin_ktheta: np.ndarray = field(init=False)

    weights: np.ndarray = field(init=False)
    integration_matrix: np.ndarray = field(init=False, default=None)
    differentiation_matrix: np.ndarray = field(init=False, default=None)
    odd_integration_matrix: np.ndarray = field(init=False, default=None)
    even_integration_matrix: np.ndarray = field(init=False, default=None)
    odd_differentiation_matrix: np.ndarray = field(init=False, default=None)
    even_differentiation_matrix: np.ndarray = field(init=False, default=None)
    ff_r_regularization_matrix: np.ndarray = field(init=False, default=None)

    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    T_fields: np.ndarray = field(init=False)

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """声明可序列化的根属性."""

        return {
            "Nr": int,
            "Nt": int,
            "scheme": str,
            "calculus": str,
            "L_max": int,
            "M_max": int,
            "K_max": int | None,
        }

    def __post_init__(self):
        """根据根参数构造网格与谱表."""
        scheme = self.scheme.lower()
        if scheme not in available_quadrature_schemes():
            raise ValueError(f"Unknown grid scheme: {scheme}")
        object.__setattr__(self, "scheme", scheme)
        calculus = "compact" if self.calculus is None else self.calculus.lower()
        object.__setattr__(self, "calculus", calculus)

        if scheme == "lobatto":
            warnings.warn(
                "'lobatto' grid scheme is deprecated.",
                FutureWarning,
                stacklevel=2,
            )

        if self.Nr < 4:
            raise ValueError("Nr must be at least 4 for stable spectral methods")
        if self.Nt < 1:
            raise ValueError("Nt must be positive")
        if self.L_max < 0:
            raise ValueError("L_max must be non-negative")
        if self.M_max < 2:
            raise ValueError("M_max must be at least 2")
        K_max = _normalize_fourier_power_K_max(self.K_max)
        object.__setattr__(self, "K_max", K_max)

        rho, weights = make_quadrature(self.Nr, scheme=scheme)
        theta = np.linspace(0.0, 2.0 * np.pi, self.Nt, endpoint=False)
        harmonics = np.arange(self.M_max + 1, dtype=np.float64)[:, None]
        ktheta = harmonics * theta[None, :]
        cos_ktheta = np.cos(ktheta)
        sin_ktheta = np.sin(ktheta)
        k_cos_ktheta = harmonics * cos_ktheta
        k_sin_ktheta = harmonics * sin_ktheta
        k2 = harmonics * harmonics
        k2_cos_ktheta = k2 * cos_ktheta
        k2_sin_ktheta = k2 * sin_ktheta
        rho2 = rho * rho
        fourier_radial_powers = _build_fourier_radial_powers(self.M_max, K_max)
        max_rho_power = max(2, int(np.max(fourier_radial_powers)) + 1)
        rho_powers = np.empty((max_rho_power + 1, self.Nr), dtype=np.float64)
        rho_powers[0].fill(1.0)
        for power in range(1, max_rho_power + 1):
            rho_powers[power] = rho**power
        x = 2.0 * rho2 - 1.0
        y = 1.0 - rho2

        integration_matrix, differentiation_matrix = make_calculus(
            rho,
            calculus=calculus,
        )
        odd_integration_matrix = corrected_integration_matrix(
            rho,
            differentiation_matrix,
            p=1,
        )
        even_integration_matrix = corrected_integration_matrix(
            rho,
            differentiation_matrix,
            p=2,
        )
        corrected_linear_derivative = corrected_differentiation_matrix(
            rho,
            differentiation_matrix,
            p=1,
        )
        corrected_even_derivative = corrected_differentiation_matrix(
            rho,
            differentiation_matrix,
            p=2,
        )
        ff_r_regularization_matrix = _build_ff_r_regularization_matrix(rho)

        T_fields = _build_chebyshev_tables(rho, x, self.L_max)

        trig_tables = (
            np.asarray(cos_ktheta, dtype=np.float64),
            np.asarray(sin_ktheta, dtype=np.float64),
            np.asarray(k_cos_ktheta, dtype=np.float64),
            np.asarray(k_sin_ktheta, dtype=np.float64),
            np.asarray(k2_cos_ktheta, dtype=np.float64),
            np.asarray(k2_sin_ktheta, dtype=np.float64),
        )
        for table in trig_tables:
            table.flags.writeable = False

        object.__setattr__(self, "rho", np.asarray(rho, dtype=np.float64))
        rho_powers.flags.writeable = False
        object.__setattr__(self, "rho_powers", rho_powers)
        fourier_radial_powers.flags.writeable = False
        object.__setattr__(self, "fourier_radial_powers", fourier_radial_powers)
        object.__setattr__(self, "theta", np.asarray(theta, dtype=np.float64))
        object.__setattr__(self, "cos_ktheta", trig_tables[0])
        object.__setattr__(self, "sin_ktheta", trig_tables[1])
        object.__setattr__(self, "k_cos_ktheta", trig_tables[2])
        object.__setattr__(self, "k_sin_ktheta", trig_tables[3])
        object.__setattr__(self, "k2_cos_ktheta", trig_tables[4])
        object.__setattr__(self, "k2_sin_ktheta", trig_tables[5])
        object.__setattr__(self, "weights", np.asarray(weights, dtype=np.float64))
        object.__setattr__(
            self, "integration_matrix", np.asarray(integration_matrix, dtype=np.float64)
        )
        object.__setattr__(
            self, "differentiation_matrix", np.asarray(differentiation_matrix, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "odd_integration_matrix",
            np.asarray(odd_integration_matrix, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "even_integration_matrix",
            np.asarray(even_integration_matrix, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "odd_differentiation_matrix",
            np.asarray(corrected_linear_derivative, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "even_differentiation_matrix",
            np.asarray(corrected_even_derivative, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "ff_r_regularization_matrix",
            np.asarray(ff_r_regularization_matrix, dtype=np.float64),
        )
        object.__setattr__(self, "x", np.asarray(x, dtype=np.float64))
        object.__setattr__(self, "y", np.asarray(y, dtype=np.float64))
        object.__setattr__(self, "T_fields", np.asarray(T_fields, dtype=np.float64))
        self.integration_matrix.flags.writeable = False
        self.differentiation_matrix.flags.writeable = False
        self.odd_integration_matrix.flags.writeable = False
        self.even_integration_matrix.flags.writeable = False
        self.odd_differentiation_matrix.flags.writeable = False
        self.even_differentiation_matrix.flags.writeable = False
        self.ff_r_regularization_matrix.flags.writeable = False

    def __rich__(self):
        tree = Tree("[bold blue]Grid[/]")
        tree.add(f"Nr: {self.Nr}")
        tree.add(f"Nt: {self.Nt}")
        tree.add(f"scheme: {self.scheme}")
        tree.add(f"calculus: {self.calculus}")
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

    def resolve_fourier_power(self, order: int) -> int:
        """返回当前 Grid 规则下 Fourier shape profile 的径向 prefactor 幂次."""
        order = int(order)
        if order <= 0:
            return 0
        if order <= self.M_max:
            return int(self.fourier_radial_powers[order])
        if self.K_max is None:
            return order
        return min(order, int(self.K_max))

    @property
    def T(self) -> np.ndarray:
        return self.T_fields[0]

    @property
    def T_r(self) -> np.ndarray:
        return self.T_fields[1]

    @property
    def T_rr(self) -> np.ndarray:
        return self.T_fields[2]

    def differentiate(self, f_1D: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        """在当前 Grid 上对 1D 场做谱微分."""
        if out is None:
            out = np.empty_like(f_1D)
        return full_differentiation(out, f_1D, self.differentiation_matrix)

    def integrate(
        self, f_1D: np.ndarray, *, p: int | None = None, out: np.ndarray | None = None
    ) -> np.ndarray:
        """在当前 Grid 上对 1D 场做积分."""
        if out is None:
            out = np.empty_like(f_1D)
        if p is None:
            return full_integration(out, f_1D, self.integration_matrix)
        if p == 1:
            np.matmul(self.odd_integration_matrix, f_1D, out=out)
            return out
        if p == 2:
            np.matmul(self.even_integration_matrix, f_1D, out=out)
            return out

        matrix = corrected_integration_matrix(
            self.rho,
            self.differentiation_matrix,
            p=p,
        )
        np.matmul(matrix, f_1D, out=out)
        return out

    def corrected_linear_derivative(
        self, f_1D: np.ndarray, *, out: np.ndarray | None = None
    ) -> np.ndarray:
        """在当前 Grid 上对轴心线性起始量做预计算修正微分."""
        if out is None:
            out = np.empty_like(f_1D)
        np.matmul(self.odd_differentiation_matrix, f_1D, out=out)
        return out

    def corrected_even_derivative(
        self, f_1D: np.ndarray, *, out: np.ndarray | None = None
    ) -> np.ndarray:
        """在当前 Grid 上对轴心偶函数量做预计算修正微分."""
        if out is None:
            out = np.empty_like(f_1D)
        if not np.all(np.isfinite(f_1D)):
            out.fill(0.0)
            return out
        np.matmul(self.even_differentiation_matrix, f_1D, out=out)
        return out

    def regularize_ff_r(self, f_1D: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        """将 FF_r 投影到共享的轴心/边界正则基底上."""
        if out is None:
            out = np.empty_like(f_1D)
        np.matmul(self.ff_r_regularization_matrix, f_1D, out=out)
        return out

    def quadrature(
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


def _build_ff_r_regularization_matrix(rho: np.ndarray, *, degree: int = 3) -> np.ndarray:
    s = np.asarray(rho, dtype=np.float64) ** 2
    basis = np.column_stack([rho * (1.0 - s) * (s**k) for k in range(degree + 1)])
    return basis @ np.linalg.pinv(basis)


def _normalize_fourier_power_K_max(value: int | None) -> int | None:
    """Normalize and validate a Fourier radial-power cap owned by Grid."""
    if value is None:
        return None
    K_max = int(value)
    if K_max < 1:
        raise ValueError(f"K_max must be None or an integer >= 1, got {value!r}")
    return K_max


def _build_fourier_radial_powers(M_max: int, K_max: int | None) -> np.ndarray:
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
