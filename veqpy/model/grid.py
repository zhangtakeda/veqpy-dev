"""
Module: model.grid

Role:
- 负责持有径向-极角网格配置及其派生 tables.
- 负责生成节点, 权重, 谱矩阵与 basis fields.

Public API:
- Grid

Notes:
- `Grid` 是不可变的 model 层配置对象.
- 不负责 source route, residual 组装, 或 solver runtime 状态.
"""

from dataclasses import dataclass, field
from typing import Literal
import warnings

import numpy as np
from rich.console import Console
from rich.tree import Tree
from scipy.linalg import eigh_tridiagonal

from veqpy.engine import (
    RHO_AXIS,
    THETA_AXIS,
    corrected_integration,
    full_differentiation,
    full_integration,
    theta_reduction,
    quadrature,
)
from veqpy.model.serial import Serial


@dataclass(frozen=True, slots=True)
class Grid(Serial):
    """径向-极角离散化的网格配置."""

    Nr: int
    Nt: int
    scheme: Literal["legendre", "chebyshev", "lobatto", "radau", "uniform"]
    L_max: int = 21

    rho: np.ndarray = field(init=False)
    rho2: np.ndarray = field(init=False)
    theta: np.ndarray = field(init=False)
    cos_theta: np.ndarray = field(init=False)
    sin_theta: np.ndarray = field(init=False)
    cos_2theta: np.ndarray = field(init=False)
    sin_2theta: np.ndarray = field(init=False)

    weights: np.ndarray = field(init=False)
    integration_matrix: np.ndarray = field(init=False, default=None)
    differentiation_matrix: np.ndarray = field(init=False, default=None)

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
            "L_max": int,
        }

    def __post_init__(self):
        """根据根参数构造网格与谱表."""
        scheme = self.scheme.lower()
        if scheme not in {"legendre", "chebyshev", "lobatto", "radau", "uniform"}:
            raise ValueError(f"Unknown grid scheme: {scheme}")
        object.__setattr__(self, "scheme", scheme)

        if scheme == "lobatto":
            warnings.warn(
                "Grid scheme 'lobatto' is deprecated and is not benchmark-stable in the current implementation.",
                FutureWarning,
                stacklevel=2,
            )

        if self.Nr < 4:
            raise ValueError("Nr must be at least 4 for stable spectral methods")
        if self.Nt < 1:
            raise ValueError("Nt must be positive")
        if self.L_max < 0:
            raise ValueError("L_max must be non-negative")

        rho, weights = _build_rho_and_weights(self.Nr, scheme)
        theta = np.linspace(0.0, 2.0 * np.pi, self.Nt, endpoint=False)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_2theta = cos_theta * cos_theta - sin_theta * sin_theta
        sin_2theta = 2.0 * sin_theta * cos_theta
        rho2 = rho * rho
        x = 2.0 * rho2 - 1.0
        y = 1.0 - rho2

        if scheme == "uniform":
            integration_matrix = _build_uniform_integration_matrix(self.Nr)
            differentiation_matrix = _build_uniform_differentiation_matrix(self.Nr)
        else:
            integration_matrix = _build_integration_matrix(rho)
            differentiation_matrix = _build_differentiation_matrix(rho)

        T_fields = _build_chebyshev_tables(rho, x, self.L_max)

        object.__setattr__(self, "rho", np.asarray(rho, dtype=np.float64))
        object.__setattr__(self, "rho2", np.asarray(rho2, dtype=np.float64))
        object.__setattr__(self, "theta", np.asarray(theta, dtype=np.float64))
        object.__setattr__(self, "cos_theta", np.asarray(cos_theta, dtype=np.float64))
        object.__setattr__(self, "sin_theta", np.asarray(sin_theta, dtype=np.float64))
        object.__setattr__(self, "cos_2theta", np.asarray(cos_2theta, dtype=np.float64))
        object.__setattr__(self, "sin_2theta", np.asarray(sin_2theta, dtype=np.float64))
        object.__setattr__(self, "weights", np.asarray(weights, dtype=np.float64))
        object.__setattr__(self, "integration_matrix", np.asarray(integration_matrix, dtype=np.float64))
        object.__setattr__(self, "differentiation_matrix", np.asarray(differentiation_matrix, dtype=np.float64))
        object.__setattr__(self, "x", np.asarray(x, dtype=np.float64))
        object.__setattr__(self, "y", np.asarray(y, dtype=np.float64))
        object.__setattr__(self, "T_fields", np.asarray(T_fields, dtype=np.float64))

    @property
    def T(self) -> np.ndarray:
        return self.T_fields[0]

    @property
    def T_r(self) -> np.ndarray:
        return self.T_fields[1]

    @property
    def T_rr(self) -> np.ndarray:
        return self.T_fields[2]

    def __rich__(self):
        tree = Tree("[bold blue]Grid[/]")
        tree.add(f"Nr: {self.Nr}")
        tree.add(f"Nt: {self.Nt}")
        tree.add(f"scheme: {self.scheme}")
        return tree

    def __str__(self) -> str:
        console = Console(color_system=None, force_terminal=False, width=120, record=True, soft_wrap=False)
        with console.capture() as capture:
            console.print(self.__rich__())
        return capture.get().rstrip()

    def __repr__(self) -> str:
        return str(self)

    def differentiate(self, f_1D: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
        """在当前 Grid 上对 1D 场做谱微分."""
        if out is None:
            out = np.empty_like(f_1D)
        return full_differentiation(out, f_1D, self.differentiation_matrix)

    def integrate(self, f_1D: np.ndarray, *, p: int | None = None, out: np.ndarray | None = None) -> np.ndarray:
        """在当前 Grid 上对 1D 场做积分."""
        if out is None:
            out = np.empty_like(f_1D)
        if p is None:
            return full_integration(out, f_1D, self.integration_matrix)

        return corrected_integration(
            out,
            f_1D,
            self.integration_matrix,
            p=p,
            rho=self.rho,
            differentiation_matrix=self.differentiation_matrix,
        )

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
                return quadrature(f, self.weights)

            if f.ndim != 2:
                raise ValueError(f"Expected a 2D array when axis={axis}, got {f.shape}")
            if axis == RHO_AXIS:
                out = np.empty(f.shape[1], dtype=f.dtype)
            elif axis == THETA_AXIS:
                out = np.empty(f.shape[0], dtype=f.dtype)
            else:
                raise ValueError(f"Unsupported quadrature axis {axis}")

        return theta_reduction(out, f, self.weights, axis)


def _build_rho_and_weights(Nr: int, scheme: str) -> tuple[np.ndarray, np.ndarray]:
    if scheme == "legendre":
        nodes, w = _golub_welsch_legendre(Nr)
        rho = 0.5 * (nodes + 1.0)
        weights = 0.5 * w
        return np.asarray(rho, dtype=np.float64), np.asarray(weights, dtype=np.float64)

    if scheme == "chebyshev":
        k = np.arange(1, Nr + 1, dtype=np.float64)
        xg = np.cos((2.0 * k - 1.0) * np.pi / (2.0 * Nr))[::-1]
        rho = 0.5 * (xg + 1.0)
        weights = _quadrature_weights(rho)
        return np.asarray(rho, dtype=np.float64), np.asarray(weights, dtype=np.float64)

    if scheme == "lobatto":
        nodes, w = _golub_welsch_lobatto(Nr)
        rho = 0.5 * (nodes + 1.0)
        weights = 0.5 * w
        return np.asarray(rho, dtype=np.float64), np.asarray(weights, dtype=np.float64)

    if scheme == "radau":
        nodes, w = _golub_welsch_radau(Nr)
        rho = 0.5 * (nodes + 1.0)
        weights = 0.5 * w
        return np.asarray(rho, dtype=np.float64), np.asarray(weights, dtype=np.float64)

    rho = np.linspace(0.0, 1.0, Nr)
    h = 1.0 / (Nr - 1)
    weights = np.full(Nr, h, dtype=np.float64)
    weights[0] = 0.5 * h
    weights[-1] = 0.5 * h
    return rho, weights


def _quadrature_weights(rho: np.ndarray) -> np.ndarray:
    n = len(rho)
    V = np.polynomial.legendre.legvander(2.0 * rho - 1.0, n - 1)
    int_P = np.zeros(n, dtype=np.float64)
    int_P[0] = 2.0
    return 0.5 * np.linalg.solve(V.T, int_P)


def _legendre_jacobi_coeffs(n: int) -> tuple[np.ndarray, np.ndarray]:
    diag = np.zeros(n, dtype=np.float64)
    k = np.arange(1, n, dtype=np.float64)
    offdiag = k / np.sqrt(4.0 * k * k - 1.0)
    return diag, offdiag


def _golub_welsch_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    diag, offdiag = _legendre_jacobi_coeffs(n)
    nodes, vecs = eigh_tridiagonal(diag, offdiag)
    weights = 2.0 * vecs[0, :] ** 2
    return nodes, weights


def _golub_welsch_radau(n: int) -> tuple[np.ndarray, np.ndarray]:
    diag, offdiag = _legendre_jacobi_coeffs(n)
    diag[-1] = n / (2.0 * n - 1.0)
    nodes, vecs = eigh_tridiagonal(diag, offdiag)
    weights = 2.0 * vecs[0, :] ** 2
    return nodes, weights


def _golub_welsch_lobatto(n: int) -> tuple[np.ndarray, np.ndarray]:
    diag = np.zeros(n, dtype=np.float64)
    offdiag = np.empty(n - 1, dtype=np.float64)
    k = np.arange(1, n - 1, dtype=np.float64)
    offdiag[:-1] = k / np.sqrt(4.0 * k * k - 1.0)
    offdiag[-1] = np.sqrt((n - 1.0) / (2.0 * n - 3.0))
    nodes, vecs = eigh_tridiagonal(diag, offdiag)
    weights = 2.0 * vecs[0, :] ** 2
    return nodes, weights


def _barycentric_log_weights(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = x[:, None] - x[None, :]
    mask = ~np.eye(len(x), dtype=bool)
    if np.any(np.isclose(diff[mask], 0.0)):
        raise ValueError("rho nodes must be distinct")

    abs_diff = np.abs(diff)
    sign_diff = np.sign(diff)
    np.fill_diagonal(abs_diff, 1.0)
    np.fill_diagonal(sign_diff, 1.0)

    log_abs_prod = np.sum(np.log(abs_diff), axis=1)
    signs = np.prod(sign_diff, axis=1)
    return signs, -log_abs_prod


def _build_differentiation_matrix(rho: np.ndarray) -> np.ndarray:
    n = len(rho)
    signs, logw = _barycentric_log_weights(rho)
    diff = rho[:, None] - rho[None, :]
    np.fill_diagonal(diff, 1.0)
    log_ratio = logw[None, :] - logw[:, None]
    mag_ratio = np.exp(np.clip(log_ratio, -700.0, 700.0))
    sign_ratio = signs[None, :] * signs[:, None]
    D = sign_ratio * mag_ratio / diff
    np.fill_diagonal(D, 0.0)
    D[np.diag_indices(n)] = -np.sum(D, axis=1)
    return D


def _build_integration_matrix(rho: np.ndarray) -> np.ndarray:
    n = len(rho)
    xg = 2.0 * rho - 1.0
    Pfull = np.polynomial.legendre.legvander(xg, n)
    P = Pfull[:, :n]

    S = np.zeros((n, n), dtype=np.float64)
    S[:, 0] = 0.5 * (xg + 1.0)
    m = np.arange(1, n, dtype=np.float64)
    S[:, 1:] = 0.5 * (Pfull[:, 2 : n + 1] - Pfull[:, : n - 1]) / (2.0 * m + 1.0)

    x_a = -1.0
    Pa = np.polynomial.legendre.legvander(np.array([x_a]), n)[0]
    sa = np.zeros(n, dtype=np.float64)
    sa[0] = 0.5 * (x_a + 1.0)
    sa[1:] = 0.5 * (Pa[2 : n + 1] - Pa[: n - 1]) / (2.0 * m + 1.0)

    rhs = (S - sa[None, :]).T
    cond = np.linalg.cond(P)
    eps = np.finfo(np.float64).eps

    if np.isfinite(cond) and cond <= 1.0 / np.sqrt(eps):
        return np.linalg.solve(P.T, rhs).T

    return np.linalg.lstsq(P.T, rhs, rcond=None)[0].T


def _build_uniform_differentiation_matrix(Nr: int) -> np.ndarray:
    h = 1.0 / (Nr - 1)
    D = np.zeros((Nr, Nr), dtype=np.float64)
    D[0, 0] = -3.0 / (2.0 * h)
    D[0, 1] = 4.0 / (2.0 * h)
    D[0, 2] = -1.0 / (2.0 * h)
    for i in range(1, Nr - 1):
        D[i, i - 1] = -1.0 / (2.0 * h)
        D[i, i + 1] = 1.0 / (2.0 * h)
    D[-1, -3] = 1.0 / (2.0 * h)
    D[-1, -2] = -4.0 / (2.0 * h)
    D[-1, -1] = 3.0 / (2.0 * h)
    return D


def _build_uniform_integration_matrix(Nr: int) -> np.ndarray:
    h = 1.0 / (Nr - 1)
    Q = np.zeros((Nr, Nr), dtype=np.float64)
    for i in range(1, Nr):
        Q[i, 0] = 0.5 * h
        Q[i, i] = 0.5 * h
        if i > 1:
            Q[i, 1:i] = h
    return Q


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
