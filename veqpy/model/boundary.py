"""
Boundary parameter aggregate and GEQDSK-to-Boundary fitting helpers.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.text import Text
from rich.tree import Tree
from scipy.optimize import least_squares

from veqpy.model.geqdsk import Geqdsk

MAX_FOURIER_ORDER = 10


@dataclass(slots=True, frozen=True)
class Boundary:
    a: float
    R0: float
    Z0: float
    B0: float
    ka: float = 1.0
    c_offsets: np.ndarray | None = None
    s_offsets: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", float(self.a))
        object.__setattr__(self, "R0", float(self.R0))
        object.__setattr__(self, "Z0", float(self.Z0))
        object.__setattr__(self, "B0", float(self.B0))
        object.__setattr__(self, "ka", float(self.ka))
        object.__setattr__(self, "c_offsets", _normalize_offset_array(self.c_offsets, name="c_offsets"))
        object.__setattr__(self, "s_offsets", _normalize_offset_array(self.s_offsets, name="s_offsets"))

    def __rich__(self):
        tree = Tree("[bold blue]Boundary[/]")
        tree.add(Text(f"a: {self.a:.3f} [m]"))
        tree.add(Text(f"R0: {self.R0:.3f} [m]"))
        tree.add(Text(f"Z0: {self.Z0:.3f} [m]"))
        tree.add(f"B0: {self.B0:.3f} [T]")
        tree.add(f"ka: {self.ka:.3f}")
        if np.any(self.c_offsets != 0.0):
            tree.add(f"c_offsets: {np.array2string(self.c_offsets, precision=3, separator=', ')}")
        if np.any(self.s_offsets != 0.0):
            tree.add(f"s_offsets: {np.array2string(self.s_offsets, precision=3, separator=', ')}")
        return tree

    def __str__(self) -> str:
        console = Console(color_system=None, force_terminal=False, width=120, record=True, soft_wrap=False)
        with console.capture() as capture:
            console.print(self.__rich__())
        return capture.get().rstrip()

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def from_geqdsk(
        cls,
        geqdsk: Geqdsk,
        *,
        M: int | None = None,
        N: int | None = None,
        maxtol: float = 1.0e-2,
        R0: float | None = None,
        Z0: float | None = None,
        a: float | None = None,
        ka: float | None = None,
    ) -> Boundary:
        if not isinstance(geqdsk, Geqdsk):
            raise TypeError(f"geqdsk must be Geqdsk, got {type(geqdsk).__name__}")
        params = _fit_boundary_params(geqdsk, M=M, N=N, maxtol=maxtol, R0=R0, Z0=Z0, a=a, ka=ka)
        return cls(
            a=float(params["a"]),
            R0=float(params["R0"]),
            Z0=float(params["Z0"]),
            B0=float(geqdsk.Bt0),
            ka=float(params["ka"]),
            c_offsets=np.asarray(params["c_offsets"], dtype=np.float64),
            s_offsets=np.asarray(params["s_offsets"], dtype=np.float64),
        )


def _fit_boundary_params(
    geqdsk: Geqdsk,
    *,
    M: int | None,
    N: int | None,
    maxtol: float,
    R0: float | None,
    Z0: float | None,
    a: float | None,
    ka: float | None,
) -> dict[str, float | np.ndarray]:
    if not isinstance(geqdsk, Geqdsk):
        raise TypeError(f"geqdsk must be Geqdsk, got {type(geqdsk).__name__}")

    maxtol = float(maxtol)
    if maxtol <= 0.0:
        raise ValueError(f"maxtol must be positive, got {maxtol!r}")

    if (M is None) != (N is None):
        raise ValueError("M and N must be provided together or both omitted")
    if M is None and N is None:
        return _fit_minimal_order_boundary(geqdsk, maxtol=maxtol, R0=R0, Z0=Z0, a=a, ka=ka)

    assert M is not None and N is not None
    params = _fit_boundary_for_orders(geqdsk, M=M, N=N, R0=R0, Z0=Z0, a=a, ka=ka)
    if params["rms"] >= maxtol:
        warnings.warn(
            (
                f"Boundary fit with fixed M/N={M}/{N} did not satisfy maxtol={maxtol:.6e}; "
                f"got rms={float(params['rms']):.6e}"
            ),
            stacklevel=2,
        )
    return params


def _fit_minimal_order_boundary(
    geqdsk: Geqdsk,
    *,
    maxtol: float,
    R0: float | None,
    Z0: float | None,
    a: float | None,
    ka: float | None,
) -> dict[str, float | np.ndarray]:
    best = None
    curve_tol = max(maxtol * 0.25, 1.0e-6)

    for step in range(MAX_FOURIER_ORDER):
        M = step
        N = step + 1
        params = _fit_boundary_for_orders(geqdsk, M=M, N=N, R0=R0, Z0=Z0, a=a, ka=ka)
        if best is None or params["rms"] < best["rms"]:
            best = params
        if params["rms"] < maxtol and params["max_curve_error"] < curve_tol:
            return params

    if best is None:
        raise RuntimeError("Boundary fitting failed to produce any candidate.")
    return best


def _fit_boundary_for_orders(
    geqdsk: Geqdsk,
    *,
    M: int,
    N: int,
    R0: float | None,
    Z0: float | None,
    a: float | None,
    ka: float | None,
) -> dict[str, float | np.ndarray]:
    _validate_orders(M, N)

    if geqdsk.boundary.size == 0:
        raise ValueError("Boundary is empty. Read GEQDSK first.")

    R = geqdsk.boundary[:, 0].astype(np.float64)
    Z = geqdsk.boundary[:, 1].astype(np.float64)

    r_min = float(np.nanmin(R))
    r_max = float(np.nanmax(R))
    z_min = float(np.nanmin(Z))
    z_max = float(np.nanmax(Z))
    r_mid = 0.5 * (r_max + r_min)
    z_mid = 0.5 * (z_max + z_min)
    span_r = r_max - r_min
    span_z = z_max - z_min

    initial_R0 = float(R0) if R0 is not None else r_mid
    initial_Z0 = float(Z0) if Z0 is not None else z_mid
    initial_a = float(a) if a is not None else 0.5 * span_r
    if initial_a <= 0.0:
        raise ValueError("Boundary width must be positive")
    ka0 = max(float(ka) if ka is not None else float(0.5 * span_z / initial_a), 1.0e-6)

    def ordered_boundary_variants():
        start = int(np.argmin(Z))
        r_ordered = np.roll(R, -start)
        z_ordered = np.roll(Z, -start)
        yield r_ordered, z_ordered
        yield np.concatenate(([r_ordered[0]], r_ordered[:0:-1])), np.concatenate(([z_ordered[0]], z_ordered[:0:-1]))

    lower_bounds = [r_min - 0.25 * span_r, z_min - 0.25 * span_z, max(1.0e-6, 0.25 * initial_a), 1.0e-6]
    upper_bounds = [r_max + 0.25 * span_r, z_max + 0.25 * span_z, max(4.0 * initial_a, span_z, 1.0), 10.0]
    lower_bounds.extend([-10.0] * (M + 1))
    upper_bounds.extend([10.0] * (M + 1))
    lower_bounds.extend([-10.0] * N)
    upper_bounds.extend([10.0] * N)
    bounds = (np.asarray(lower_bounds, dtype=np.float64), np.asarray(upper_bounds, dtype=np.float64))

    def pack_params(params: dict[str, float | np.ndarray]) -> np.ndarray:
        vector = [
            float(params["R0"]),
            float(params["Z0"]),
            float(params["a"]),
            float(params["ka"]),
        ]
        vector.extend(np.asarray(params["c_offsets"], dtype=np.float64).tolist())
        if N > 0:
            vector.extend(np.asarray(params["s_offsets"], dtype=np.float64)[1:].tolist())
        return np.asarray(vector, dtype=np.float64)

    def unpack_params(vector: np.ndarray) -> dict[str, float | np.ndarray]:
        idx = 0
        params = {
            "R0": float(vector[idx]),
            "Z0": float(vector[idx + 1]),
            "a": float(vector[idx + 2]),
            "ka": float(vector[idx + 3]),
        }
        idx += 4
        c_offsets = np.asarray(vector[idx : idx + M + 1], dtype=np.float64).copy()
        idx += M + 1
        s_offsets = np.zeros(N + 1, dtype=np.float64)
        if N > 0:
            s_offsets[1:] = np.asarray(vector[idx : idx + N], dtype=np.float64)
        params["c_offsets"] = c_offsets
        params["s_offsets"] = s_offsets
        return params

    best_fit = None
    for r_points, z_points in ordered_boundary_variants():
        start = {
            "R0": initial_R0,
            "Z0": initial_Z0,
            "a": initial_a,
            "ka": ka0,
            "c_offsets": np.zeros(M + 1, dtype=np.float64),
            "s_offsets": np.zeros(N + 1, dtype=np.float64),
        }

        def residual(vector: np.ndarray) -> np.ndarray:
            params = unpack_params(vector)
            theta = _infer_theta(z_points, float(params["Z0"]), float(params["a"]), float(params["ka"]))
            fitted_boundary = _build_boundary(
                R0=float(params["R0"]),
                Z0=float(params["Z0"]),
                a=float(params["a"]),
                ka=float(params["ka"]),
                c_offsets=np.asarray(params["c_offsets"], dtype=np.float64),
                s_offsets=np.asarray(params["s_offsets"], dtype=np.float64),
                theta=theta,
            )
            r_res = r_points - fitted_boundary[:, 0]
            z_res = z_points - fitted_boundary[:, 1]
            return np.concatenate((r_res, z_res))

        start["a"] = max(float(start["a"]), bounds[0][2])
        start["ka"] = max(float(start["ka"]), bounds[0][3])
        fit = least_squares(
            residual,
            x0=np.clip(pack_params(start), bounds[0], bounds[1]),
            bounds=bounds,
            method="trf",
        )
        params = unpack_params(fit.x)
        theta = _infer_theta(z_points, float(params["Z0"]), float(params["a"]), float(params["ka"]))
        fitted_boundary = _build_boundary(
            R0=float(params["R0"]),
            Z0=float(params["Z0"]),
            a=float(params["a"]),
            ka=float(params["ka"]),
            c_offsets=np.asarray(params["c_offsets"], dtype=np.float64),
            s_offsets=np.asarray(params["s_offsets"], dtype=np.float64),
            theta=theta,
        )
        rms = float(np.sqrt(np.mean(fit.fun**2)))
        max_curve_error = _max_bidirectional_distance(np.column_stack((r_points, z_points)), fitted_boundary)
        if best_fit is None or rms < best_fit["rms"]:
            best_fit = {"rms": rms, "params": params, "max_curve_error": max_curve_error}

    fitted = best_fit["params"]
    c_offsets = np.asarray(fitted["c_offsets"], dtype=np.float64).copy()
    s_offsets = np.asarray(fitted["s_offsets"], dtype=np.float64).copy()
    c_offsets[0] = float((c_offsets[0] + np.pi) % (2.0 * np.pi) - np.pi)
    if s_offsets.size > 0:
        s_offsets[0] = 0.0

    return {
        "R0": float(fitted["R0"]),
        "Z0": float(fitted["Z0"]),
        "a": float(fitted["a"]),
        "ka": float(fitted["ka"]),
        "c_offsets": c_offsets,
        "s_offsets": s_offsets,
        "rms": float(best_fit["rms"]),
        "max_curve_error": float(best_fit["max_curve_error"]),
        "M": int(M),
        "N": int(N),
    }


def _validate_orders(M: int, N: int) -> None:
    if int(M) < 0 or int(N) < 0:
        raise ValueError(f"M and N must be non-negative, got M={M!r}, N={N!r}")
    if int(M) > MAX_FOURIER_ORDER or int(N) > MAX_FOURIER_ORDER:
        raise ValueError(f"M and N must be <= {MAX_FOURIER_ORDER}, got M={M!r}, N={N!r}")


def _as_1d_array(value: np.ndarray | list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr


def _normalize_offset_array(value, *, name: str) -> np.ndarray:
    if value is None:
        return _default_offset_array(name)
    arr = _as_1d_array(value, name=name).copy()
    if arr.size == 0:
        raise ValueError(f"{name} must have at least one entry")
    if name == "s_offsets":
        arr[0] = 0.0
    return arr


def _default_offset_array(name: str) -> np.ndarray:
    if name in {"c_offsets", "s_offsets"}:
        return np.zeros(1, dtype=np.float64)
    raise KeyError(f"Unknown offset array {name!r}")


def _infer_theta(z_points: np.ndarray, z0: float, a_value: float, ka: float) -> np.ndarray:
    sin_theta = np.clip(-(z_points - z0) / (a_value * max(float(ka), 1.0e-6)), -1.0, 1.0)
    theta = np.empty_like(sin_theta)
    theta[0] = 0.5 * np.pi
    previous = theta[0]
    step = 2.0 * np.pi / max(len(z_points), 1)

    for index in range(1, len(z_points)):
        alpha = np.arcsin(sin_theta[index])
        candidates = []
        for candidate in (alpha, np.pi - alpha):
            while candidate < previous - 1.0e-12:
                candidate += 2.0 * np.pi
            candidates.extend((candidate, candidate + 2.0 * np.pi))
        target = previous + step
        theta[index] = min(candidates, key=lambda value: abs(value - target))
        previous = theta[index]

    return theta


def _max_bidirectional_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    diff = points_a[:, None, :] - points_b[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def _build_boundary(
    *,
    R0: float,
    Z0: float,
    a: float,
    ka: float,
    c_offsets: np.ndarray,
    s_offsets: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    theta_bar = theta + c_offsets[0]
    for order in range(1, c_offsets.shape[0]):
        theta_bar += c_offsets[order] * np.cos(order * theta)
    for order in range(1, s_offsets.shape[0]):
        theta_bar += s_offsets[order] * np.sin(order * theta)
    R = R0 + a * np.cos(theta_bar)
    Z = Z0 - a * ka * np.sin(theta)
    return np.column_stack((R, Z))
