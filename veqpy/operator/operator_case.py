"""
Module: operator.operator_case

Role:
- 负责把算例输入规范化为稳定的 case 配置对象.

Public API:
- OperatorCase

Notes:
- `OperatorCase` 只保存 case 输入.
- 不负责 layout 构造, residual 计算, 或 solver 策略管理.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from rich.console import Console
from rich.tree import Tree

from veqpy.model.boundary import Boundary

ProfileCoeffInput = list[float] | np.ndarray | int | None
ProfileCoeff = np.ndarray | None


@dataclass(slots=True)
class OperatorCase:
    """描述一次 operator 求值所需的静态 case 输入."""

    route: str
    coordinate: str
    profile_coeffs: dict[str, ProfileCoeffInput]
    boundary: Boundary
    heat_input: np.ndarray
    current_input: np.ndarray
    nodes: str = "uniform"
    Ip: float | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        """在构造后把各字段规整为稳定运行时表示."""
        object.__setattr__(self, "route", _normalize_case_value("route", self.route))
        object.__setattr__(self, "coordinate", _normalize_case_value("coordinate", self.coordinate))
        object.__setattr__(self, "nodes", _normalize_case_value("nodes", self.nodes))
        object.__setattr__(self, "profile_coeffs", _normalize_case_value("profile_coeffs", self.profile_coeffs))
        object.__setattr__(self, "boundary", _normalize_case_value("boundary", self.boundary))
        object.__setattr__(self, "Ip", _normalize_case_value("Ip", self.Ip))
        object.__setattr__(self, "beta", _normalize_case_value("beta", self.beta))
        object.__setattr__(self, "heat_input", _normalize_case_value("heat_input", self.heat_input))
        object.__setattr__(self, "current_input", _normalize_case_value("current_input", self.current_input))
        if self.heat_input.shape != self.current_input.shape:
            raise ValueError(
                f"heat_input and current_input must share the same shape, "
                f"got {self.heat_input.shape} and {self.current_input.shape}"
            )
        _autoscale_legacy_mu0_inputs(self)

    def __setattr__(self, name: str, value) -> None:
        if name in (
            "profile_coeffs",
            "route",
            "boundary",
            "coordinate",
            "nodes",
            "Ip",
            "beta",
            "heat_input",
            "current_input",
        ):
            value = _normalize_case_value(name, value)
        object.__setattr__(self, name, value)

    def __rich__(self):
        tree = Tree("[bold blue]OperatorCase[/]")
        tree.add(f"route: {self.route}")
        tree.add(f"coordinate: {self.coordinate}")
        tree.add(f"nodes: {self.nodes}")
        tree.add(
            f"heat_input: shape={self.heat_input.shape}, "
            f"min={float(np.min(self.heat_input)):.3f}, max={float(np.max(self.heat_input)):.3f}"
        )
        tree.add(
            f"current_input: shape={self.current_input.shape}, "
            f"min={float(np.min(self.current_input)):.3f}, max={float(np.max(self.current_input)):.3f}"
        )
        if np.isfinite(self.Ip):
            tree.add(f"Ip(mu0-scaled): {self.Ip:.3e}")
        if np.isfinite(self.beta):
            tree.add(f"beta: {self.beta:.3e}")
        tree.add(self.boundary)
        return tree

    def __str__(self) -> str:
        console = Console(color_system=None, force_terminal=False, width=120, record=True, soft_wrap=False)
        with console.capture() as capture:
            console.print(self.__rich__())
        return capture.get().rstrip()

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> OperatorCase:
        """创建一个与当前 case 独立的副本."""
        return OperatorCase(
            route=self.route,
            profile_coeffs=_copy_coeffs(self.profile_coeffs),
            boundary=Boundary(
                a=self.a,
                R0=self.R0,
                Z0=self.Z0,
                B0=self.B0,
                ka=self.ka,
                c_offsets=self.c_offsets.copy(),
                s_offsets=self.s_offsets.copy(),
            ),
            heat_input=self.heat_input.copy(),
            current_input=self.current_input.copy(),
            coordinate=self.coordinate,
            nodes=self.nodes,
            Ip=self.Ip,
            beta=self.beta,
        )

    @property
    def a(self) -> float:
        return self.boundary.a

    @property
    def R0(self) -> float:
        return self.boundary.R0

    @property
    def Z0(self) -> float:
        return self.boundary.Z0

    @property
    def B0(self) -> float:
        return self.boundary.B0

    @property
    def ka(self) -> float:
        return self.boundary.ka

    @property
    def c_offsets(self) -> np.ndarray:
        return self.boundary.c_offsets

    @property
    def s_offsets(self) -> np.ndarray:
        return self.boundary.s_offsets


def _normalize_coeffs(
    profile_coeffs: dict[str, ProfileCoeffInput],
) -> dict[str, ProfileCoeff]:
    return {name: _normalize_profile_coeff(name, coeff) for name, coeff in profile_coeffs.items()}


def _normalize_profile_coeff(name: str, coeff: ProfileCoeffInput) -> ProfileCoeff:
    if coeff is None:
        return None
    if isinstance(coeff, bool):
        raise TypeError(f"{name} coeff length indicator must be an integer, got bool")
    if isinstance(coeff, Integral):
        length = int(coeff)
        if length <= 0:
            raise ValueError(f"{name} coeff length indicator must be positive, got {coeff}")
        return np.zeros(length, dtype=np.float64)
    if isinstance(coeff, (list, np.ndarray)):
        return _as_1d_array(coeff, name=f"{name} coeff").astype(np.float64, copy=True)
    raise TypeError(
        f"{name} coeff must be list[float], numpy.ndarray, positive int, or None; got {type(coeff).__name__}"
    )


def _copy_coeffs(profile_coeffs: dict[str, ProfileCoeffInput]) -> dict[str, ProfileCoeff]:
    return {name: _normalize_profile_coeff(name, coeff) for name, coeff in profile_coeffs.items()}


def _as_1d_array(value: np.ndarray | list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr


def _normalize_case_value(name: str, value):
    if name == "route":
        return str(value).upper()
    if name == "profile_coeffs":
        return _normalize_coeffs(value)
    if name == "boundary":
        if isinstance(value, Boundary):
            return value
        if isinstance(value, dict):
            return Boundary(**value)
        raise TypeError(f"boundary must be Boundary or dict, got {type(value).__name__}")
    if name == "coordinate":
        coord = str(value).lower()
        if coord not in ("rho", "psin"):
            raise ValueError(f"coordinate must be one of ('rho', 'psin'), got {value!r}")
        return coord
    if name == "nodes":
        nodes = str(value).lower()
        if nodes not in ("uniform", "grid"):
            raise ValueError(f"nodes must be one of ('uniform', 'grid'), got {value!r}")
        return nodes
    if name in ("Ip", "beta"):
        return np.nan if value is None else float(value)
    if name in ("heat_input", "current_input"):
        return _as_1d_array(value, name=name).copy()
    return value


def _autoscale_legacy_mu0_inputs(case: OperatorCase) -> None:
    warnings_needed: list[str] = []
    legacy_unscaled_abs_limit = 1.0e4
    mu0 = 4.0e-7 * np.pi

    max_abs = float(np.max(np.abs(case.heat_input))) if case.heat_input.size else 0.0
    if max_abs > legacy_unscaled_abs_limit:
        case.heat_input *= mu0
        warnings_needed.append("heat_input")

    if case.route in {"PI", "PJ1", "PJ2"}:
        max_abs = float(np.max(np.abs(case.current_input))) if case.current_input.size else 0.0
        if max_abs > legacy_unscaled_abs_limit:
            case.current_input *= mu0
            warnings_needed.append("current_input")

    if np.isfinite(case.Ip):
        ip_value = float(case.Ip)
        if abs(ip_value) > legacy_unscaled_abs_limit:
            object.__setattr__(case, "Ip", ip_value * mu0)
            warnings_needed.append("Ip")

    if warnings_needed:
        fields = ", ".join(warnings_needed)
        warnings.warn(
            f"Auto-scaled legacy inputs for {fields}; canonical OperatorCase inputs are mu0-scaled.",
            RuntimeWarning,
            stacklevel=3,
        )
