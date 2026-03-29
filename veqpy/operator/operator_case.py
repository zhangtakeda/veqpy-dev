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

from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.tree import Tree

from veqpy.model import Boundary


@dataclass(slots=True)
class OperatorCase:
    """描述一次 operator 求值所需的静态 case 输入."""

    name: str
    coordinate: str
    nodes: str
    profile_coeffs: dict[str, list[float] | None]
    boundary: Boundary
    heat_input: np.ndarray
    current_input: np.ndarray
    Ip: float | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        """在构造后把各字段规整为稳定运行时表示."""
        object.__setattr__(self, "name", _normalize_case_value("name", self.name))
        object.__setattr__(self, "coordinate", _normalize_case_value("coordinate", self.coordinate))
        object.__setattr__(self, "nodes", _normalize_case_value("nodes", self.nodes))
        object.__setattr__(self, "profile_coeffs", _normalize_case_value("profile_coeffs", self.profile_coeffs))
        object.__setattr__(self, "boundary", _normalize_case_value("boundary", self.boundary))
        for name in _ORDERED_OPTIONAL_FLOAT_FIELD_NAMES:
            object.__setattr__(self, name, _normalize_case_value(name, getattr(self, name)))
        for name in _ORDERED_ARRAY_FIELD_NAMES:
            object.__setattr__(self, name, _normalize_case_value(name, getattr(self, name)))
        if self.heat_input.shape != self.current_input.shape:
            raise ValueError(
                f"heat_input and current_input must share the same shape, "
                f"got {self.heat_input.shape} and {self.current_input.shape}"
            )

    def __setattr__(self, name: str, value) -> None:
        if name in _CASE_FIELD_NAMES:
            value = _normalize_case_value(name, value)
        object.__setattr__(self, name, value)

    def __rich__(self):
        tree = Tree("[bold blue]OperatorCase[/]")
        tree.add(f"name: {self.name}")
        tree.add(self.boundary)
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
            tree.add(f"Ip: {self.Ip:.3e} [A]")
        if np.isfinite(self.beta):
            tree.add(f"beta: {self.beta:.3e}")
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
            name=self.name,
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
    profile_coeffs: dict[str, list[float] | None],
) -> dict[str, list[float] | None]:
    coeffs: dict[str, list[float] | None] = {}
    for name, coeff in profile_coeffs.items():
        if coeff is None:
            coeffs[name] = None
            continue
        if not isinstance(coeff, list):
            raise TypeError(f"{name} coeff must be list[float] or None, got {type(coeff).__name__}")
        coeffs[name] = _as_1d_coeff_list(coeff, name=f"{name} coeff")
    return coeffs


def _copy_coeffs(profile_coeffs: dict[str, list[float] | None]) -> dict[str, list[float] | None]:
    copied: dict[str, list[float] | None] = {}
    for name, coeff in profile_coeffs.items():
        copied[name] = None if coeff is None else list(coeff)
    return copied


def _as_1d_array(value: np.ndarray | list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr


def _as_1d_coeff_list(value: list[float], *, name: str) -> list[float]:
    arr = _as_1d_array(value, name=name)
    return arr.astype(float, copy=False).tolist()


def _normalize_case_value(name: str, value):
    if name == "name":
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
        if coord not in _COORDINATE_FIELD_VALUES:
            raise ValueError(f"coordinate must be one of {_COORDINATE_FIELD_VALUES}, got {value!r}")
        return coord
    if name == "nodes":
        nodes = str(value).lower()
        if nodes not in _NODE_FIELD_VALUES:
            raise ValueError(f"nodes must be one of {_NODE_FIELD_VALUES}, got {value!r}")
        return nodes
    if name in _OPTIONAL_FLOAT_FIELD_NAMES:
        return np.nan if value is None else float(value)
    if name in _ARRAY_FIELD_NAMES:
        return _as_1d_array(value, name=name).copy()
    return value


_OPTIONAL_FLOAT_FIELD_NAMES = {"Ip", "beta"}
_ARRAY_FIELD_NAMES = {"heat_input", "current_input"}
_COORDINATE_FIELD_VALUES = ("rho", "psin")
_NODE_FIELD_VALUES = ("uniform", "grid")
_ORDERED_OPTIONAL_FLOAT_FIELD_NAMES = ("Ip", "beta")
_ORDERED_ARRAY_FIELD_NAMES = ("heat_input", "current_input")
_CASE_FIELD_NAMES = {
    "profile_coeffs",
    "name",
    "boundary",
    "coordinate",
    "nodes",
    *_OPTIONAL_FLOAT_FIELD_NAMES,
    *_ARRAY_FIELD_NAMES,
}
