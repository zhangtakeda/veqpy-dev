"""
Module: operator.operator_case

Role:
- 负责把算例输入规范化为稳定的 case 配置对象.

Public API:
- OperatorCase
- SHAPE_PROFILE_OFFSET_FIELDS
- SHAPE_PROFILE_OFFSET_FIELD_NAMES

Notes:
- `OperatorCase` 只保存 case 输入.
- 不负责 layout 构造, residual 计算, 或 solver 策略管理.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.text import Text
from rich.tree import Tree

SHAPE_PROFILE_OFFSET_FIELDS: dict[str, str] = {
    "k": "ka",
    "c0": "c0a",
    "c1": "c1a",
    "s1": "s1a",
    "s2": "s2a",
}
SHAPE_PROFILE_OFFSET_FIELD_NAMES = tuple(SHAPE_PROFILE_OFFSET_FIELDS.values())


@dataclass(slots=True)
class OperatorCase:
    """描述一次 operator 求值所需的静态 case 输入."""

    coeffs_by_name: dict[str, list[float] | None]
    a: float
    R0: float
    Z0: float
    B0: float
    heat_input: np.ndarray
    current_input: np.ndarray
    ka: float = 1.0
    c0a: float = 0.0
    c1a: float = 0.0
    s1a: float = 0.0
    s2a: float = 0.0
    Ip: float | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        """在构造后把各字段规整为稳定运行时表示."""
        for name in _CASE_FIELD_NAMES:
            object.__setattr__(self, name, _normalize_case_value(name, getattr(self, name)))

    def __setattr__(self, name: str, value) -> None:
        if name in _CASE_FIELD_NAMES:
            value = _normalize_case_value(name, value)
        object.__setattr__(self, name, value)

    def copy(self) -> OperatorCase:
        """创建一个与当前 case 独立的副本."""
        return OperatorCase(
            coeffs_by_name=_copy_coeffs(self.coeffs_by_name),
            a=self.a,
            R0=self.R0,
            Z0=self.Z0,
            B0=self.B0,
            **{field_name: getattr(self, field_name) for field_name in SHAPE_PROFILE_OFFSET_FIELD_NAMES},
            heat_input=self.heat_input.copy(),
            current_input=self.current_input.copy(),
            Ip=self.Ip,
            beta=self.beta,
        )

    def __rich__(self):
        tree = Tree("[bold blue]OperatorCase[/]")
        tree.add(Text(f"a: {self.a:.3f} [m]"))
        tree.add(Text(f"R0: {self.R0:.3f} [m]"))
        tree.add(Text(f"Z0: {self.Z0:.3f} [m]"))
        tree.add(f"B0: {self.B0:.3f} [T]")
        for field_name in SHAPE_PROFILE_OFFSET_FIELD_NAMES:
            tree.add(f"{field_name}: {getattr(self, field_name):.3f}")
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


def _normalize_coeffs(
    coeffs_by_name: dict[str, list[float] | None],
) -> dict[str, list[float] | None]:
    coeffs: dict[str, list[float] | None] = {}
    for name, coeff in coeffs_by_name.items():
        if coeff is None:
            coeffs[name] = None
            continue
        if not isinstance(coeff, list):
            raise TypeError(f"{name} coeff must be list[float] or None, got {type(coeff).__name__}")
        coeffs[name] = _as_1d_coeff_list(coeff, name=f"{name} coeff")
    return coeffs


def _copy_coeffs(coeffs_by_name: dict[str, list[float] | None]) -> dict[str, list[float] | None]:
    copied: dict[str, list[float] | None] = {}
    for name, coeff in coeffs_by_name.items():
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
    if name == "coeffs_by_name":
        return _normalize_coeffs(value)
    if name in _FLOAT_FIELD_NAMES:
        return float(value)
    if name in _OPTIONAL_FLOAT_FIELD_NAMES:
        return np.nan if value is None else float(value)
    if name in _ARRAY_FIELD_NAMES:
        return _as_1d_array(value, name=name).copy()
    return value


_FLOAT_FIELD_NAMES = {"a", "R0", "Z0", "B0", *SHAPE_PROFILE_OFFSET_FIELD_NAMES}
_OPTIONAL_FLOAT_FIELD_NAMES = {"Ip", "beta"}
_ARRAY_FIELD_NAMES = {"heat_input", "current_input"}
_CASE_FIELD_NAMES = {
    "coeffs_by_name",
    *_FLOAT_FIELD_NAMES,
    *_OPTIONAL_FLOAT_FIELD_NAMES,
    *_ARRAY_FIELD_NAMES,
}
