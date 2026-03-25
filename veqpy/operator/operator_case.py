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
    c_offsets: np.ndarray | None = None
    s_offsets: np.ndarray | None = None
    c0a: float = 0.0
    c1a: float = 0.0
    s1a: float = 0.0
    s2a: float = 0.0
    Ip: float | None = None
    beta: float | None = None
    _boundary_sync_ready: bool = False

    def __post_init__(self) -> None:
        """在构造后把各字段规整为稳定运行时表示."""
        object.__setattr__(self, "_boundary_sync_ready", False)
        object.__setattr__(self, "coeffs_by_name", _normalize_case_value("coeffs_by_name", self.coeffs_by_name))
        for name in _ORDERED_FLOAT_FIELD_NAMES:
            object.__setattr__(self, name, _normalize_case_value(name, getattr(self, name)))
        for name in _ORDERED_OPTIONAL_FLOAT_FIELD_NAMES:
            object.__setattr__(self, name, _normalize_case_value(name, getattr(self, name)))
        for name in _ORDERED_ARRAY_FIELD_NAMES:
            object.__setattr__(self, name, _normalize_case_value(name, getattr(self, name)))
        _sync_boundary_offset_fields(self, source="post_init")
        object.__setattr__(self, "_boundary_sync_ready", True)

    def __setattr__(self, name: str, value) -> None:
        if name in _CASE_FIELD_NAMES:
            value = _normalize_case_value(name, value)
        object.__setattr__(self, name, value)
        if name in _BOUNDARY_OFFSET_FIELD_NAMES and getattr(self, "_boundary_sync_ready", False) and _boundary_fields_ready(self):
            _sync_boundary_offset_fields(self, source=name)

    def copy(self) -> OperatorCase:
        """创建一个与当前 case 独立的副本."""
        return OperatorCase(
            coeffs_by_name=_copy_coeffs(self.coeffs_by_name),
            a=self.a,
            R0=self.R0,
            Z0=self.Z0,
            B0=self.B0,
            c_offsets=self.c_offsets.copy(),
            s_offsets=self.s_offsets.copy(),
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
        tree.add(f"c_offsets: {np.array2string(self.c_offsets, precision=3, separator=', ')}")
        tree.add(f"s_offsets: {np.array2string(self.s_offsets, precision=3, separator=', ')}")
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
        if name in {"c_offsets", "s_offsets"}:
            return _normalize_offset_array(value, name=name)
        return _as_1d_array(value, name=name).copy()
    return value


def _normalize_offset_array(value, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = _as_1d_array(value, name=name).copy()
    if arr.size == 0:
        raise ValueError(f"{name} must have at least one entry")
    if name == "s_offsets":
        arr[0] = 0.0
    return arr


def _boundary_fields_ready(case: OperatorCase) -> bool:
    return all(hasattr(case, name) for name in _BOUNDARY_OFFSET_FIELD_NAMES)


def _sync_boundary_offset_fields(case: OperatorCase, *, source: str) -> None:
    if source == "post_init":
        _initialize_boundary_offsets(case)
    elif source in {"c_offsets", "s_offsets"}:
        _validate_legacy_offset_consistency(case)
    elif source in {"c0a", "c1a", "s1a", "s2a"}:
        _update_arrays_from_legacy_offsets(case)
    elif source == "ka":
        return
    _update_legacy_offsets_from_arrays(case)


def _initialize_boundary_offsets(case: OperatorCase) -> None:
    if case.c_offsets is None and case.s_offsets is None:
        object.__setattr__(case, "c_offsets", np.asarray([case.c0a, case.c1a], dtype=np.float64))
        object.__setattr__(case, "s_offsets", np.asarray([0.0, case.s1a, case.s2a], dtype=np.float64))
        return

    if case.c_offsets is None:
        object.__setattr__(case, "c_offsets", np.asarray([case.c0a, case.c1a], dtype=np.float64))
    if case.s_offsets is None:
        object.__setattr__(case, "s_offsets", np.asarray([0.0, case.s1a, case.s2a], dtype=np.float64))
    _validate_legacy_offset_consistency(case)


def _validate_legacy_offset_consistency(case: OperatorCase) -> None:
    tol = 1.0e-12
    if case.c_offsets is not None:
        c0, c1 = _offset_value(case.c_offsets, 0), _offset_value(case.c_offsets, 1)
        if (abs(case.c0a) > tol and abs(case.c0a - c0) > tol) or (abs(case.c1a) > tol and abs(case.c1a - c1) > tol):
            raise ValueError("Legacy c0a/c1a conflict with c_offsets")
    if case.s_offsets is not None:
        s1, s2 = _offset_value(case.s_offsets, 1), _offset_value(case.s_offsets, 2)
        if (abs(case.s1a) > tol and abs(case.s1a - s1) > tol) or (abs(case.s2a) > tol and abs(case.s2a - s2) > tol):
            raise ValueError("Legacy s1a/s2a conflict with s_offsets")


def _update_arrays_from_legacy_offsets(case: OperatorCase) -> None:
    c_offsets = np.zeros(max(2, 0 if case.c_offsets is None else case.c_offsets.size), dtype=np.float64)
    s_offsets = np.zeros(max(3, 0 if case.s_offsets is None else case.s_offsets.size), dtype=np.float64)
    if case.c_offsets is not None:
        c_offsets[: case.c_offsets.size] = case.c_offsets
    if case.s_offsets is not None:
        s_offsets[: case.s_offsets.size] = case.s_offsets
    c_offsets[0] = case.c0a
    c_offsets[1] = case.c1a
    s_offsets[0] = 0.0
    s_offsets[1] = case.s1a
    s_offsets[2] = case.s2a
    object.__setattr__(case, "c_offsets", c_offsets)
    object.__setattr__(case, "s_offsets", s_offsets)


def _update_legacy_offsets_from_arrays(case: OperatorCase) -> None:
    object.__setattr__(case, "c0a", _offset_value(case.c_offsets, 0))
    object.__setattr__(case, "c1a", _offset_value(case.c_offsets, 1))
    object.__setattr__(case, "s1a", _offset_value(case.s_offsets, 1))
    object.__setattr__(case, "s2a", _offset_value(case.s_offsets, 2))


def _offset_value(arr: np.ndarray, index: int) -> float:
    if arr is None or arr.size <= index:
        return 0.0
    return float(arr[index])


_FLOAT_FIELD_NAMES = {"a", "R0", "Z0", "B0", "ka", *SHAPE_PROFILE_OFFSET_FIELD_NAMES}
_OPTIONAL_FLOAT_FIELD_NAMES = {"Ip", "beta"}
_ARRAY_FIELD_NAMES = {"heat_input", "current_input", "c_offsets", "s_offsets"}
_BOUNDARY_OFFSET_FIELD_NAMES = {"ka", "c_offsets", "s_offsets", *SHAPE_PROFILE_OFFSET_FIELD_NAMES}
_ORDERED_FLOAT_FIELD_NAMES = ("a", "R0", "Z0", "B0", "ka", *SHAPE_PROFILE_OFFSET_FIELD_NAMES)
_ORDERED_OPTIONAL_FLOAT_FIELD_NAMES = ("Ip", "beta")
_ORDERED_ARRAY_FIELD_NAMES = ("heat_input", "current_input", "c_offsets", "s_offsets")
_CASE_FIELD_NAMES = {
    "coeffs_by_name",
    *_FLOAT_FIELD_NAMES,
    *_OPTIONAL_FLOAT_FIELD_NAMES,
    *_ARRAY_FIELD_NAMES,
    "_boundary_sync_ready",
}
