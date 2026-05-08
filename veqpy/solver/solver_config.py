"""
Module: solver.solver_config

Role:
- 负责描述一次 nonlinear solve 的配置.
- 负责注册 solver method 名称到 SciPy optimize callable 的映射.

Public API:
- SolverConfig
- ROOT_METHODS
- LEAST_SQUARES_METHODS
- SUPPORTED_METHODS

Notes:
- `SolverConfig` 类只保存配置, 不执行求解.
- 不负责 history 存储, 或 residual 评估.
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Callable

from rich.console import Console
from rich.tree import Tree
from scipy.optimize import least_squares, root

OptimizeMethod = Callable[..., Any]


def _root_method(method: str) -> OptimizeMethod:
    def run(fun, x0, **kwargs):
        return root(fun, x0, method=method, **kwargs)

    return run


def _least_squares_method(method: str) -> OptimizeMethod:
    def run(fun, x0, **kwargs):
        return least_squares(fun, x0, method=method, **kwargs)

    return run


ROOT_METHODS: dict[str, OptimizeMethod] = {
    "hybr": _root_method("hybr"),
}

LEAST_SQUARES_METHODS: dict[str, OptimizeMethod] = {
    "lm": _least_squares_method("lm"),
    "trf": _least_squares_method("trf"),
}

SUPPORTED_METHODS: dict[str, OptimizeMethod] = {
    **ROOT_METHODS,
    **LEAST_SQUARES_METHODS,
}

DEFAULT_VARIATIONAL_METHOD = "hybr"
DEFAULT_COLLOCATION_METHOD = "trf"
DEFAULT_VARIATIONAL_FALLBACK_METHODS = ("lm",)
DEFAULT_COLLOCATION_FALLBACK_METHODS = ("lm",)

RESIDUAL_FORMS = (
    "variational",
    "collocation",
)


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """描述 Solver 的默认配置与单次求解覆盖项."""

    method: str | None = None
    residual_form: str = "variational"
    max_residual: float = 1e-6
    max_evaluations: int = 1000
    enable_warmstart: bool = True
    enable_fallback: bool = True
    fallback_methods: tuple[str, ...] | list[str] | None = field(default=None)
    enable_verbose: bool = False
    enable_history: bool = True

    def __post_init__(self) -> None:
        """校验方法名与 fallback 相关参数是否合法."""

        residual_form = str(self.residual_form)
        method = _default_method_for_residual_form(residual_form) if self.method is None else str(self.method)
        fallback_methods = (
            _default_fallback_methods(residual_form) if self.fallback_methods is None else self.fallback_methods
        )
        fallback_methods = tuple(str(method_name) for method_name in fallback_methods)
        deduped_fallback_methods: list[str] = []
        seen: set[str] = set()
        for method_name in fallback_methods:
            if method_name in seen:
                continue
            seen.add(method_name)
            deduped_fallback_methods.append(method_name)

        if method not in SUPPORTED_METHODS:
            supported = ", ".join(SUPPORTED_METHODS)
            raise ValueError(
                f"Unsupported solver method {method!r}. "
                f"Supported methods are: {supported}. "
                f"Use root method 'hybr', "
                f"or use 'lm'/'trf' to call scipy.optimize.least_squares directly."
            )
        if residual_form not in RESIDUAL_FORMS:
            supported = ", ".join(RESIDUAL_FORMS)
            raise ValueError(f"Unsupported residual form {residual_form!r}. Supported residual forms are: {supported}.")
        unsupported_fallbacks = [
            method_name for method_name in deduped_fallback_methods if method_name not in SUPPORTED_METHODS
        ]
        if unsupported_fallbacks:
            supported = ", ".join(SUPPORTED_METHODS)
            unsupported = ", ".join(repr(method_name) for method_name in unsupported_fallbacks)
            raise ValueError(
                f"Unsupported fallback solver method(s): {unsupported}. Supported methods are: {supported}."
            )
        max_residual = float(self.max_residual)
        max_evaluations = int(self.max_evaluations)
        if not isfinite(max_residual) or max_residual <= 0.0:
            raise ValueError(f"SolverConfig.max_residual must be a positive finite float, got {self.max_residual!r}.")
        if max_evaluations < 0:
            raise ValueError(f"SolverConfig.max_evaluations must be non-negative; got {self.max_evaluations!r}.")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "residual_form", residual_form)
        object.__setattr__(self, "max_residual", max_residual)
        object.__setattr__(self, "max_evaluations", max_evaluations)
        object.__setattr__(self, "enable_fallback", bool(self.enable_fallback))
        object.__setattr__(self, "fallback_methods", tuple(deduped_fallback_methods))

    def __rich__(self):
        tree = Tree("[bold blue]SolverConfig[/]")
        tree.add(f"method: {self.method}")
        tree.add(f"residual_form: {self.residual_form}")
        tree.add(f"max_residual: {self.max_residual:.6g}")
        tree.add(f"max_evaluations: {self.max_evaluations}")
        tree.add(f"enable_warmstart: {self.enable_warmstart}")
        tree.add(f"enable_fallback: {self.enable_fallback}")
        if self.enable_fallback:
            tree.add(f"fallback_methods: {list(self.fallback_methods)}")
        tree.add(f"enable_verbose: {self.enable_verbose}")
        tree.add(f"enable_history: {self.enable_history}")
        return tree

    def __str__(self) -> str:
        console = Console(color_system=None, force_terminal=False, width=120, record=True, soft_wrap=False)
        with console.capture() as capture:
            console.print(self.__rich__())
        return capture.get().rstrip()

    def __repr__(self) -> str:
        return str(self)


def _default_method_for_residual_form(residual_form: str) -> str:
    if residual_form == "collocation":
        return DEFAULT_COLLOCATION_METHOD
    return DEFAULT_VARIATIONAL_METHOD


def _default_fallback_methods(residual_form: str) -> tuple[str, ...]:
    if residual_form == "collocation":
        return DEFAULT_COLLOCATION_FALLBACK_METHODS
    return DEFAULT_VARIATIONAL_FALLBACK_METHODS
