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
SUPPORTED_INITIAL_POLICIES = frozenset(("zeros", "warm", "homothetic", "optimize"))
SUPPORTED_RESIDUAL_NORMALIZATIONS = frozenset(("balanced", "legacy", "none"))

_RESIDUAL_NORMALIZATION_ALIASES = {
    "balanced": "balanced",
    "new": "balanced",
    "industrial": "balanced",
    "on": "balanced",
    "true": "balanced",
    "yes": "balanced",
    "1": "balanced",
    "legacy": "legacy",
    "old": "legacy",
    "block-rms-asinh": "legacy",
    "none": "none",
    "off": "none",
    "disabled": "none",
    "false": "none",
    "no": "none",
    "0": "none",
}


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """描述 Solver 的默认配置与单次求解覆盖项."""

    method: str | None = None
    max_residual: float = 1e-6
    max_evaluations: int = 1000
    enable_warmstart: bool = False
    initial_policy: str | None = None
    initial_homothetic_lambda: float = 1.0
    enable_fallback: bool = True
    fallback_methods: tuple[str, ...] | list[str] | None = field(default=None)
    enable_verbose: bool = False
    enable_history: bool = True
    residual_normalization: str | None = "legacy"
    residual_normalization_floor: float = 1.0
    residual_normalization_max_ratio: float = 1.0e30
    residual_normalization_root_global_blocks: int = 16

    enable_collocation: bool = False
    collocation_method: str = DEFAULT_COLLOCATION_METHOD
    collocation_max_residual: float | None = None
    collocation_max_evaluations: int | None = None

    def __post_init__(self) -> None:
        """校验方法名与 fallback 相关参数是否合法."""

        method = DEFAULT_VARIATIONAL_METHOD if self.method is None else str(self.method)
        fallback_methods = (
            DEFAULT_VARIATIONAL_FALLBACK_METHODS
            if self.fallback_methods is None
            else self.fallback_methods
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
            raise ValueError(f"Unsupported solver method {method!r}; supported: {supported}.")
        collocation_method = str(self.collocation_method)
        if collocation_method not in LEAST_SQUARES_METHODS:
            supported = ", ".join(LEAST_SQUARES_METHODS)
            raise ValueError(
                f"Unsupported collocation_method {collocation_method!r}; supported: {supported}."
            )
        unsupported_fallbacks = [
            method_name
            for method_name in deduped_fallback_methods
            if method_name not in SUPPORTED_METHODS
        ]
        if unsupported_fallbacks:
            supported = ", ".join(SUPPORTED_METHODS)
            unsupported = ", ".join(repr(method_name) for method_name in unsupported_fallbacks)
            raise ValueError(
                f"Unsupported fallback solver method(s): {unsupported}. "
                f"Supported methods are: {supported}."
            )
        max_residual = float(self.max_residual)
        max_evaluations = int(self.max_evaluations)
        initial_policy = None if self.initial_policy is None else str(self.initial_policy).lower()
        if initial_policy == "zero":
            initial_policy = "zeros"
        if initial_policy == "warmstart":
            initial_policy = "warm"
        if initial_policy is not None and initial_policy not in SUPPORTED_INITIAL_POLICIES:
            supported = ", ".join(sorted(SUPPORTED_INITIAL_POLICIES))
            raise ValueError(
                f"Unsupported initial_policy {self.initial_policy!r}; supported: {supported}."
            )
        initial_homothetic_lambda = float(self.initial_homothetic_lambda)
        if not isfinite(initial_homothetic_lambda):
            raise ValueError(
                "SolverConfig.initial_homothetic_lambda must be finite; "
                f"got {self.initial_homothetic_lambda!r}."
            )
        if not isfinite(max_residual) or max_residual <= 0.0:
            raise ValueError(
                f"SolverConfig.max_residual must be a positive finite float, "
                f"got {self.max_residual!r}."
            )
        if max_evaluations < 0:
            raise ValueError(
                f"SolverConfig.max_evaluations must be non-negative; got {self.max_evaluations!r}."
            )
        collocation_max_residual = (
            None if self.collocation_max_residual is None else float(self.collocation_max_residual)
        )
        collocation_max_evaluations = (
            None
            if self.collocation_max_evaluations is None
            else int(self.collocation_max_evaluations)
        )
        if collocation_max_residual is not None and (
            not isfinite(collocation_max_residual) or collocation_max_residual <= 0.0
        ):
            raise ValueError(
                "collocation_max_residual must be positive finite; "
                f"got {self.collocation_max_residual!r}."
            )
        if collocation_max_evaluations is not None and collocation_max_evaluations < 0:
            raise ValueError(
                "collocation_max_evaluations must be non-negative; "
                f"got {self.collocation_max_evaluations!r}."
            )
        residual_normalization = _normalize_residual_normalization(self.residual_normalization)
        residual_normalization_floor = float(self.residual_normalization_floor)
        residual_normalization_max_ratio = float(self.residual_normalization_max_ratio)
        residual_normalization_root_global_blocks = int(
            self.residual_normalization_root_global_blocks
        )
        if not isfinite(residual_normalization_floor) or residual_normalization_floor <= 0.0:
            raise ValueError(
                "SolverConfig.residual_normalization_floor must be positive finite; "
                f"got {self.residual_normalization_floor!r}."
            )
        if not isfinite(residual_normalization_max_ratio) or residual_normalization_max_ratio < 1.0:
            raise ValueError(
                "SolverConfig.residual_normalization_max_ratio must be finite and >= 1; "
                f"got {self.residual_normalization_max_ratio!r}."
            )
        if residual_normalization_root_global_blocks < 0:
            raise ValueError(
                "SolverConfig.residual_normalization_root_global_blocks must be "
                "non-negative; "
                f"got {self.residual_normalization_root_global_blocks!r}."
            )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "enable_collocation", bool(self.enable_collocation))
        object.__setattr__(self, "collocation_method", collocation_method)
        object.__setattr__(self, "collocation_max_residual", collocation_max_residual)
        object.__setattr__(self, "collocation_max_evaluations", collocation_max_evaluations)
        object.__setattr__(self, "max_residual", max_residual)
        object.__setattr__(self, "max_evaluations", max_evaluations)
        object.__setattr__(self, "initial_policy", initial_policy)
        object.__setattr__(self, "initial_homothetic_lambda", initial_homothetic_lambda)
        object.__setattr__(self, "enable_fallback", bool(self.enable_fallback))
        object.__setattr__(self, "fallback_methods", tuple(deduped_fallback_methods))
        object.__setattr__(self, "residual_normalization", residual_normalization)
        object.__setattr__(self, "residual_normalization_floor", residual_normalization_floor)
        object.__setattr__(
            self, "residual_normalization_max_ratio", residual_normalization_max_ratio
        )
        object.__setattr__(
            self,
            "residual_normalization_root_global_blocks",
            residual_normalization_root_global_blocks,
        )

    def __rich__(self):
        tree = Tree("[bold blue]SolverConfig[/]")
        tree.add(f"method: {self.method}")
        tree.add(f"enable_collocation: {self.enable_collocation}")
        if self.enable_collocation:
            tree.add(f"collocation_method: {self.collocation_method}")
            if self.collocation_max_residual is not None:
                tree.add(f"collocation_max_residual: {self.collocation_max_residual:.6g}")
            if self.collocation_max_evaluations is not None:
                tree.add(f"collocation_max_evaluations: {self.collocation_max_evaluations}")
        tree.add(f"max_residual: {self.max_residual:.6g}")
        tree.add(f"max_evaluations: {self.max_evaluations}")
        tree.add(f"initial_policy: {self.initial_policy}")
        if self.initial_policy == "homothetic":
            tree.add(f"initial_homothetic_lambda: {self.initial_homothetic_lambda:.6g}")
        tree.add(f"enable_warmstart: {self.enable_warmstart}")
        tree.add(f"enable_fallback: {self.enable_fallback}")
        if self.enable_fallback:
            tree.add(f"fallback_methods: {list(self.fallback_methods)}")
        tree.add(f"enable_verbose: {self.enable_verbose}")
        tree.add(f"enable_history: {self.enable_history}")
        tree.add(f"residual_normalization: {self.residual_normalization}")
        if self.residual_normalization != "none":
            tree.add(f"residual_normalization_floor: {self.residual_normalization_floor:.6g}")
            tree.add(
                f"residual_normalization_max_ratio: {self.residual_normalization_max_ratio:.6g}"
            )
            tree.add(
                "residual_normalization_root_global_blocks: "
                f"{self.residual_normalization_root_global_blocks}"
            )
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


def _normalize_residual_normalization(value: str | None) -> str:
    if value is None:
        return "balanced"
    key = str(value).strip().lower().replace("_", "-")
    try:
        return _RESIDUAL_NORMALIZATION_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_RESIDUAL_NORMALIZATIONS))
        raise ValueError(
            f"Unsupported residual_normalization {value!r}; supported: {supported}."
        ) from exc
