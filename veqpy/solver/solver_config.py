"""
solver 层配置对象.
负责描述一次 nonlinear solve 的方法选择, 容差, 迭代上限和同伦开关.
不负责实际求解执行, history 存储, residual 评估.
"""

from dataclasses import dataclass

from rich.console import Console
from rich.tree import Tree

SUPPORTED_ROOT_METHODS = (
    "hybr",
    "krylov",
    "root-lm",
    "broyden1",
    "broyden2",
)

SUPPORTED_LEAST_SQUARES_METHODS = (
    "trf",
    "dogbox",
    "lm",
)

SUPPORTED_SOLVER_METHODS = SUPPORTED_ROOT_METHODS + SUPPORTED_LEAST_SQUARES_METHODS


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """
    描述 Solver 的默认配置与单次求解覆盖项.

    Args:
        method: 求解方法名. 支持 root 系列方法与 least_squares 系列方法.
        rtol, atol: 相对与绝对容差控制参数.
        root_maxiter, root_maxfev: 迭代与函数评估上限.
        enable_warmstart: 是否默认沿用上次求解后的 x0.
        enable_homotopy: 是否启用按阶次逐步放开的 homotopy 求解.
        enable_verbose: 是否打印 solve record.
        enable_history: 是否记录 solve history.
        homotopy_truncation_tol: 同伦截断判定的小系数阈值.
        homotopy_truncation_patience: 连续多少阶都足够小后冻结该 profile.
    """

    method: str = "hybr"
    rtol: float = 1e-6
    atol: float = 1e-6
    root_maxiter: int = 100
    root_maxfev: int = 1000
    enable_warmstart: bool = True
    enable_homotopy: bool = False
    enable_verbose: bool = False
    enable_history: bool = True
    homotopy_truncation_tol: float = 1e-3
    homotopy_truncation_patience: int = 1

    def __post_init__(self) -> None:
        """校验方法名与 homotopy 相关参数是否合法."""

        method = str(self.method)

        if method not in SUPPORTED_SOLVER_METHODS:
            supported = ", ".join(SUPPORTED_SOLVER_METHODS)
            raise ValueError(
                f"Unsupported solver method {method!r}. "
                f"Supported methods are: {supported}. "
                f"Use a root method such as 'hybr', 'krylov', or 'root-lm', "
                f"or use 'lm'/'trf'/'dogbox' to call scipy.optimize.least_squares directly."
            )
        if float(self.homotopy_truncation_tol) < 0.0:
            raise ValueError("homotopy_truncation_tol must be >= 0")
        if int(self.homotopy_truncation_patience) < 1:
            raise ValueError("homotopy_truncation_patience must be >= 1")

    def __rich__(self):
        tree = Tree("[bold blue]SolverConfig[/]")
        tree.add(f"method: {self.method}")
        tree.add(f"rtol: {self.rtol:.6g}")
        tree.add(f"atol: {self.atol:.6g}")
        tree.add(f"root_maxiter: {self.root_maxiter}")
        tree.add(f"root_maxfev: {self.root_maxfev}")
        tree.add(f"enable_warmstart: {self.enable_warmstart}")
        tree.add(f"enable_homotopy: {self.enable_homotopy}")
        tree.add(f"enable_verbose: {self.enable_verbose}")
        tree.add(f"enable_history: {self.enable_history}")
        tree.add(f"homotopy_truncation_tol: {self.homotopy_truncation_tol:.6g}")
        tree.add(f"homotopy_truncation_patience: {self.homotopy_truncation_patience}")
        return tree

    def __str__(self) -> str:
        console = Console(color_system=None, force_terminal=False, width=120, record=True, soft_wrap=False)
        with console.capture() as capture:
            console.print(self.__rich__())
        return capture.get().rstrip()

    def __repr__(self) -> str:
        return str(self)
