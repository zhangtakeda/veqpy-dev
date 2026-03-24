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
