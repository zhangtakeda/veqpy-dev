from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.text import Text
from rich.tree import Tree


@dataclass(frozen=True, slots=True)
class SolverResult:
    x0: np.ndarray
    x: np.ndarray
    success: bool
    message: str
    residual_norm_initial: float
    residual_norm_final: float
    nfev: int
    njev: int
    nit: int
    elapsed: float

    def __post_init__(self) -> None:
        """Detach packed state arrays from caller-owned inputs."""

        object.__setattr__(self, "x0", _as_1d_array(self.x0, name="x0"))
        object.__setattr__(self, "x", _as_1d_array(self.x, name="x"))

    def __rich__(self):
        tree = Tree("[bold blue]SolverResult[/]")
        tree.add(f"success: {self.success}")
        tree.add(f"message: {self.message}")
        tree.add(f"residual_norm: {self.residual_norm_final:.6e}")
        tree.add(f"nfev: {self.nfev}")
        tree.add(f"njev: {self.njev}")
        tree.add(f"nit: {self.nit}")
        tree.add(Text(f"elapsed: {(self.elapsed / 1000):.3f} [ms]"))
        tree.add(f"x0: shape={self.x0.shape}, min={float(np.min(self.x0)):.3f}, max={float(np.max(self.x0)):.3f}")
        tree.add(f"x: shape={self.x.shape}, min={float(np.min(self.x)):.3f}, max={float(np.max(self.x)):.3f}")
        return tree

    def __str__(self) -> str:
        console = Console(color_system=None, force_terminal=False, width=120, record=True, soft_wrap=False)
        with console.capture() as capture:
            console.print(self.__rich__())
        return capture.get().rstrip()

    def __repr__(self) -> str:
        return str(self)


def _as_1d_array(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr.copy()
