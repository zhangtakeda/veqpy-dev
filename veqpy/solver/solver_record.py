"""
Module: solver.solver_record

Role:
- 负责持有一次求解对应的 history 快照.

Public API:
- SolverRecord

Notes:
- `SolverRecord` 只打包 case, config 和 result 快照.
- 不负责求解执行, packed codec, 或数值核更新.
"""

from dataclasses import dataclass

from rich.console import Console
from rich.tree import Tree

from veqpy.operator.operator_case import OperatorCase
from veqpy.solver.solver_config import SolverConfig
from veqpy.solver.solver_result import SolverResult


@dataclass(frozen=True, slots=True)
class SolverRecord:
    """描述一次求解完成后的不可变 history 快照."""

    case_snapshot: OperatorCase
    config_snapshot: SolverConfig
    result_snapshot: SolverResult

    def __rich__(self):
        tree = Tree("[bold blue]SolverRecord[/]")
        tree.add(self.case_snapshot)
        tree.add(self.config_snapshot)
        tree.add(self.result_snapshot)
        return tree

    def __str__(self) -> str:
        console = Console(color_system=None, force_terminal=False, width=120, record=True, soft_wrap=False)
        with console.capture() as capture:
            console.print(self.__rich__())
        return capture.get().rstrip()

    def __repr__(self) -> str:
        return str(self)
