"""
solver 层 history 快照对象.
负责把一次求解对应的 case, config 和 result 打包为不可变记录, 便于追踪和重建.
不负责求解执行, packed codec, 数值核更新.
"""

from dataclasses import dataclass

from rich.console import Console
from rich.tree import Tree

from veqpy.operator.operator_case import OperatorCase
from veqpy.solver.solver_config import SolverConfig
from veqpy.solver.solver_result import SolverResult


@dataclass(frozen=True, slots=True)
class SolverRecord:
    """
    描述一次求解完成后的不可变 history 快照.

    Args:
        case_snapshot: 求解时使用的 OperatorCase 副本.
        config_snapshot: 本次求解实际生效的 SolverConfig.
        result_snapshot: 本次求解输出的 SolverResult.
    """

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
