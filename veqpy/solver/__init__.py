"""
Module: solver.__init__

Role:
- 负责导出 solver 层的公开类型与包级入口.

Public API:
- Solver
- SolverConfig
- SolverRecord
- SolverResult

Notes:
- 这里只做包级导出.
- 不负责 packed layout 定义, engine backend 选择, 或 benchmark 组织.
"""

from veqpy.solver.solver import Solver
from veqpy.solver.solver_config import SolverConfig
from veqpy.solver.solver_record import SolverRecord
from veqpy.solver.solver_result import SolverResult

__all__ = [
    "Solver",
    "SolverConfig",
    "SolverRecord",
    "SolverResult",
]
