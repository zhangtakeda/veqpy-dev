"""
solver 层导出面.
负责暴露 Solver, SolverConfig, SolverRecord 和 SolverResult 四个稳定接口.
不负责 packed layout 定义, engine backend 选择, benchmark 组织.
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
