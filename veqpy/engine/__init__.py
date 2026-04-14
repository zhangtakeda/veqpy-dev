"""
Module: engine.__init__

Role:
- 作为 engine 的唯一对外导入入口.

Notes:
- 求解后端已统一为 numba.
- operator layout 与 solver orchestration 保留在上层.
- 外层模块应统一 `from veqpy.engine import ...`, 不直接导入内部文件.
"""

from veqpy.engine import numba_operator as operator_ops
from veqpy.engine import numba_profile as profile_ops
from veqpy.engine import numba_residual as residual_ops
from veqpy.engine import numba_source as source_ops
from veqpy.engine import backend_abi
from veqpy.engine import orchestration
from veqpy.engine.numba_source import (
    COORDINATE_NAMES,
    PSIN_COORDINATE,
    RHO_AXIS,
    RHO_COORDINATE,
    THETA_AXIS,
    corrected_even_derivative,
    corrected_integration,
    corrected_linear_derivative,
    full_differentiation,
    full_integration,
    quadrature,
    theta_reduction,
    validate_route,
)

__all__ = [
    "operator_ops",
    "profile_ops",
    "residual_ops",
    "source_ops",
    "backend_abi",
    "orchestration",
    "RHO_AXIS",
    "THETA_AXIS",
    "COORDINATE_NAMES",
    "PSIN_COORDINATE",
    "RHO_COORDINATE",
    "validate_route",
    "full_differentiation",
    "theta_reduction",
    "quadrature",
    "full_integration",
    "corrected_integration",
    "corrected_linear_derivative",
    "corrected_even_derivative",
]
