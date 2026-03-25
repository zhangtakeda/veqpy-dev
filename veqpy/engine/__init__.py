"""
Module: engine.__init__

Role:
- 负责选择当前数组后端.
- 负责导出 engine 层稳定入口.

Public API:
- update_profile
- update_profiles_packed_bulk
- update_geometry
- update_residual
- bind_source_runner
- bind_residual_runner
- validate_operator

Notes:
- 这里只负责 backend dispatch.
- operator layout 与 solver orchestration 保留在上层.
"""

import os

BACKEND = os.environ.get("VEQPY_BACKEND", "numba")
if BACKEND not in ("numpy", "numba"):
    raise ValueError(f"Unsupported VEQPY_BACKEND={BACKEND!r}. Supported backends: numpy, numba")

if BACKEND == "numpy":
    from veqpy.engine.numpy_geometry import update_geometry
    from veqpy.engine.numpy_profile import update_profile, update_profiles_packed_bulk
    from veqpy.engine.numpy_residual import (
        update_residual,
        bind_residual_runner,
    )
    from veqpy.engine.numpy_source import (
        RHO_AXIS,
        THETA_AXIS,
        DERIVATIVE_NAMES,
        PSI_DERIVATIVE,
        RHO_DERIVATIVE,
        bind_source_runner,
        validate_operator,
        full_differentiation,
        theta_reduction,
        quadrature,
        full_integration,
        corrected_integration,
    )
elif BACKEND == "numba":
    from veqpy.engine.numba_geometry import update_geometry
    from veqpy.engine.numba_profile import update_profile, update_profiles_packed_bulk
    from veqpy.engine.numba_residual import (
        update_residual,
        bind_residual_runner,
    )
    from veqpy.engine.numba_source import (
        RHO_AXIS,
        THETA_AXIS,
        DERIVATIVE_NAMES,
        PSI_DERIVATIVE,
        RHO_DERIVATIVE,
        bind_source_runner,
        validate_operator,
        full_differentiation,
        theta_reduction,
        quadrature,
        full_integration,
        corrected_integration,
    )


__all__ = [
    "update_profile",
    "update_profiles_packed_bulk",
    "update_geometry",
    "update_residual",
    "bind_residual_runner",
    "RHO_AXIS",
    "THETA_AXIS",
    "DERIVATIVE_NAMES",
    "PSI_DERIVATIVE",
    "RHO_DERIVATIVE",
    "bind_source_runner",
    "validate_operator",
    "full_differentiation",
    "theta_reduction",
    "quadrature",
    "full_integration",
    "corrected_integration",
]
