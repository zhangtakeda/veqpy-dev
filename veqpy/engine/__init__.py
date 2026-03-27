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
- build_source_remap_cache
- resolve_source_inputs
- validate_operator

Notes:
- 这里只负责 backend dispatch.
- operator layout 与 solver orchestration 保留在上层.
- source 输入解析正式入口是 `resolve_source_inputs`.
"""

import os

BACKEND = os.environ.get("VEQPY_BACKEND", "numba")
if BACKEND not in ("numpy", "numba"):
    raise ValueError(f"Unsupported VEQPY_BACKEND={BACKEND!r}. Supported backends: numpy, numba")

if BACKEND == "numpy":
    from veqpy.engine.numpy_geometry import update_geometry
    from veqpy.engine.numpy_profile import update_profile, update_profiles_packed_bulk
    from veqpy.engine.numpy_residual import (
        bind_residual_runner,
        update_residual,
    )
    from veqpy.engine.numpy_source import (
        COORDINATE_NAMES,
        PSIN_COORDINATE,
        RHO_AXIS,
        RHO_COORDINATE,
        THETA_AXIS,
        bind_source_runner,
        build_source_remap_cache,
        corrected_integration,
        full_differentiation,
        full_integration,
        quadrature,
        resolve_source_inputs,
        theta_reduction,
        validate_operator,
    )
elif BACKEND == "numba":
    from veqpy.engine.numba_geometry import update_geometry
    from veqpy.engine.numba_profile import update_profile, update_profiles_packed_bulk
    from veqpy.engine.numba_residual import (
        bind_residual_runner,
        update_residual,
    )
    from veqpy.engine.numba_source import (
        COORDINATE_NAMES,
        PSIN_COORDINATE,
        RHO_AXIS,
        RHO_COORDINATE,
        THETA_AXIS,
        bind_source_runner,
        build_source_remap_cache,
        corrected_integration,
        full_differentiation,
        full_integration,
        quadrature,
        resolve_source_inputs,
        theta_reduction,
        validate_operator,
    )


__all__ = [
    "update_profile",
    "update_profiles_packed_bulk",
    "update_geometry",
    "update_residual",
    "bind_residual_runner",
    "RHO_AXIS",
    "THETA_AXIS",
    "COORDINATE_NAMES",
    "PSIN_COORDINATE",
    "RHO_COORDINATE",
    "bind_source_runner",
    "build_source_remap_cache",
    "validate_operator",
    "resolve_source_inputs",
    "full_differentiation",
    "theta_reduction",
    "quadrature",
    "full_integration",
    "corrected_integration",
]
