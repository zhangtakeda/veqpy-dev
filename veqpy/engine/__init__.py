"""
engine 层 backend 导出面.
负责按 VEQPY_BACKEND 选择数组导向数值核并导出 profile, geometry, residual, source helper 的稳定接口.
不负责算子定义, packed layout/codec, solver facade.
"""

import os

BACKEND = os.environ.get("VEQPY_BACKEND", "numba")
if BACKEND not in ("numpy", "numba"):
    raise ValueError(f"Unsupported VEQPY_BACKEND={BACKEND!r}. Supported backends: numpy, numba")

if BACKEND == "numpy":
    from veqpy.engine.numpy_geometry import update_geometry
    from veqpy.engine.numpy_profile import update_profile, update_profile_packed, update_profiles_packed_bulk
    from veqpy.engine.numpy_residual import (
        update_residual,
        bind_residual_block,
        bind_residual_runner,
        assemble_h_residual_block,
        assemble_v_residual_block,
        assemble_k_residual_block,
        assemble_c0_residual_block,
        assemble_c1_residual_block,
        assemble_s1_residual_block,
        assemble_s2_residual_block,
        assemble_psin_residual_block,
        assemble_F_residual_block,
    )
    from veqpy.engine.numpy_source import (
        RHO_AXIS,
        THETA_AXIS,
        DERIVATIVE_NAMES,
        PSI_DERIVATIVE,
        RHO_DERIVATIVE,
        bind_runner,
        validate_operator,
        full_differentiation,
        theta_reduction,
        quadrature,
        full_integration,
        corrected_integration,
    )
elif BACKEND == "numba":
    from veqpy.engine.numba_geometry import update_geometry
    from veqpy.engine.numba_profile import update_profile, update_profile_packed, update_profiles_packed_bulk
    from veqpy.engine.numba_residual import (
        update_residual,
        bind_residual_block,
        bind_residual_runner,
        assemble_h_residual_block,
        assemble_v_residual_block,
        assemble_k_residual_block,
        assemble_c0_residual_block,
        assemble_c1_residual_block,
        assemble_s1_residual_block,
        assemble_s2_residual_block,
        assemble_psin_residual_block,
        assemble_F_residual_block,
    )
    from veqpy.engine.numba_source import (
        RHO_AXIS,
        THETA_AXIS,
        DERIVATIVE_NAMES,
        PSI_DERIVATIVE,
        RHO_DERIVATIVE,
        bind_runner,
        validate_operator,
        full_differentiation,
        theta_reduction,
        quadrature,
        full_integration,
        corrected_integration,
    )


__all__ = [
    "update_profile",
    "update_profile_packed",
    "update_profiles_packed_bulk",
    "update_geometry",
    "update_residual",
    "bind_residual_block",
    "bind_residual_runner",
    "assemble_h_residual_block",
    "assemble_v_residual_block",
    "assemble_k_residual_block",
    "assemble_c0_residual_block",
    "assemble_c1_residual_block",
    "assemble_s1_residual_block",
    "assemble_s2_residual_block",
    "assemble_psin_residual_block",
    "assemble_F_residual_block",
    "RHO_AXIS",
    "THETA_AXIS",
    "DERIVATIVE_NAMES",
    "PSI_DERIVATIVE",
    "RHO_DERIVATIVE",
    "bind_runner",
    "validate_operator",
    "full_differentiation",
    "theta_reduction",
    "quadrature",
    "full_integration",
    "corrected_integration",
]
