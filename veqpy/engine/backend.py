"""
Module: engine.backend

Role:
- 定义 per-Operator backend capability surface.
- 提供最小 backend 选择入口, 让 Operator 在构造时绑定后端.

Public API:
- BackendCapabilities
- resolve_backend

Notes:
- `numba` 是默认且用户可用的 backend.
- `jax` backend 仅供内部开发验证, 仍在开发中.
- 避免模块级 registry, 直到确有第三实现需要.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from veqpy.engine import numba_operator, numba_profile, numba_residual


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    name: str
    apply_f2_profile_fields: Callable
    bind_source_eval_runner: Callable
    bind_fused_residual_runner: Callable
    update_profiles_packed_bulk: Callable
    update_residual_compact: Callable
    run_residual_blocks_packed_precomputed: Callable


def resolve_backend(name: str) -> BackendCapabilities:
    backend_name = str(name).lower()
    if backend_name == "numba":
        return BackendCapabilities(
            name="numba",
            apply_f2_profile_fields=numba_operator.apply_f2_profile_fields,
            bind_source_eval_runner=numba_operator.bind_source_eval_runner,
            bind_fused_residual_runner=numba_operator.bind_fused_residual_runner,
            update_profiles_packed_bulk=numba_profile.update_profiles_packed_bulk,
            update_residual_compact=numba_residual.update_residual_compact,
            run_residual_blocks_packed_precomputed=numba_residual._run_residual_blocks_packed_precomputed,
        )
    if backend_name == "jax":
        try:
            from veqpy.engine import jax_operator
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "JAX backend is an experimental development path and is not intended for end users. "
                "Install a JAX extra only if you are working on backend development."
            ) from exc

        return BackendCapabilities(
            name="jax",
            apply_f2_profile_fields=numba_operator.apply_f2_profile_fields,
            bind_source_eval_runner=numba_operator.bind_source_eval_runner,
            bind_fused_residual_runner=jax_operator.bind_fused_residual_runner,
            update_profiles_packed_bulk=numba_profile.update_profiles_packed_bulk,
            update_residual_compact=numba_residual.update_residual_compact,
            run_residual_blocks_packed_precomputed=numba_residual._run_residual_blocks_packed_precomputed,
        )
    raise ValueError(f"Unsupported backend {name!r}")
