"""
Module: operator.plans

Role:
- 定义 operator 层下发给 backend runner 的执行计划对象.
- 把 source 语义从散落参数收敛为显式 plan.

Public API:
- build_residual_plan
- ResidualPlan
- SourcePlan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

RUNNER_SINGLE_PASS = 0
RUNNER_PROFILE_OWNED_PSIN = 1
RUNNER_FIXED_POINT_PSIN = 2


@dataclass(frozen=True, slots=True)
class SourcePlan:
    """描述 source 语义与 runner 绑定所需的只读计划."""

    kernel: Callable
    coordinate: str
    nodes: str
    coordinate_code: int
    strategy: str
    parameterization: str
    parameterization_code: int
    n_src: int
    heat_input: np.ndarray
    current_input: np.ndarray
    Ip: float
    beta: float
    has_projection_policy: bool
    projection_domain: str
    use_projected_finalize: bool
    heat_projection_degree: int
    current_projection_degree: int
    projection_domain_code: int
    endpoint_policy_code: int
    allow_query_warmstart: bool

    @property
    def is_grid_nodes(self) -> bool:
        return self.nodes == "grid"

    @property
    def is_psin_coordinate(self) -> bool:
        return self.coordinate == "psin"


@dataclass(frozen=True, slots=True)
class ResidualPlan:
    """描述 residual runner 绑定层需要的高层计划."""

    source_plan: SourcePlan
    runner_code: int
    supports_fused_residual: bool
    requires_psin_profile_fields: bool = False

    @property
    def is_single_pass(self) -> bool:
        return self.runner_code == RUNNER_SINGLE_PASS

    @property
    def is_profile_owned_psin(self) -> bool:
        return self.runner_code == RUNNER_PROFILE_OWNED_PSIN

    @property
    def is_fixed_point_psin(self) -> bool:
        return self.runner_code == RUNNER_FIXED_POINT_PSIN


def build_residual_plan(source_plan: SourcePlan) -> ResidualPlan:
    """把 source plan 映射为 runner 绑定层可消费的 residual plan."""

    strategy = source_plan.strategy
    if strategy == "single_pass":
        return ResidualPlan(
            source_plan=source_plan,
            runner_code=RUNNER_SINGLE_PASS,
            supports_fused_residual=True,
        )
    if strategy == "profile_owned_psin":
        return ResidualPlan(
            source_plan=source_plan,
            runner_code=RUNNER_PROFILE_OWNED_PSIN,
            supports_fused_residual=True,
            requires_psin_profile_fields=True,
        )
    if strategy == "fixed_point_psin":
        return ResidualPlan(
            source_plan=source_plan,
            runner_code=RUNNER_FIXED_POINT_PSIN,
            supports_fused_residual=True,
        )
    return ResidualPlan(
        source_plan=source_plan,
        runner_code=-1,
        supports_fused_residual=False,
    )
