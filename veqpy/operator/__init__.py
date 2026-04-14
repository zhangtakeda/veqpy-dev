"""
Module: operator.__init__

Role:
- 负责导出 operator 层的公开类型与包级入口.

Public API:
- Operator
- OperatorCase
- StaticLayout
- ResidualBindingLayout
- RuntimeLayout
- BackendState
- FieldRuntimeState
- ExecutionState
- SourceRuntimeState
- SourceConstState
- SourceWorkState
- SourceAuxState
- build_profile_layout
- build_profile_names
- build_profile_index
- decode_packed_blocks

Notes:
- 包级默认 `Operator` 现在是 fused 主实现.
- 不负责 engine backend 选择, solver 驱动, 或文档示例编排.
"""

from veqpy.operator.codec import decode_packed_blocks
from veqpy.operator.layout import build_profile_index, build_profile_layout, build_profile_names
from veqpy.operator.layouts import (
    BackendState,
    ExecutionState,
    FieldRuntimeState,
    ResidualBindingLayout,
    SourceAuxState,
    SourceConstState,
    RuntimeLayout,
    SourceRuntimeState,
    SourceWorkState,
    StaticLayout,
)
from veqpy.operator.operator import Operator
from veqpy.operator.operator_case import OperatorCase

__all__ = [
    "build_profile_layout",
    "build_profile_index",
    "build_profile_names",
    "decode_packed_blocks",
    "Operator",
    "OperatorCase",
    "StaticLayout",
    "ResidualBindingLayout",
    "RuntimeLayout",
    "BackendState",
    "FieldRuntimeState",
    "ExecutionState",
    "SourceRuntimeState",
    "SourceConstState",
    "SourceWorkState",
    "SourceAuxState",
]
