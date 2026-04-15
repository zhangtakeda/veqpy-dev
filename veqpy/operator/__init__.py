"""
Module: operator.__init__

Role:
- 负责导出 operator 层的公开类型与包级入口.

Public API:
- Operator
- OperatorCase

Notes:
- 包级默认 `Operator` 现在是 fused 主实现.
- 不负责 engine backend 选择, solver 驱动, 或文档示例编排.
"""

from veqpy.operator.operator import Operator
from veqpy.operator.operator_case import OperatorCase

__all__ = [
    "Operator",
    "OperatorCase",
]
