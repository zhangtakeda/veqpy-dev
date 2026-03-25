"""
Module: operator.__init__

Role:
- 负责导出 operator 层的公开类型与包级入口.

Public API:
- Operator
- OperatorCase
- build_profile_layout
- decode_packed_blocks
- PROFILE_INDEX
- PROFILE_NAMES

Notes:
- 这里只做包级导出.
- 不负责 engine backend 选择, solver 驱动, 或文档示例编排.
"""

from veqpy.operator.codec import decode_packed_blocks
from veqpy.operator.layout import PROFILE_INDEX, PROFILE_NAMES, build_profile_layout
from veqpy.operator.operator import Operator
from veqpy.operator.operator_case import OperatorCase

__all__ = [
    "build_profile_layout",
    "decode_packed_blocks",
    "Operator",
    "OperatorCase",
    "PROFILE_INDEX",
    "PROFILE_NAMES",
]
