"""
operator 层导出面.
负责暴露 packed layout helper, packed block decode 入口, 以及 Operator 和 OperatorCase 两个稳定构造接口.
不负责 engine backend 选择, solver 驱动, 文档示例编排.
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
