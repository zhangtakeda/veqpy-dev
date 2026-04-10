"""
Module: residual_blocks

Role:
- 定义 residual block 的共享命名规则与轻量级解码辅助函数.
- 为 operator layout 与 backend residual binder 提供共同的 block metadata 语义.

Public API:
- BLOCK_CODE_BY_NAME
- build_residual_block_metadata
- decode_residual_block_code
- decode_residual_block_kind
"""

from __future__ import annotations

import numpy as np

BLOCK_CODE_BY_NAME = {
    "h": 0,
    "v": 1,
    "k": 2,
    "c0": 3,
    "c_family": 4,
    "s_family": 5,
    "psin": 6,
    "F": 7,
}
F2_BLOCK_CODE = 8


def decode_residual_block_code(name: str) -> tuple[int, int]:
    """把 residual block 名字映射到 numba residual binder 使用的紧凑编码."""

    if name.startswith("c") and name[1:].isdigit():
        order = int(name[1:])
        if order == 0:
            return (BLOCK_CODE_BY_NAME["c0"], 0)
        return (BLOCK_CODE_BY_NAME["c_family"], order)
    if name.startswith("s") and name[1:].isdigit():
        order = int(name[1:])
        if order == 0:
            raise KeyError("s0 is not a valid residual block")
        return (BLOCK_CODE_BY_NAME["s_family"], order)
    try:
        return (BLOCK_CODE_BY_NAME[name], 0)
    except KeyError as exc:
        supported = ", ".join(BLOCK_CODE_BY_NAME)
        raise KeyError(f"Unknown residual block {name!r}. Supported blocks: {supported}") from exc


def decode_residual_block_kind(name: str) -> tuple[str, int, str | None]:
    """把 residual block 名字映射到 backend-agnostic 的 block kind."""

    if name.startswith("c") and name[1:].isdigit():
        order = int(name[1:])
        if order == 0:
            return ("fixed", 0, "c0")
        return ("c_family", order, None)
    if name.startswith("s") and name[1:].isdigit():
        order = int(name[1:])
        if order == 0:
            raise KeyError("s0 is not a valid residual block")
        return ("s_family", order, None)
    return ("fixed", 0, name)


def build_residual_block_metadata(profile_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """批量构建 residual block 编码与 family 阶数."""

    block_codes = np.empty(len(profile_names), dtype=np.int64)
    block_orders = np.zeros(len(profile_names), dtype=np.int64)
    for i, name in enumerate(profile_names):
        block_codes[i], block_orders[i] = decode_residual_block_code(name)
    return block_codes, block_orders
