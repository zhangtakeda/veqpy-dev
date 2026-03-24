"""Encode and decode packed operator vectors at the operator boundary."""

from typing import Protocol

import numpy as np

from veqpy.operator.layout import (
    PROFILE_COUNT,
    PROFILE_NAMES,
    coeff_array_from_list,
    packed_size,
    validate_packed_state,
)


class ResidualAssembleSlot(Protocol):
    coeff_row: np.ndarray
    coeff_indices: np.ndarray

    def assemble(self) -> None: ...


def encode_packed_state(
    coeffs_by_name: dict[str, list[float] | None],
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
) -> np.ndarray:
    x = np.empty(packed_size(coeff_index), dtype=np.float64)

    for p, name in enumerate(PROFILE_NAMES):
        L = int(profile_L[p])
        coeff = coeffs_by_name.get(name)

        if L < 0:
            continue
        if coeff is None:
            raise ValueError(f"{name} is active in layout but coeff is None")

        coeff_arr = coeff_array_from_list(name, coeff)
        if coeff_arr.size != L + 1:
            raise ValueError(f"{name} coeff shape mismatch: expected {(L + 1,)}, got {coeff_arr.shape}")

        for k in range(L + 1):
            x[coeff_index[p, k]] = coeff_arr[k]

    return x


def encode_packed_residual(
    residual_slots: tuple[ResidualAssembleSlot, ...],
    residual_size: int,
) -> np.ndarray:
    out = np.zeros(residual_size, dtype=np.float64)
    for slot in residual_slots:
        slot.assemble()
        out[slot.coeff_indices] = slot.coeff_row
    return out


def decode_packed_state_inplace(
    x: np.ndarray,
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
    coeff_matrix: np.ndarray,
) -> None:
    x = validate_packed_state(x, coeff_index)
    if coeff_matrix.shape != coeff_index.shape:
        raise ValueError(f"Expected coeff_matrix shape {coeff_index.shape}, got {coeff_matrix.shape}")

    coeff_matrix.fill(0.0)
    for p in range(PROFILE_COUNT):
        L = int(profile_L[p])
        if L < 0:
            continue
        for k in range(L + 1):
            coeff_matrix[p, k] = x[coeff_index[p, k]]


def decode_packed_blocks(
    x: np.ndarray,
    profile_L: np.ndarray,
    coeff_index: np.ndarray,
) -> tuple[np.ndarray | None, ...]:
    x = validate_packed_state(x, coeff_index)

    coeff_matrix = np.zeros_like(coeff_index, dtype=np.float64)
    decode_packed_state_inplace(x, profile_L, coeff_index, coeff_matrix)

    blocks: list[np.ndarray | None] = []
    for p in range(PROFILE_COUNT):
        L = int(profile_L[p])
        if L < 0:
            blocks.append(None)
        else:
            blocks.append(coeff_matrix[p, : L + 1].copy())
    return tuple(blocks)
