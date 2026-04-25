"""
Module: engine.profile_regularization

Role:
- Centralize cold-path profile regularization rules.
- Keep configurable Fourier radial powers out of hot kernels.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

FOURIER_POWER_K_MAX: int | None = None


def normalize_fourier_power_K_max(value: int | None) -> int | None:
    """Normalize and validate a Fourier radial-power cap."""
    if value is None:
        return None
    K_max = int(value)
    if K_max < 1:
        raise ValueError(f"K_max must be None or an integer >= 1, got {value!r}")
    return K_max


def resolve_fourier_power(order: int, K_max: int | None = None) -> int:
    """Resolve the radial prefactor power for a Fourier shape profile order."""
    order = int(order)
    if order <= 0:
        return 0
    normalized_K_max = normalize_fourier_power_K_max(K_max)
    if normalized_K_max is None:
        return order
    return min(order, normalized_K_max)


def get_fourier_power_K_max() -> int | None:
    """Return the module-level experimental default K_max."""
    return FOURIER_POWER_K_MAX


def set_fourier_power_K_max(value: int | None) -> None:
    """Set the module-level experimental default K_max."""
    global FOURIER_POWER_K_MAX
    FOURIER_POWER_K_MAX = normalize_fourier_power_K_max(value)


@contextmanager
def fourier_power_K_max(value: int | None) -> Iterator[None]:
    """Temporarily override the module-level experimental default K_max."""
    old_value = FOURIER_POWER_K_MAX
    set_fourier_power_K_max(value)
    try:
        yield
    finally:
        set_fourier_power_K_max(old_value)
