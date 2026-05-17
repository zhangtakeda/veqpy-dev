"""Residual/root-stage runtime memory ownership."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ResidualWorkspace:
    """Residual/root stage memory owner.

    ``root_fields`` shape: ``(5, Nr)`` with rows psin, psin_r, psin_rr,
    FFn_psin, and Pn_psin.

    ``surface_fields`` shape: ``(4, Nr, Nt)`` with rows G, G*psin_R,
    G*psin_Z, and G*psin_R*sin_tb.
    """

    root_fields: np.ndarray
    packed_residual: np.ndarray
    surface_fields: np.ndarray
    pack_scratch: np.ndarray
    collocation_sqrt_weights: np.ndarray
