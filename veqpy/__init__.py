"""
Module: veqpy

Role:
- Define package metadata and top-level package exports.

Public API:
- engine
- model
- operator
- solver

Notes:
- Package roots are the only modules that declare ``__all__``.
- Subpackages own their narrower public export surfaces.
"""

from __future__ import annotations

__all__ = ["engine", "model", "operator", "solver"]
__version__ = "0.3.1"
