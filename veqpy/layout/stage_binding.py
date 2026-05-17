"""Deprecated compatibility re-exports for stage binding helpers.

Stage-specific binding now lives in ``veqpy.layout.source_binding`` and
``veqpy.layout.geometry_binding``.  Keep this module as a narrow import shim
only; do not add new binding logic here.
"""

from veqpy.layout.geometry_binding import build_geometry_stage_runner
from veqpy.layout.source_binding import build_bound_source_stage_runner

__all__ = ["build_bound_source_stage_runner", "build_geometry_stage_runner"]
