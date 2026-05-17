"""
Compatibility re-exports for stage binding helpers.

Executable stage binding now lives under ``veqpy.layout.stage_binding``.
This module remains as a narrow import shim for legacy callers.
"""

from veqpy.layout.stage_binding import (  # noqa: F401
    build_bound_source_stage_runner,
    build_geometry_stage_runner,
)

__all__ = ["build_bound_source_stage_runner", "build_geometry_stage_runner"]
