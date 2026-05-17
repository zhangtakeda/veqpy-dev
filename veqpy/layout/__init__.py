"""Executable layout objects for operator stages."""

from veqpy.layout.runtime import (
    GeometryLayout,
    OperatorLayout,
    ProfileLayout,
    ResidualLayout,
    SourceLayout,
)

__all__ = [
    "build_operator_layout",
    "GeometryLayout",
    "OperatorLayout",
    "ProfileLayout",
    "ResidualLayout",
    "SourceLayout",
]


def __getattr__(name: str):
    if name == "build_operator_layout":
        from veqpy.layout.binding import build_operator_layout

        return build_operator_layout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
