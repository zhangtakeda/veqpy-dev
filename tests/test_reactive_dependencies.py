from __future__ import annotations

from veqpy.base import Reactive


class ReactiveLeaf(Reactive):
    root_properties = {"x"}

    def __init__(self, x: int) -> None:
        super().__init__()
        self.x = x

    @property
    def y(self) -> int:
        return self.x + 1


class TupleParent(Reactive):
    root_properties = {"children"}

    def __init__(self, child: ReactiveLeaf) -> None:
        super().__init__()
        self.children = ((1, child),)

    @property
    def z(self) -> int:
        return self.children[0][1].y + 1


def test_nested_reactive_revision_inside_tuple_invalidates_parent_cache() -> None:
    child = ReactiveLeaf(1)
    parent = TupleParent(child)

    assert parent.z == 3
    assert parent.z == 3

    child.x = 3

    assert parent.z == 5
