from __future__ import annotations

import numpy as np

from veqpy.base import Reactive
from veqpy.model.profile import Profile


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


class ProfileParent(Reactive):
    root_properties = {"profile"}

    def __init__(self, profile: Profile) -> None:
        super().__init__()
        self.profile = profile

    @property
    def coeff_sum(self) -> float:
        coeff = self.profile.coeff
        return 0.0 if coeff is None else float(np.sum(coeff))


def test_profile_is_reactive_passive_spec() -> None:
    coeff = np.array([1.0, 2.0], dtype=np.float64)
    profile = Profile(scale=2, power=np.int64(3), envelope_power=2, offset=None, coeff=coeff)

    assert isinstance(profile, Reactive)
    assert profile.scale == 2.0
    assert profile.power == 3
    assert profile.envelope_power == 2
    assert profile.offset == 0.0
    np.testing.assert_allclose(profile.coeff, [1.0, 2.0])
    assert not profile.coeff.flags.writeable
    assert not coeff.flags.writeable

    revision = profile._revision
    profile.coeff = [3.0, 4.0]

    assert profile._revision == revision + 1
    np.testing.assert_allclose(profile.coeff, [3.0, 4.0])
    assert not profile.coeff.flags.writeable
    for runtime_attr in ("u_fields", "rp_fields", "env_fields", "T", "T_r", "T_rr"):
        assert not hasattr(profile, runtime_attr)


def test_profile_root_write_invalidates_parent_reactive_cache() -> None:
    profile = Profile(coeff=[1.0, 2.0])
    parent = ProfileParent(profile)

    assert parent.coeff_sum == 3.0
    assert parent.coeff_sum == 3.0

    profile.coeff = [3.0, 4.0]

    assert parent.coeff_sum == 7.0
