from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from veqpy.base import Reactive, Registry, Serial, read_serializer, write_serializer
from veqpy.base.reactive import depends_on

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------


def test_registry_decorator_populates_read_only_mapping():
    registry = Registry(str, type(lambda: None))

    @registry("alpha", "beta")
    def handler():
        return "ok"

    assert "alpha" in registry
    assert registry["alpha"] is handler
    assert registry["beta"] is handler
    assert list(registry) == ["alpha", "beta"]

    with pytest.raises(TypeError):
        registry.registry["gamma"] = handler


def test_registry_normalizes_string_keys_to_lowercase():
    registry = Registry(str, type(lambda: None))

    @registry("Alpha")
    def handler():
        return "ok"

    assert "alpha" in registry
    assert "ALPHA" in registry
    assert registry["alpha"] is handler
    assert registry["ALPHA"] is handler
    assert list(registry) == ["alpha"]


def test_registry_validates_key_and_value_types():
    registry = Registry(str, type(lambda: None))

    with pytest.raises(TypeError, match="Registry key must be str"):
        registry(1)

    with pytest.raises(TypeError, match="Registry value must be function"):
        registry("bad")(42)


# -----------------------------------------------------------------------------
# Reactive
# -----------------------------------------------------------------------------


def test_reactive_requires_explicit_root_properties():
    with pytest.raises(TypeError, match="must define root_properties explicitly"):

        class _BadReactive(Reactive):
            @property
            def value(self):
                return 1


def test_reactive_caches_and_invalidates_downstream_dependencies():
    class _SampleReactive(Reactive):
        root_properties = {"x"}

        def __init__(self, x):
            super().__init__()
            self.calls = 0
            self.x = x

        @property
        def y(self):
            self.calls += 1
            return self.x + 1

        @property
        def z(self):
            return self.y * 2

    sample = _SampleReactive(1)

    assert _SampleReactive.dependency_graph == {"y": {"x"}, "z": {"y"}}
    assert sample.y == 2
    assert sample.y == 2
    assert sample.calls == 1

    assert sample.z == 4
    assert sample.calls == 1

    sample.x = 3
    assert sample.y == 4
    assert sample.z == 8
    assert sample.calls == 2


def test_reactive_root_normalization_hook_applies_to_later_assignments():
    class _NormalizedReactive(Reactive):
        root_properties = {"x", "label"}

        def __init__(self, x, label):
            super().__init__()
            self.x = x
            self.label = label

        @classmethod
        def reactive_inspections(cls, name: str, value):
            if name == "x":
                value = int(value)
                if value < 0:
                    raise ValueError("x must be non-negative")
                return value
            if name == "label":
                return str(value).lower()
            return super().reactive_inspections(name, value)

        @property
        def y(self):
            return self.x + 1

    sample = _NormalizedReactive("1", "UPPER")

    assert sample.x == 1
    assert sample.cached_x == 1
    assert sample.label == "upper"
    assert sample.y == 2

    sample.x = "2"
    sample.label = "Mixed"
    assert sample.x == 2
    assert sample.cached_x == 2
    assert sample.label == "mixed"
    assert sample.y == 3

    with pytest.raises(ValueError, match="x must be non-negative"):
        sample.x = -1


def test_reactive_depends_on_declares_non_ast_dependencies():
    class _ExplicitReactive(Reactive):
        root_properties = {"raw"}

        def __init__(self, raw):
            super().__init__()
            self.raw = raw

        @property
        @depends_on("raw")
        def alias(self):
            return 10

    assert _ExplicitReactive.dependency_graph == {"alias": {"raw"}}


def test_reactive_rejects_circular_property_dependencies():
    with pytest.raises(ValueError, match="Circular dependency detected"):

        class _CyclicReactive(Reactive):
            root_properties = {"root"}

            @property
            def a(self):
                return self.b

            @property
            def b(self):
                return self.a


# -----------------------------------------------------------------------------
# Serial
# -----------------------------------------------------------------------------


class _BaseSerialChild(Serial):
    def __init__(self, value: int):
        self.value = value

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        return {"value": int}


class _BaseSerialParent(Serial):
    def __init__(self, child: _BaseSerialChild, values: np.ndarray, labels: list[str]):
        self.child = child
        self.values = values
        self.labels = labels

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        return {
            "child": _BaseSerialChild,
            "values": np.ndarray,
            "labels": list[str],
        }


@dataclass
class _BaseDataclassSerial(Serial):
    count: int
    names: list[str]


class _BaseDispatchSerial(Serial):
    def __init__(self, value: int = 0):
        self.value = value

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        return {"value": int}


@write_serializer("basecustom")
def _write_basecustom(instance: _BaseDispatchSerial, file: str) -> None:
    Path(file).write_text(str(instance.value), encoding="utf-8")


@read_serializer("basecustom")
def _read_basecustom(instance: _BaseDispatchSerial, file: str) -> None:
    instance.value = int(Path(file).read_text(encoding="utf-8"))


def test_serial_json_roundtrip_preserves_nested_serial_and_arrays(tmp_path):
    expected = _BaseSerialParent(
        child=_BaseSerialChild(3),
        values=np.array([1.0, 2.0, 3.0]),
        labels=["left", "right"],
    )
    path = tmp_path / "serial.json"

    expected.write(str(path))
    loaded = _BaseSerialParent.load(str(path))

    assert isinstance(loaded, _BaseSerialParent)
    assert isinstance(loaded.child, _BaseSerialChild)
    assert loaded.child.value == expected.child.value
    assert np.allclose(loaded.values, expected.values)
    assert loaded.labels == expected.labels


def test_serial_pickle_roundtrip_preserves_nested_serial_and_arrays(tmp_path):
    expected = _BaseSerialParent(
        child=_BaseSerialChild(5),
        values=np.array([4.0, 5.0]),
        labels=["pkl"],
    )
    path = tmp_path / "serial.pkl"

    expected.write(str(path))
    loaded = _BaseSerialParent.load(str(path))

    assert isinstance(loaded, _BaseSerialParent)
    assert isinstance(loaded.child, _BaseSerialChild)
    assert loaded.child.value == expected.child.value
    assert np.allclose(loaded.values, expected.values)
    assert loaded.labels == expected.labels


def test_serial_dataclass_attributes_are_inferred():
    assert _BaseDataclassSerial.serial_attributes() == {
        "count": int,
        "names": list[str],
    }


def test_serial_custom_serializer_dispatches_by_extension(tmp_path):
    path = tmp_path / "payload.basecustom"

    _BaseDispatchSerial(17).write(str(path))
    loaded = _BaseDispatchSerial().read(str(path))

    assert loaded.value == 17


def test_serial_check_validates_type_and_empty_values():
    bad_type = _BaseSerialChild("not-an-int")
    with pytest.raises(TypeError, match="Attribute 'value'"):
        bad_type.check()

    bad_empty = _BaseSerialParent(
        child=_BaseSerialChild(1),
        values=np.array([], dtype=np.float64),
        labels=["ok"],
    )
    with pytest.raises(ValueError, match="empty array"):
        bad_empty.check()
