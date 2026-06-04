from __future__ import annotations

import numpy as np

import veqpy.operator.packed_layout as packed_layout


def test_packed_layout_profile_first_switch_orders_by_name(monkeypatch) -> None:
    monkeypatch.setattr(packed_layout, "PACKED_LAYOUT_PROFILE_FIRST", True)
    profile_names = packed_layout.build_profile_names(1)
    profile_coeffs = {
        "h": [10.0, 11.0],
        "v": [20.0],
        "psin": [30.0, 31.0, 32.0],
    }

    profile_L, coeff_index, _ = packed_layout.build_profile_layout(
        profile_coeffs,
        profile_names=profile_names,
    )
    profile_index = {name: index for index, name in enumerate(profile_names)}

    assert coeff_index[profile_index["h"], 0] == 0
    assert coeff_index[profile_index["h"], 1] == 1
    assert coeff_index[profile_index["v"], 0] == 2
    assert coeff_index[profile_index["psin"], 0] == 3
    assert coeff_index[profile_index["psin"], 1] == 4
    assert coeff_index[profile_index["psin"], 2] == 5

    x = packed_layout.encode_packed_state(
        profile_coeffs,
        profile_L,
        coeff_index,
        profile_names=profile_names,
    )
    np.testing.assert_allclose(x, [10.0, 11.0, 20.0, 30.0, 31.0, 32.0])

    decoded = packed_layout.decode_packed_blocks(
        x,
        profile_L,
        coeff_index,
        profile_names=profile_names,
    )
    np.testing.assert_allclose(decoded[profile_index["h"]], [10.0, 11.0])
    np.testing.assert_allclose(decoded[profile_index["v"]], [20.0])
    np.testing.assert_allclose(decoded[profile_index["psin"]], [30.0, 31.0, 32.0])


def test_packed_layout_degree_first_switch_orders_by_degree(monkeypatch) -> None:
    monkeypatch.setattr(packed_layout, "PACKED_LAYOUT_PROFILE_FIRST", False)
    profile_names = packed_layout.build_profile_names(1)
    profile_coeffs = {
        "h": [10.0, 11.0],
        "v": [20.0],
        "psin": [30.0, 31.0, 32.0],
    }

    profile_L, coeff_index, _ = packed_layout.build_profile_layout(
        profile_coeffs,
        profile_names=profile_names,
    )
    profile_index = {name: index for index, name in enumerate(profile_names)}

    assert coeff_index[profile_index["h"], 0] == 0
    assert coeff_index[profile_index["v"], 0] == 1
    assert coeff_index[profile_index["psin"], 0] == 2
    assert coeff_index[profile_index["h"], 1] == 3
    assert coeff_index[profile_index["psin"], 1] == 4
    assert coeff_index[profile_index["psin"], 2] == 5

    x = packed_layout.encode_packed_state(
        profile_coeffs,
        profile_L,
        coeff_index,
        profile_names=profile_names,
    )
    np.testing.assert_allclose(x, [10.0, 20.0, 30.0, 11.0, 31.0, 32.0])
