from pathlib import Path

import numpy as np
import pytest

from veqpy.model import Boundary
from veqpy.model.geqdsk import Geqdsk


def _build_boundary(*, R0, Z0, a, ka, c_offsets, s_offsets, n=721):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta_bar = theta + c_offsets[0]
    for order in range(1, len(c_offsets)):
        theta_bar += c_offsets[order] * np.cos(order * theta)
    for order in range(1, len(s_offsets)):
        theta_bar += s_offsets[order] * np.sin(order * theta)
    R = R0 + a * np.cos(theta_bar)
    Z = Z0 - a * ka * np.sin(theta)
    return np.column_stack((R, Z))


def _max_bidirectional_distance(points_a, points_b):
    diff = points_a[:, None, :] - points_b[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    return max(distances.min(axis=1).max(), distances.min(axis=0).max())


def test_boundary_from_geqdsk_recovers_synthetic_boundary():
    truth = {
        "R0": 1.72,
        "Z0": -0.08,
        "a": 0.43,
        "ka": 1.68,
        "c_offsets": np.array([0.01, 0.08]),
        "s_offsets": np.array([0.0, -0.01, 0.01]),
    }

    geqdsk = Geqdsk()
    geqdsk.boundary = _build_boundary(**truth)
    geqdsk.R0 = truth["R0"]
    geqdsk.Z0 = truth["Z0"]

    boundary = Boundary.from_geqdsk(geqdsk, M=1, N=2)
    fitted_boundary = _build_boundary(
        R0=boundary.R0,
        Z0=boundary.Z0,
        a=boundary.a,
        ka=boundary.ka,
        c_offsets=boundary.c_offsets,
        s_offsets=boundary.s_offsets,
    )

    assert boundary.R0 == pytest.approx(truth["R0"])
    assert boundary.Z0 == pytest.approx(truth["Z0"])
    assert boundary.a == pytest.approx(truth["a"])
    assert boundary.ka == pytest.approx(truth["ka"], abs=1.0e-2)
    assert np.sign(boundary.c_offsets[1]) == np.sign(truth["c_offsets"][1])
    assert np.sign(boundary.s_offsets[1]) == np.sign(truth["s_offsets"][1])
    assert np.sign(boundary.s_offsets[2]) == np.sign(truth["s_offsets"][2])
    assert _max_bidirectional_distance(geqdsk.boundary, fitted_boundary) < 1.5e-2


def test_boundary_from_geqdsk_jointly_optimizes_r0_z0_a():
    truth = {
        "R0": 1.72,
        "Z0": -0.08,
        "a": 0.43,
        "ka": 1.68,
        "c_offsets": np.array([0.01, 0.08]),
        "s_offsets": np.array([0.0, -0.01, 0.01]),
    }

    geqdsk = Geqdsk()
    geqdsk.boundary = _build_boundary(**truth)
    geqdsk.R0 = truth["R0"] + 0.12
    geqdsk.Z0 = truth["Z0"] - 0.15

    boundary = Boundary.from_geqdsk(geqdsk, M=1, N=2)
    fitted_boundary = _build_boundary(
        R0=boundary.R0,
        Z0=boundary.Z0,
        a=boundary.a,
        ka=boundary.ka,
        c_offsets=boundary.c_offsets,
        s_offsets=boundary.s_offsets,
    )

    assert boundary.R0 == pytest.approx(truth["R0"], abs=1.0e-3)
    assert boundary.Z0 == pytest.approx(truth["Z0"], abs=1.0e-3)
    assert boundary.a == pytest.approx(truth["a"], abs=1.0e-3)
    assert boundary.ka == pytest.approx(truth["ka"], abs=1.0e-2)
    assert _max_bidirectional_distance(geqdsk.boundary, fitted_boundary) < 1.5e-2


def test_boundary_from_geqdsk_raises_for_empty_boundary():
    geqdsk = Geqdsk()

    with pytest.raises(ValueError, match="Boundary is empty"):
        Boundary.from_geqdsk(geqdsk, M=1, N=2)


def test_boundary_from_geqdsk_rejects_partial_order_specification():
    geqdsk = Geqdsk()
    geqdsk.boundary = _build_boundary(
        R0=1.72,
        Z0=-0.08,
        a=0.43,
        ka=1.68,
        c_offsets=np.array([0.01, 0.08]),
        s_offsets=np.array([0.0, -0.01, 0.01]),
    )

    with pytest.raises(ValueError, match="provided together or both omitted"):
        Boundary.from_geqdsk(geqdsk, M=2)


def test_boundary_from_geqdsk_rejects_non_positive_maxtol():
    geqdsk = Geqdsk()
    geqdsk.boundary = _build_boundary(
        R0=1.72,
        Z0=-0.08,
        a=0.43,
        ka=1.68,
        c_offsets=np.array([0.01, 0.08]),
        s_offsets=np.array([0.0, -0.01, 0.01]),
    )

    with pytest.raises(ValueError, match="maxtol must be positive"):
        Boundary.from_geqdsk(geqdsk, maxtol=0.0)


def test_boundary_from_geqdsk_rejects_orders_above_limit():
    geqdsk = Geqdsk()
    geqdsk.boundary = _build_boundary(
        R0=1.72,
        Z0=-0.08,
        a=0.43,
        ka=1.68,
        c_offsets=np.array([0.01, 0.08]),
        s_offsets=np.array([0.0, -0.01, 0.01]),
    )

    with pytest.raises(ValueError, match="must be <= 10"):
        Boundary.from_geqdsk(geqdsk, M=11, N=2)


def test_boundary_from_geqdsk_auto_selects_minimal_orders_on_synthetic_boundary():
    truth = {
        "R0": 1.72,
        "Z0": -0.08,
        "a": 0.43,
        "ka": 1.68,
        "c_offsets": np.array([0.01, 0.08, -0.02]),
        "s_offsets": np.array([0.0, -0.01, 0.01, 0.015]),
    }

    geqdsk = Geqdsk()
    geqdsk.boundary = _build_boundary(**truth)

    boundary = Boundary.from_geqdsk(geqdsk)
    fitted_boundary = _build_boundary(
        R0=boundary.R0,
        Z0=boundary.Z0,
        a=boundary.a,
        ka=boundary.ka,
        c_offsets=boundary.c_offsets,
        s_offsets=boundary.s_offsets,
    )

    assert len(boundary.c_offsets) == 3
    assert len(boundary.s_offsets) == 4
    rms = float(
        np.sqrt(
            np.mean(
                np.concatenate(
                    (
                        geqdsk.boundary[:, 0] - fitted_boundary[:, 0],
                        geqdsk.boundary[:, 1] - fitted_boundary[:, 1],
                    )
                )
                ** 2
            )
        )
    )
    assert rms < 1.0e-2


def test_boundary_from_geqdsk_warns_when_fixed_orders_miss_maxtol():
    geqdsk = Geqdsk(str(Path("geqdsk.txt")))

    with pytest.warns(UserWarning, match="did not satisfy maxtol"):
        boundary = Boundary.from_geqdsk(geqdsk, M=1, N=2, maxtol=1.0e-3)

    assert boundary.a > 0.0
    assert boundary.ka > 1.0


def test_boundary_from_geqdsk_builds_boundary_from_instance():
    truth = {
        "R0": 1.72,
        "Z0": -0.08,
        "a": 0.43,
        "ka": 1.68,
        "c_offsets": np.array([0.01, 0.08]),
        "s_offsets": np.array([0.0, -0.01, 0.01]),
    }

    geqdsk = Geqdsk()
    geqdsk.boundary = _build_boundary(**truth)
    geqdsk.Bt0 = 3.2
    geqdsk.R0 = truth["R0"]
    geqdsk.Z0 = truth["Z0"]

    boundary = Boundary.from_geqdsk(geqdsk, M=1, N=2)

    assert boundary.B0 == pytest.approx(3.2)
    assert boundary.R0 == pytest.approx(truth["R0"])
    assert boundary.Z0 == pytest.approx(truth["Z0"])
    assert boundary.a == pytest.approx(truth["a"])
    assert boundary.ka == pytest.approx(truth["ka"], abs=1.0e-2)
    assert np.sign(boundary.c_offsets[1]) == np.sign(truth["c_offsets"][1])
    assert np.sign(boundary.s_offsets[1]) == np.sign(truth["s_offsets"][1])
    assert np.sign(boundary.s_offsets[2]) == np.sign(truth["s_offsets"][2])


def test_boundary_from_geqdsk_reads_from_path():
    boundary = Boundary.from_geqdsk(Path("geqdsk.txt"), M=1, N=2, maxtol=1.0e-1)

    assert boundary.B0 > 0.0
    assert boundary.a > 0.0
    assert boundary.ka > 1.0


def test_boundary_from_geqdsk_fits_real_gfile():
    geqdsk = Geqdsk(str(Path("geqdsk.txt")))
    boundary = Boundary.from_geqdsk(geqdsk, M=1, N=2, maxtol=1.0e-1)

    assert boundary.a > 0.0
    assert boundary.ka > 1.0
    assert -np.pi <= boundary.c_offsets[0] <= np.pi
