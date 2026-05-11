from pathlib import Path
from types import SimpleNamespace
import pickle

import numpy as np

from tests import benchmark as benchmark_script


def _make_reference_equilibrium(*, rho: np.ndarray) -> SimpleNamespace:
    rho = np.asarray(rho, dtype=np.float64)
    source = np.linspace(0.0, 1.0, rho.shape[-1], dtype=np.float64)
    return SimpleNamespace(
        grid=SimpleNamespace(rho=rho),
        psin=source.copy(),
        psin_r=source.copy(),
        FFn_psin=source.copy(),
        Pn_psin=source.copy(),
    )


def test_reference_equilibrium_cache_compatibility_rejects_non_1d_rho():
    equilibrium = _make_reference_equilibrium(rho=np.ones((4, 8), dtype=np.float64))

    assert not benchmark_script._is_reference_equilibrium_cache_compatible(equilibrium)


def test_load_reference_cache_rejects_incompatible_cached_equilibrium(tmp_path, monkeypatch):
    cache_path = Path(tmp_path) / "reference_bundle.pkl"
    payload = {
        "signature": benchmark_script._reference_cache_signature(),
        "bundle": {
            "result": object(),
            "equilibrium": _make_reference_equilibrium(rho=np.ones((4, 8), dtype=np.float64)),
            "ref_profiles": {},
            "reference_shape_x": np.zeros(3, dtype=np.float64),
            "rho_axis": np.linspace(0.0, 1.0, 8, dtype=np.float64),
            "psin_axis": np.linspace(0.0, 1.0, 8, dtype=np.float64),
        },
    }
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    monkeypatch.setattr(benchmark_script, "_reference_cache_path", lambda: cache_path)

    assert benchmark_script._load_reference_cache() is None
