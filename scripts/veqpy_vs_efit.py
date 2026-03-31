from pathlib import Path

from detail.gfile_compare import CompareConfig, run_compare

BOUNDARY_FIT_M = 6
DEFAULT_PROFILE_COEFFS = {
    "psin": [0.0] * 5,
    "h": [0.0] * 5,
    "k": [0.0] * 5,
    "v": [0.0] * 5,
    "c0": [0.0] * 3,
    "c1": [0.0] * 3,
    "c2": [0.0] * 3,
    "c3": [0.0] * 3,
    "s1": [0.0] * 3,
    "s2": [0.0] * 3,
}

script_dir = Path(__file__).resolve().parent
compare_config = CompareConfig(
    gfile_path=script_dir / "EFIT",
    plot_path=script_dir / "veqpy_vs_efit.png",
    summary_path=script_dir / "veqpy_vs_efit.json",
    profile_coeffs=DEFAULT_PROFILE_COEFFS,
    nr=32,
    nt=32,
    solver_method="trf",
    solver_maxfev=2000,
    boundary_fit_m=BOUNDARY_FIT_M,
    boundary_maxtol=1.0e-2,
)
run_compare(compare_config)
