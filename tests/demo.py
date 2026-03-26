import numpy as np
from rich import print

from veqpy.model import Boundary, Grid
from veqpy.operator import Operator, OperatorCase
from veqpy.solver import Solver, SolverConfig

WARMUP_TIMES = 10

BOUNDARY = Boundary(
    a=1.05 / 1.85,
    R0=1.05,
    Z0=0.0,
    B0=3.0,
    ka=2.2,
    s_offsets=np.array([0.0, float(np.arcsin(0.5))]),
)

CONFIG = SolverConfig(
    method="hybr",
    enable_verbose=True,
    enable_warmstart=False,
)


LOWER_COEFFS = {
    "h": [0.0] * 3,
    "k": [0.0] * 3,
    "s1": [0.0] * 3,
}


HIGHER_COEFFS = {
    "h": [0.0] * 10,
    "v": [0.0] * 10,
    "k": [0.0] * 10,
    "c0": [0.0] * 10,
    "c1": [0.0] * 10,
    "c2": [0.0] * 10,
    "s1": [0.0] * 10,
    "s2": [0.0] * 10,
    "s3": [0.0] * 10,
}


LOWER_GRID = Grid(
    Nr=12,
    Nt=12,
    scheme="legendre",
)

HIGHER_GRID = Grid(
    Nr=32,
    Nt=32,
    scheme="legendre",
)


def warmup_solver(solver: Solver) -> None:
    """Run one silent solve so demo timings exclude first-call JIT overhead."""
    for _ in range(WARMUP_TIMES):
        solver.solve(enable_verbose=False, enable_history=False)


def pf_reference_profiles(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75
    psin = rho**2
    psin_r = 2.0 * rho

    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p, den_f = 1.0 + exp_ap * (alpha_p - 1.0), 1.0 + exp_af * (alpha_f - 1.0)

    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f * psin_r
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p * psin_r
    return current_input, heat_input


lower_FFn_r, lower_Pn_r = pf_reference_profiles(LOWER_GRID.rho)
higher_FFn_r, higher_Pn_r = pf_reference_profiles(HIGHER_GRID.rho)

LOWER_CASE = OperatorCase(
    profile_coeffs=LOWER_COEFFS,
    boundary=BOUNDARY,
    heat_input=lower_Pn_r,
    current_input=lower_FFn_r,
    Ip=3.0e6,
)

HIGHER_CASE = OperatorCase(
    profile_coeffs=HIGHER_COEFFS,
    boundary=BOUNDARY,
    heat_input=higher_Pn_r,
    current_input=higher_FFn_r,
    Ip=3.0e6,
)


if __name__ == "__main__":
    operator_1 = Operator(grid=LOWER_GRID, case=LOWER_CASE, name="PF", derivative="rho")
    operator_2 = Operator(grid=HIGHER_GRID, case=HIGHER_CASE, name="PF", derivative="rho")

    solver_1 = Solver(operator=operator_1, config=CONFIG)
    solver_2 = Solver(operator=operator_2, config=CONFIG)

    warmup_solver(solver_1)
    warmup_solver(solver_2)

    print("\n\n=== Typical Coefficients + Typical Grid ===")
    x_typical = solver_1.solve()
    eq1 = solver_1.build_equilibrium()
    eq1.write("tests/demo/demo-1.json")
    eq1.plot("tests/demo/demo-1.png")
    print(eq1)

    print("\n\n=== High Precision Coefficients + High Precision Grid ===")
    x_high_precision = solver_2.solve()
    eq2 = solver_2.build_equilibrium()
    eq2.write("tests/demo/demo-2.json")
    eq2.plot("tests/demo/demo-2.png")
    eq2.compare(
        eq1,
        "tests/demo/demo-comparison.png",
        label_ref="Higher",
        label_other="Lower",
    )
    print(eq2)
