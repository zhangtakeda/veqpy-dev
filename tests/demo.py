import numpy as np
from rich import print

from veqpy.model import Grid
from veqpy.operator import Operator, OperatorCase
from veqpy.solver import Solver, SolverConfig

COEFFS = {
    "h": [0.0] * 3,
    "v": None,
    "k": [0.0] * 3,
    "c0": None,
    "c1": None,
    "s1": [0.0] * 3,
    "s2": None,
}


HIGH_PRECISION_COEFFS = {
    "h": [0.0] * 10,
    "v": [0.0] * 5,
    "k": [0.0] * 10,
    "c0": [0.0] * 5,
    "c1": [0.0] * 5,
    "s1": [0.0] * 10,
    "s2": [0.0] * 5,
}

GRID = Grid(
    Nr=12,
    Nt=12,
    scheme="legendre",
)

HIGH_PRECISION_GRID = Grid(
    Nr=32,
    Nt=32,
    scheme="legendre",
)


CONFIG = SolverConfig(
    method="hybr",
    enable_verbose=True,
    enable_warmstart=False,
)


HOMO_CONFIG = SolverConfig(
    method="lm",
    enable_verbose=True,
    enable_warmstart=False,
    enable_homotopy=True,
)


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


FFn_r_1, Pn_r_1 = pf_reference_profiles(GRID.rho)
FFn_r_2, Pn_r_2 = pf_reference_profiles(HIGH_PRECISION_GRID.rho)

CASE_1 = OperatorCase(
    coeffs_by_name=COEFFS,
    a=1.05 / 1.85,
    R0=1.05,
    Z0=0.0,
    B0=3.0,
    ka=2.2,
    s1a=float(np.arcsin(0.5)),
    heat_input=Pn_r_1,
    current_input=FFn_r_1,
    Ip=3.0e6,
)

CASE_2 = OperatorCase(
    coeffs_by_name=HIGH_PRECISION_COEFFS,
    a=1.05 / 1.85,
    R0=1.05,
    Z0=0.0,
    B0=3.0,
    ka=2.2,
    s1a=float(np.arcsin(0.5)),
    heat_input=Pn_r_2,
    current_input=FFn_r_2,
    Ip=3.0e6,
)

CASE_3 = OperatorCase(
    coeffs_by_name=COEFFS,
    a=1.05 / 1.85,
    R0=1.05,
    Z0=0.0,
    B0=3.0,
    ka=2.2,
    s1a=float(np.arcsin(0.5)),
    heat_input=Pn_r_2,
    current_input=FFn_r_2,
    Ip=3.0e6,
)


if __name__ == "__main__":
    operator_1 = Operator(grid=GRID, case=CASE_1, name="PF", derivative="rho")
    operator_2 = Operator(grid=HIGH_PRECISION_GRID, case=CASE_2, name="PF", derivative="rho")
    operator_3 = Operator(grid=HIGH_PRECISION_GRID, case=CASE_3, name="PF", derivative="rho")
    solver_1 = Solver(operator=operator_1, config=CONFIG)
    solver_2 = Solver(operator=operator_2, config=CONFIG)
    solver_3 = Solver(operator=operator_3, config=CONFIG)
    solver_4 = Solver(operator=operator_2, config=HOMO_CONFIG)

    print("\n=== Typical Coefficients + Typical Grid ===")
    x1 = solver_1.solve()
    eq1 = solver_1.build_equilibrium()
    eq1.write("tests/demo-1.json")
    eq1.plot("tests/demo-1.png")
    print(eq1)

    print("\n=== High Precision Coefficients + High Precision Grid ===")
    x2 = solver_2.solve()
    eq2 = solver_2.build_equilibrium()
    eq2.write("tests/demo-2.json")
    eq2.plot("tests/demo-2.png")
    eq2.compare(
        eq1,
        "tests/demo-coeffs-comparison.png",
        label_ref="High Precision",
        label_other="Typical",
    )
    print(eq2)

    print("\n=== Typical Coefficients + High Precision Grid ===")
    x3 = solver_3.solve()
    eq3 = solver_3.build_equilibrium()
    eq3.write("tests/demo-3.json")
    eq3.plot("tests/demo-3.png")
    eq3.compare(
        eq1,
        "tests/demo-grid-comparison.png",
        label_ref="High Precision",
        label_other="Typical",
    )
    print(eq3)

    print("\n=== Homogeneous + High Precision Coefficients + High Precision Grid ===")
    x4 = solver_4.solve()
    eq4 = solver_4.build_equilibrium()
    eq4.write("tests/demo-4.json")
    eq4.plot("tests/demo-4.png")
    eq4.compare(
        eq2,
        "tests/demo-homo-comparison.png",
        label_ref="Homogeneous",
        label_other="Non-Homogeneous",
    )
    print(eq4)
