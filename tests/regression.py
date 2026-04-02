import numpy as np
from scipy.interpolate import interp1d

from veqpy.model import Boundary, Grid
from veqpy.operator import Operator, OperatorCase
from veqpy.solver import Solver, SolverConfig

coeffs = {
    "h": [0.0] * 3,
    "k": [0.0] * 3,
    "s1": [0.0] * 3,
}

psin_coeffs = {
    **coeffs,
    "psin": [0.0] * 5,
}


bdry = Boundary(
    a=1.05 / 1.85,
    R0=1.05,
    Z0=0.0,
    B0=3.0,
    ka=2.2,
    s_offsets=np.array([0.0, float(np.arcsin(0.5))]),
)

config = SolverConfig(
    method="hybr",
    enable_verbose=True,
    enable_warmstart=False,
)


grid = Grid(
    Nr=12,
    Nt=12,
    scheme="legendre",
)


def pf_reference_profiles(psin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta0 = 0.75

    alpha_p, alpha_f = 5.0, 3.32
    exp_ap, exp_af = np.exp(alpha_p), np.exp(alpha_f)
    den_p, den_f = 1.0 + exp_ap * (alpha_p - 1.0), 1.0 + exp_af * (alpha_f - 1.0)

    current_input = (1.0 - beta0) * alpha_f * (np.exp(alpha_f * psin) - exp_af) / den_f
    heat_input = beta0 * alpha_p * (np.exp(alpha_p * psin) - exp_ap) / den_p
    return current_input, heat_input


def warmup_solver(solver: Solver) -> None:
    """Run one silent solve so demo timings exclude first-call JIT overhead."""
    for _ in range(10):
        solver.solve(
            enable_verbose=False,
            enable_history=False,
            enable_warmstart=False,
        )


# uniform inputs(rho)
rho_uni = np.linspace(0.0, 1.0, 21)
psin_uni = rho_uni * rho_uni
psin_r_uni = 2.0 * rho_uni
FFn_psin_uni, Pn_psin_uni = pf_reference_profiles(psin_uni)
FFn_r_uni = FFn_psin_uni * psin_r_uni
Pn_r_uni = Pn_psin_uni * psin_r_uni

case_1 = OperatorCase(
    route="PF",
    coordinate="rho",
    nodes="uniform",
    profile_coeffs=coeffs,
    boundary=bdry,
    heat_input=Pn_r_uni,
    current_input=FFn_r_uni,
    Ip=3.0e6,
)

solver = Solver(operator=Operator(grid, case_1), config=config)
warmup_solver(solver)
solver.solve()

res_1 = solver.result
eq_1 = solver.build_equilibrium()
eq_1.plot("tests/regression/1.png")

print(res_1)
print(eq_1)

psin_src_grid = eq_1.psin
psin_uni = np.linspace(0.0, 1.0, 21)
rho_from_psin = interp1d(psin_src_grid, eq_1.rho, kind="cubic")(psin_uni)
psin_r_from_psin = interp1d(psin_src_grid, eq_1.psin_r, kind="cubic")(psin_uni)
FFn_r_from_psin = interp1d(rho_uni, FFn_r_uni, kind="cubic")(rho_from_psin)
Pn_r_from_psin = interp1d(rho_uni, Pn_r_uni, kind="cubic")(rho_from_psin)
FFn_psin_uni = FFn_r_from_psin / psin_r_from_psin
Pn_psin_uni = Pn_r_from_psin / psin_r_from_psin

case_2 = OperatorCase(
    route="PF",
    coordinate="psin",
    nodes="uniform",
    profile_coeffs=psin_coeffs,
    boundary=bdry,
    heat_input=Pn_psin_uni,
    current_input=FFn_psin_uni,
    Ip=3.0e6,
)

solver = Solver(operator=Operator(grid, case_2), config=config)
warmup_solver(solver)
solver.solve()

res_2 = solver.result
eq_2 = solver.build_equilibrium()
eq_2.plot("tests/regression/2.png")
eq_1.compare(eq_2, "tests/regression/compare.png")

print(res_2)
print(eq_2)
