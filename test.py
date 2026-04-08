from veqpy.model import Geqdsk

import numpy as np
from veqpy.model import Boundary, Grid
from veqpy.operator import Operator, OperatorCase
from veqpy.solver import Solver, SolverConfig

# 1. 读 GEQDSK
geqdsk = Geqdsk()
geqdsk.read_geqdsk("constructed_from_pq.geqdsk")

# 2. 从 GEQDSK 里拿两条剖面
P_psi = np.asarray(geqdsk.dp, dtype=float)
q = np.asarray(geqdsk.q, dtype=float)

# 3. 从 GEQDSK 里拿边界
boundary = Boundary.from_geqdsk(geqdsk, M=4, N=5)

# 4. 给求解器一些初始系数
coeffs = {
    "h": [0.0] * 10,
    "k": [0.0] * 10,
    "v": [0.0] * 10,
    "c0": [0.0] * 10,
    "c1": [0.0] * 5,
    "c2": [0.0] * 5,
    "s1": [0.0] * 10,
    "s2": [0.0] * 5,
    "s3": [0.0] * 5,
}

# 5. 数值网格
grid = Grid(Nr=32, Nt=32, scheme="legendre")

# 6. 打包成一个求解任务
op_case = OperatorCase(
    route="PQ",  # 如果你的本地版本不认 name，再换回 route="PQ"
    profile_coeffs=coeffs,
    boundary=boundary,
    heat_input=P_psi,
    current_input=q,
    coordinate="psin",
    nodes="uniform",
    Ip=6.0e5,
)

# 7. 求解器设置
config = SolverConfig(
    method="hybr",
    enable_verbose=True,
    enable_warmstart=False,
)

# 8. 求解
op = Operator(grid=grid, case=op_case)
solver = Solver(operator=op, config=config)

solver.solve()
eq = solver.build_equilibrium()

print(eq)
eq.plot("demo-eq.png")
