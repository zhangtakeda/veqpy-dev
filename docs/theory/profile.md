## profile

Chebyshev 多项式进行径向展开:

$$
\begin{aligned}
x &= 2 \rho^2-1\\
T_0(x)&=1 \\
T_1(x)&=x \\
T_{l+1}(x)&=2 x T_l(x)-T_{l-1}(x)
\end{aligned}
$$

Fourier 进行极向展开:

$$
\begin{aligned}
\bar{\theta}(\rho, \theta)&=\theta+c_0+\sum_{m = 1}^{M} c_m \cos (m \theta)+\sum_{n = 1}^N s_n \sin (n \theta)\\
R(\rho, \theta) &= R_0+a(h +\rho  \cos \bar{\theta})\\
Z(\rho, \theta) &= Z_0+a(v -\rho  \kappa  \sin \theta)\\
\end{aligned}
$$

当前实现对应的运行时约束是:

- `Grid.K_max` 给出可表示的最高 Fourier 阶数
- `Grid.rho_powers` 保存 `rho^0 .. rho^(K+1)`
- `Grid.cos_ktheta` / `Grid.sin_ktheta` 保存 `k = 0 .. K` 的 Fourier 表
- `OperatorCase.c_offsets` 保存 `c0 .. cK`
- `OperatorCase.s_offsets` 保存 `s0 .. sK`，其中 `s0` 恒为 0
- `profile_coeffs` 中的 `c{k}` / `s{k}` 表示对应阶数的可优化径向修正

在径向展开到 $L$ 阶, 有:

$$
\begin{aligned}
h &=(1-\rho^2)\sum_{l=0}^L h_l T_l(x)\\
v &=(1-\rho^2)\sum_{l=0}^L v_l T_l(x)\\
\kappa &=\kappa_a+(1-\rho^2)\sum_{l=0}^L \kappa_l T_l(x)\\
c_0 &=c_{0 a}+(1-\rho^2)\sum_{l=0}^L c_{0 l} T_l(x)\\
c_m &=\rho^m\left[c_{m a}+(1-\rho^2)\sum_{l=0}^L c_{m l} T_l(x) \right], \quad \text{} m \ge 1\\
s_n &=\rho^n\left[s_{n a}+(1-\rho^2)\sum_{l=0}^L s_{n l} T_l(x) \right], \quad n \ge 1\\
\end{aligned}
$$

实现上还保留一个重要语义:

- 当 `c_m` 或 `s_n` 的 `coeff` 为 `None` 时，它不进入 packed 优化变量
- 此时如果边界 offset 非 0, 它会退化为固定 profile:

$$
\begin{aligned}
c_m(\rho) &= \rho^m c_{m a}, \quad m \ge 1\\
s_n(\rho) &= \rho^n s_{n a}, \quad n \ge 1
\end{aligned}
$$

- 如果边界 offset 也为 0, 则该阶 profile 在当前 runtime 中会被 `effective active order` 裁剪掉

Robust 方案额外有:

$$
\begin{aligned}
F &= R_0 B_0\left[1+(1-\rho^2)^2\sum_{l=0}^L F_l T_l(x)\right]\\
\hat{\psi} &=\rho^2\left[1+(1-\rho^2)\sum_{l=0}^L \hat{\psi}_l T_l(x)\right]\\
\end{aligned}
$$
