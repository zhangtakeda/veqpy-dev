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

Robust 方案额外有:

$$
\begin{aligned}
F &= R_0 B_0\left[1+(1-\rho^2)^2\sum_{l=0}^L F_l T_l(x)\right]\\
\hat{\psi} &=\rho^2\left[1+(1-\rho^2)\sum_{l=0}^L \hat{\psi}_l T_l(x)\right]\\
\end{aligned}
$$
