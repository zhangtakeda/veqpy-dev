## residual

势能泛函定义为:

$$
\mathcal{L}=\int L\left(R, Z, \psi ,\psi_\rho, \psi_\theta,\ldots\right)  \mathrm{d} V
$$

$$
\begin{aligned}
\delta \mathcal{L} & =\int_0^1 \mathrm{d} \rho \int_0^{2 \pi} \mathrm{d} \theta~\left(\frac{\partial L}{\partial \psi} \delta \psi+\frac{\partial L}{\partial \psi_\rho} \delta \psi_\rho+\frac{\partial L}{\partial \psi_\theta} \delta \psi_\theta\right) \cdot(2 \pi J R)  \\
& =2 \pi \int_0^1 \mathrm{d} \rho \int_0^{2 \pi} \mathrm{d} \theta ~\left[J R \frac{\partial L}{\partial \psi}-\frac{\partial}{\partial \rho}\left(J R \frac{\partial L}{\partial \psi_\rho}\right) - \frac{\partial}{\partial \theta}\left(J R \frac{\partial L}{\partial \psi_\theta}\right)\right]\delta \psi\\
\end{aligned}
$$

Lagrangian density (参考 [Lao1981](https://doi.org/10.1063/1.863562)):

$$
L = \frac{|\nabla \psi|^2}{2 \mu_0 R^2} -\left(\frac{F^2}{2 \mu_0 R^2}+P\right)
$$

考虑到 $\psi_\theta \equiv 0$, 定义 Grad-Shafranov 算子:

$$
\begin{aligned}
\hat{G}& =- \mu_0 \left[J R \frac{\partial L}{\partial \psi}-\frac{\partial}{\partial \rho}\left(J R \frac{\partial L}{\partial \psi_\rho}\right) - \frac{\partial}{\partial \theta}\left(J R \frac{\partial L}{\partial \psi_\theta}\right) \right]\\
&= \frac{J}{R}\left(F F_\psi+\mu_0 R^2 P_\psi\right)+\left(\frac{g_{\theta \theta} \psi_\rho}{J R}\right)_\rho-\left(\frac{g_{\rho \theta} \psi_\rho}{J R}\right)_\theta\\
&=\alpha_1\hat{G}_{1}+{\alpha_2} \hat{G}_{2}
\end{aligned}
$$

$\hat{G} = 0$ 对应强形式的 Grad-Shafranov 方程. 其中 $\alpha_1$ 和 $\alpha_2$ 分别为源项缩放和磁通缩放.

$$
\begin{aligned}
\hat{G}_{1} &= \frac{ J}{\hat{\psi}_\rho R}\Big(\hat{FF_\rho}  +   R^2  \hat{P_\rho}\Big) = {\color{red}\frac{ J}{R}\Big(\hat{FF_\psi}  +   R^2  \hat{P_\psi}\Big)} \\
\hat{G}_{2} &= \frac{g_{\theta \theta}}{J R} \hat{\psi}_{\rho \rho}+\left[\left(\frac{g_{\theta \theta}}{J R}\right)_\rho-\left(\frac{g_{\rho \theta}}{J R}\right)_\theta\right] \hat{\psi}_{\rho}\\
\end{aligned}
$$

### 3. 变分方程

磁通随磁面移动, 对每个形状参数都存在对流变分为 0:

$$
\delta_{\bf r} \psi + \nabla \psi \cdot \delta\mathbf{r} = \left( \frac{\partial \psi}{\partial p_k} + \nabla \psi \cdot \frac{\partial \mathbf{r}}{\partial p_k}\right) \delta p_k = 0
$$

其中:

$$
\frac{\partial  {\psi}}{\partial R} =-\frac{Z_\theta}{J}\psi_\rho,\qquad
\frac{\partial  {\psi}}{\partial Z} =\frac{ R_\theta}{J}\psi_\rho
$$

此时泛函取极值时存在:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial p_k}&=  \frac{2\pi}{\mu_0}\int_0^1 \mathrm{d} \rho \int_0^{2 \pi} \mathrm{d} \theta  ~ \left(\hat{G} \cdot \nabla \psi \cdot \frac{\partial \mathbf{r}}{\partial p_k}\right) = 0
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \psi_k}&= -\frac{2\pi}{\mu_0}\int_0^1 \mathrm{d} \rho \int_0^{2 \pi} \mathrm{d} \theta \left(\hat{G}   \cdot \frac{\partial \psi}{\partial \psi_k}  \right)= 0
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial F_k}&= -\frac{2\pi}{\mu_0}\int_0^1 \mathrm{d} \rho \int_0^{2 \pi} \mathrm{d} \theta \left(\hat{G}   \cdot \frac{\partial F}{\partial F_k}  \right)= 0
\end{aligned}
$$

去除常数项, 矩方法可以写成:

$$
\begin{aligned}
\displaystyle \int_0^1 \mathrm{d} \rho \int_0^{2 \pi} \mathrm{d} \theta  ~  \left[\hat{G} \cdot \left( \frac{\partial \hat{\psi}}{\partial R} \frac{\partial R}{\partial p_k} + \frac{\partial \hat{\psi}}{\partial Z} \frac{\partial Z}{\partial p_k} \right) \right] &= 0\\
\end{aligned}
$$

$$
\begin{aligned}
\displaystyle \int_0^1 \mathrm{d} \rho \int_0^{2 \pi} \mathrm{d} \theta  ~  \left(\hat{G}  \cdot   \frac{\partial \hat{\psi}}{\partial \psi_k}  \right) &= 0\\
\end{aligned}
$$

$$
\begin{aligned}
\displaystyle \int_0^1 \mathrm{d} \rho \int_0^{2 \pi} \mathrm{d} \theta  ~  \left(\hat{G}  \cdot   \frac{\partial \hat{F}}{\partial F_k}  \right) &= 0\\
\end{aligned}
$$

- 对于 Strict 模式, 只需要对形状参数的残差方程进行优化即可.
- 对于 Robust 模式, 还需要同时优化代表 $\hat{\psi}$ 或 $F$ 的残差方程.
