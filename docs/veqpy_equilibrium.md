---
title: test
self-contained: true
---

## 磁面几何

$\theta$ 是顺时针方向, $\phi$ 是垂直向内的方向, 仅考虑环向对称的平衡求解.

- 磁面坐标: $(\rho, \theta, \phi)$
- 几何坐标: $(R, Z, -\phi)$

### 1. 度规

协变度规 (这里下标代表协变基而不是偏导数):

$$
\left[\begin{array}{ll}
g_{\rho \rho} & g_{\rho \theta} \\
g_{\theta \rho} & g_{\theta \theta}
\end{array}\right]=\left[\begin{array}{cc}
R_\rho^2+Z_\rho^2 & R_\rho R_\theta+Z_\rho Z_\theta \\
R_\rho R_\theta+Z_\rho Z_\theta & R_\theta^2+Z_\theta^2
\end{array}\right]
$$

逆变度规:

$$
\left[\begin{array}{ll}
g^{\rho \rho} & g^{\rho \theta} \\
g^{\theta \rho} & g^{\theta \theta}
\end{array}\right]=\frac{1}{J^2}\left[\begin{array}{cc}
R_\theta^2+Z_\theta^2 & -\left(R_\rho R_\theta+Z_\rho Z_\theta\right) \\
-\left(R_\rho R_\theta+Z_\rho Z_\theta\right) & R_\rho^2+Z_\rho^2
\end{array}\right]
$$

Jacobian:

$$
\begin{aligned}
J  &= \sqrt{\mathrm{det}(\mathbf{g}_{ab})}  =R_\theta Z_\rho-R_\rho Z_\theta\\
J_\rho & =-\left(R_{\rho \rho} Z_\theta-R_{\rho \theta} Z_\rho+R_\rho Z_{\rho \theta}-R_\theta Z_{\rho \rho}\right) \\
J_\theta & =-\left(R_{\rho \theta} Z_\theta-R_{\theta \theta} Z_\rho+R_\rho Z_{\theta \theta}-R_\theta Z_{\rho \theta}\right)\\
\mathcal{J}_{\rm 3D} &\equiv JR\\
\end{aligned}
$$

### 2. 几何

微元:

$$
\begin{aligned}
\mathrm{d}V &= 2\pi JR ~\mathrm{d}\rho~\mathrm{d}\theta\\
\mathrm{d}S &= J ~\mathrm{d}\rho~\mathrm{d}\theta\\
\mathrm{d}L &= |\nabla \rho| J ~\mathrm{d}\theta = \sqrt{g_{\theta\theta}}  ~\mathrm{d}\theta
\end{aligned}
$$

磁面截面面积:

$$
\begin{aligned}
S &=-\int_0^{2 \pi} R Z_\theta ~\mathrm{d}\theta \\
S_\rho  &= \int_0^{2\pi} J ~\mathrm{d}\theta \\
\end{aligned}
$$

磁面体积:

$$
\begin{aligned}
V &=-2 \pi \int_0^\rho \mathrm{d}\rho \int_0^{2 \pi}  J R~\mathrm{d}\theta=-\pi \int_0^{2 \pi} R^2 Z_\theta ~ \mathrm{d}\theta, \\
V_\rho &= 2 \pi \int_0^{2 \pi} J R~\mathrm{d}\theta
\end{aligned}
$$

Ampere 环路定理的几何参数:

$$
\begin{aligned}
K &= \frac{1}{2\pi \mu_0} \int_0^{2 \pi} \frac{g_{\theta\theta}}{J R} \mathrm{d}\theta\\[8px]
K_\rho&=\frac{1}{2 \pi \mu_0} \int_0^{2 \pi} \left(\frac{g_{\theta \theta}}{J R}\right)_\rho \mathrm{d} \theta
\end{aligned}
$$

$$
\begin{aligned}
\hat{K} &= \frac{1}{2\pi} \int_0^{2 \pi} \frac{g_{\theta\theta}}{J R} \mathrm{d}\theta\\[8px]
\hat{K}_\rho&=\frac{1}{2 \pi} \int_0^{2 \pi} \left(\frac{g_{\theta \theta}}{J R}\right)_\rho \mathrm{d} \theta
\end{aligned}
$$

Green 公式对环向磁通的几何参数:

$$
\begin{aligned}
L &=  \frac{\mu_0}{2\pi} \int_0^{2 \pi} \frac{R_{\theta}Z}{R} \mathrm{d}\theta\\
L_\rho &=  \frac{\mu_0}{2 \pi} \int_0^{2 \pi} \frac{J}{R} \mathrm{d}\theta\\
\end{aligned}
$$

$$
\begin{aligned}
\hat{L} &=  \frac{1}{2\pi} \int_0^{2 \pi} \frac{R_{\theta}Z}{R} \mathrm{d}\theta\\
\hat{L}_\rho &=  \frac{1}{2 \pi} \int_0^{2 \pi} \frac{J}{R} \mathrm{d}\theta\\
\end{aligned}
$$

### 3. 物理量

定义与总电流和实际磁通值的物理系数分别为 $\alpha_1$ 和 $\alpha_2$:

$$
\begin{aligned}
\psi_\rho &= \alpha_2 \hat{\psi}_\rho \\
F F_\rho & =\alpha_1 \alpha_2 \hat{FF_\rho} \\
P_\rho & =\frac{\alpha_1 \alpha_2}{\mu_0} \hat{P}_\rho
\end{aligned}
$$

因此有:

$$
\color{red}
\begin{aligned}
FF_\psi  &= \alpha_1   \hat{FF_\psi} = \alpha_1\frac{\hat{FF_\rho}}{\hat{\psi}_\rho} \\
P_\psi &=\frac{\alpha_1}{\mu_0} \hat{P}_\psi = \frac{\alpha_1}{\mu_0}\frac{\hat{P_\rho}}{\hat{\psi}_\rho}\\
\end{aligned}
$$

- **环向/极向磁通**

  $$
  \begin{aligned}
  \Phi &=\int_0^\rho \mathrm{d}\rho  ~F \int_0^{2 \pi} \frac{J}{R} ~\mathrm{d}\theta = 2\pi \int_0^\rho   F \hat{L}_\rho ~\mathrm{d}\rho\\
  \Psi &= 2\pi (\psi - \psi_0)= 2\pi \alpha_2\hat{\psi}
  \end{aligned}
  $$

- **安全因子和磁剪切**

  $$
  \begin{aligned}
  q &=\frac{  \Phi_\rho}{  \Psi_\rho} =\frac{F}{2 \pi \psi_\rho} \int_0^{2 \pi} \frac{J}{R}~ \mathrm{d}\theta =\frac{F \hat{L}_\rho}{\alpha_2\hat{\psi}_\rho}\\
  s &= \rho \frac{q_\rho}{q}
  \end{aligned}
  $$

- **环向/极向电流**

  $$
  \begin{aligned}
  I_{\rm tor} &=\frac{\psi_\rho}{\mu_0} \int_0^{2 \pi} \frac{g_{\theta\theta}}{J R} \mathrm{d}\theta = \frac{2\pi \alpha_2}{\mu_0} \hat{K} \hat{\psi}_\rho=\frac{2 \pi }{\mu_0  }\frac{  \hat{K} \hat{L}_\rho}{  q} F =2 \pi F \int_0^\rho \frac{\hat{L}_\rho j_{\|} }{F } \mathrm{d} \rho\\
  I_{\rm pol} &=\frac{2\pi}{\mu_0} (F_0 - F)
  \end{aligned}
  $$

- **等离子体电流**

  $$
  I_{p}=  I_{\rm tor}(1)=\int_0^1 {\rm d} \rho \int_0^{2 \pi} j_\phi J ~{\rm d} \theta
  $$

- **平行电流密度**

  $$
  j_{\|} =\frac{\langle~\mathbf{j} \cdot \mathbf{B}\rangle}{\langle\mathbf{B} \cdot \nabla \phi\rangle}= \frac{\mu_0F}{ L_\rho}\left(\frac{\psi_\rho K}{F} \right)_\rho= \frac{\alpha_2 F}{\mu_0 \hat{L}_\rho}\left(\frac{\hat{\psi}_\rho \hat{K}}{F} \right)_\rho
  $$

- **环向电流密度**

  $$
  \begin{aligned}
  j_{\phi}(\rho,\theta)&=- \left(\frac{F F_\psi}{\mu_0 R}+R P_\psi\right) \\
  &=- \frac{\alpha_1}{\mu_0\hat{\psi}_\rho}\left(\frac{\hat{F F_\rho}}{R}+R \hat{P}_\rho\right)\\
  &=\color{red}- \frac{\alpha_1}{\mu_0}\left(\frac{\hat{F F_\psi}}{R}+R \hat{P}_\psi\right)\\
  \end{aligned}
  $$

  $$
  \begin{aligned}
  j_{\rm tor} \equiv \langle j_{\phi} \rangle_S=  \frac{1}{S_\rho} \frac{\mathrm{d} I_{\text {tor }}}{\mathrm{d} \rho}=-\frac{1}{S_\rho}\left(\frac{2\pi}{\mu_0^2}\ {F F_\psi } L_\rho +\frac{V_\rho P_\psi}{2\pi}\right)\\
  =-\frac{\alpha_1}{\mu_0\hat{\psi}_\rho S_\rho}\left( 2\pi \hat{F F_\rho } \hat{L}_\rho +\frac{V_\rho \hat{P}_\rho}{2\pi}\right) \\
  =\color{red}-\frac{\alpha_1}{\mu_0S_\rho}\left( 2\pi \hat{F F_\psi } \hat{L}_\rho +\frac{V_\rho \hat{P}_\psi}{2\pi}\right) \\
  \end{aligned}
  $$

## 变分矩方法

### 1. 参数化

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

### 2. Grad-Shafranov 方程

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
