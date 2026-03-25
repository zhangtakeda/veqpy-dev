## geometry

$\theta$ 是顺时针方向, $\phi$ 是垂直向内的方向, 仅考虑环向对称的平衡求解.

- 磁面坐标: $(\rho, \theta, \phi)$
- 几何坐标: $(R, Z, -\phi)$

$$
\begin{aligned}
\bar{\theta}(\rho, \theta)&=\theta+c_0+\sum_{m = 1}^{M} c_m \cos (m \theta)+\sum_{n = 1}^N s_n \sin (n \theta)\\
R(\rho, \theta) &= R_0+a(h +\rho  \cos \bar{\theta})\\
Z(\rho, \theta) &= Z_0+a(v -\rho  \kappa  \sin \theta)\\
\end{aligned}
$$

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
