## equilibrium

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
