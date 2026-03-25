## source

### 1. 总电流约束

该约束源于等离子体总电流 $I_p$ 的定义. 强形式的环向电流密度 $j_{\phi}$:

$$
j_{\phi}(\rho,\theta)=- \left(\frac{F F_\psi}{\mu_0 R}+R P_\psi\right) =-\frac{\alpha_1\hat{G}_1}{\mu_0J}
$$

考虑到 $I_p$ 是 $j_\phi$ 的积分, 对极向截面积分即得到 $\alpha_1$ 代表的电流密度源项缩放方程:

$$
\color{red}
\alpha_1=-\frac{\mu_0 I_p}{\displaystyle\int_0^1 \mathrm{~d} \rho \int_0^{2 \pi} \mathrm{~d} \theta \hat{G}_1}
$$

或者:

$$
\color{red}
\begin{aligned}
\alpha_2&=\frac{\mu_0 I_p}{2\pi \hat{K}(1)\hat{\psi}_\rho(1)}\\
\end{aligned}
$$

### 2. 比压约束

环向比压定义为:

$$
\beta_t=\frac{\langle P\rangle_V}{B_0^2 / 2 \mu_0}=\frac{2 \mu_0\langle P\rangle_V}{B_0^2}
$$

考虑到:

$$
\begin{aligned}
\langle P\rangle_V = \dfrac{\alpha_1\alpha_2}{\mu_0}\dfrac{\displaystyle \int_0^1 \hat{P} V_\rho ~ \mathrm{d} \rho  }{\displaystyle \int_0^1   V_\rho ~ \mathrm{d} \rho},\quad \hat{P} =\int_1^\rho \hat{P}_\rho \mathrm{d} \rho = \int_1^\rho \hat{P}_\psi \hat{\psi}_\rho \mathrm{d} \rho
\end{aligned}
$$

因此:

$$
\color{red}
\begin{aligned}
\alpha_1\alpha_2 = \dfrac{\beta_t B_0^2}{2}\dfrac{\displaystyle \int_0^1  V_\rho ~ \mathrm{d} \rho  }{\displaystyle \int_0^1 \hat{P}  V_\rho ~ \mathrm{d} \rho}
\end{aligned}
$$

### 3. 归一化约束

约定 $\hat{\psi}(1) = 1$, 从而将全部的幅值吸收到 $\alpha_2$ 中:

$$
\color{red}
\begin{aligned}
\alpha_2 &= \int_0^1 \frac{\mu_0 I_{\rm tor}}{2\pi \hat{K}} \mathrm{d}\rho
\end{aligned}
$$

### 4. 正则性条件

1. 单调性, 磁轴是唯一的 O 点, 除磁轴外不存在极值点:

   $$
   \hat{\psi}_\rho > 0
   $$

2. 磁场必须是实数:

   $$
   \begin{aligned}
   F^2 &= R_0^2B_0^2+ 2\alpha_1\alpha_2\int_1^\rho \hat{FF_\rho} ~\mathrm{d}\rho > 0\\
   F^2 &= R_0^2B_0^2+ 2\alpha_1\alpha_2\int_1^\rho \hat{FF_\psi} \cdot \hat{\psi}_\rho ~\mathrm{d}\rho   > 0
   \end{aligned}
   $$

3. 磁面不交叉, 几何到磁面坐标应当保证拓扑同胚性 (磁轴除外):
   $$
   J > 0
   $$

## 输入模式

一维 GS 方程为:

$$
F F_\rho L_\rho+\mu_0^2 \psi_\rho\left(K_\rho \psi_\rho+K \psi_{\rho \rho}\right)+\frac{\mu_0^2}{4 \pi^2} V_\rho P_\rho=0
$$

$$
\left(\hat{K} \hat{\psi}_\rho\right)_{\rho}+\frac{\alpha_1}{\alpha_2} \frac{1}{\hat{\psi}_\rho}  \left(\hat{F F_\rho} \hat{L}_\rho+\frac{V_\rho \hat{P}_\rho}{4 \pi^2}\right)=0
$$

$$
\left(\hat{K} \hat{\psi}_\rho\right)_{\rho}+\frac{\alpha_1}{\alpha_2} \left(\hat{F F_\psi} \hat{L}_\rho+\frac{V_\rho \hat{P}_\psi}{4 \pi^2}\right)=0
$$

**Robust 模式**用于不方便计算出 $\hat{\psi}_\rho$ 或者与 $F$ 强耦合无法分离的情况.

**Strict 模式**的计算基础是显式定义出 $\hat{\psi}_\rho$ 以及 $\hat{FF_\rho}, \hat{P}_\rho$

- 所有模式都支持和 $\rho$ 导数同构的 $\color{red} \psi$ 导数输入组合;
- 如果无特殊说明, 使用如下公式获得归一化剖面:

  $$
  \begin{aligned}
  \hat{\psi}_\rho &= \dfrac{\psi_\rho}{\alpha_2}  \\
  \hat{FF_\rho} & =\dfrac{F F_\rho}{\alpha_1 \alpha_2}= \color{red}\frac{FF_\psi\hat{\psi}_\rho}{\alpha_1}  \\
  \hat{P}_\rho & =\frac{\mu_0P_\rho }{\alpha_1 \alpha_2} = \color{red}\frac{\mu_0 P_\psi\hat{\psi}_\rho}{\alpha_1}  \\
  \end{aligned}
  $$

### 1. 环向场函数 (PF-Strict)

- $\hat{P}_\rho+\hat{FF_\rho} +I_p$
- $\hat{P}_\rho+\hat{FF_\rho}+\beta_t$
- ${P}_\rho+{FF_\rho}$

**PF** 支持使用总电流进行约束:

$$
\begin{aligned}
\alpha_1&=-\frac{\mu_0 I_p}{\displaystyle\int_0^1 \mathrm{~d} \rho \int_0^{2 \pi} \mathrm{~d} \theta \hat{G}_1}\\
\end{aligned}
$$

或者支持比压约束:

$$
\begin{aligned}
\alpha_1 &= \dfrac{\beta_t B_0^2}{2\alpha_2}\dfrac{\displaystyle \int_0^1  V_\rho ~ \mathrm{d} \rho  }{\displaystyle \int_0^1 \hat{P}  V_\rho ~ \mathrm{d} \rho},\quad \hat{P} = \int_1^\rho \hat{P}_\rho \mathrm{d} \rho=
\color{red}\int_1^\rho \hat{P}_\psi \hat{\psi}_\rho \mathrm{d} \rho\\
\end{aligned}
$$

选择一种约束后, 与下面方程联立即可获得 $\alpha_1$ 和 $\alpha_2$ 以及归一化剖面:

$$
\begin{aligned}
\alpha_2 &= \alpha_1 \left(\int_0^1 \hat{X}(\rho) \mathrm{d}\rho\right)^2,\quad
\hat{X}(\rho) = \frac{1}{\hat{K}} \sqrt{ -2 \int_0^\rho \left[ \hat{K}  \left( \hat{FF_\rho} \hat{L}_\rho+ \frac{V_\rho \hat{P_\rho}}{4 \pi^2}  \right) \right] \mathrm{d}\rho }\\
\color{red}\alpha_2 &\color{red}= \alpha_1\int_0^1 \hat{Y}(\rho) \mathrm{d}\rho,\quad \color{red}\hat{Y}(\rho)  \color{red}=-\frac{1}{\hat{K}} \int_0^\rho\left({ \hat{FF_\psi}}  \hat{L}_\rho+\frac{V_\rho {\hat{P}_\psi}}{4 \pi^2}\right) \mathrm{d} \rho\\
\hat{\psi}_\rho  &=\sqrt{\frac{\alpha_1}{\alpha_2}} \hat{X}(\rho)\\
\color{red}\hat{\psi}_\rho &\color{red}={\frac{\alpha_1}{\alpha_2}} \hat{Y}(\rho)
\end{aligned}
$$

或者也可以无任何限制地直接输入物理剖面 (输入为 ${FF_\rho}, {P}_\rho$):

$$
\begin{aligned}
&  \psi_\rho= \frac{1}{\hat{K}} \sqrt{ -2 \int_0^\rho \left[ \hat{K}  \left(  {FF_\rho} \hat{L}_\rho+ \frac{V_\rho  {P_\rho}}{4 \pi^2}  \right) \right] \mathrm{d}\rho }\\
& \color{red} \psi_\rho  \color{red}=-\frac{1}{\hat{K}} \int_0^\rho\left({  {FF_\psi}}  \hat{L}_\rho+\frac{V_\rho { {P}_\psi}}{4 \pi^2}\right) \mathrm{d} \rho\\
& \alpha_2=\int_0^1 \psi_\rho \mathrm{d} \rho\\
& \alpha_1=-\frac{\mu_0}{\alpha_2} \int_0^1 P_\rho \mathrm{d} \rho=\color{red}-\frac{\mu_0}{\alpha_2} \int_0^1 P_\psi  {\psi}_\rho \mathrm{d} \rho\\
\end{aligned}
$$

### 2. 极向磁通 (PP-Strict)

- $(\hat{P}_\rho+\beta_t)+(\hat{\psi}_\rho +I_p)$
- ${P}_\rho+(\hat{\psi}_\rho+I_p)$
- $(\hat{P}_\rho+\beta_t)+{\psi}_\rho$
- ${P}_\rho+{\psi}_\rho$

**PP** 支持使用总电流和比压约束:

$$
\begin{aligned}
\alpha_2&=\frac{\mu_0 I_p}{2\pi \hat{K}(1)\hat{\psi}_\rho(1)}\\
\alpha_1 &= \dfrac{\beta_t B_0^2}{2\alpha_2}\dfrac{\displaystyle \int_0^1  V_\rho ~ \mathrm{d} \rho  }{\displaystyle \int_0^1 \hat{P}  V_\rho ~ \mathrm{d} \rho},\quad \hat{P} = \int_1^\rho \hat{P}_\rho \mathrm{d} \rho=
\color{red}\int_1^\rho \hat{P}_\psi \hat{\psi}_\rho \mathrm{d} \rho\\
\end{aligned}
$$

或者直接输入对应的物理剖面:

$$
\begin{aligned}
\alpha_2&=\int_0^1\psi_\rho~\mathrm{d}\rho\\
\alpha_1 &= -\dfrac{\mu_0 }{\alpha_2}{ \int_0^1P_\rho ~\mathrm{d}\rho}=
\color{red}-\mu_0 \int_0^1 P_\psi \hat{\psi}_\rho \mathrm{d} \rho\\
\end{aligned}
$$

获得归一化剖面:

$$
\begin{aligned}
\hat{F F_\rho}&=-\frac{1}{\hat{L}_\rho}\left[\frac{\alpha_2}{\alpha_1} \hat{\psi}_\rho\left(\hat{K}_\rho \hat{\psi}_\rho+\hat{K} \hat{\psi}_{\rho \rho}\right)+\frac{V_\rho \hat{P}_\rho}{4 \pi^2}\right]\\
\color{red}{\hat{F F_\psi}}&\color{red}=-\frac{1}{\hat{L}_\rho}\left[\frac{\alpha_2}{\alpha_1}\left(\hat{K}_\rho \hat{\psi}_\rho+\hat{K} \hat{\psi}_{\rho \rho}\right)+\frac{V_\rho {\hat{P}_\psi}}{4 \pi^2}\right]
\end{aligned}
$$

### 3. 环向电流 (PI-Strict)

- $(\hat{P}_\rho+\beta_t)+(\hat{I}_{\rm tor} +I_p)$
- ${P}_\rho+(\hat{I}_{\rm tor}+I_p)$
- $(\hat{P}_\rho+\beta_t)+I_{\rm tor}$
- ${P}_\rho+I_{\rm tor}$

**PI** 支持使用总电流或比压约束, 分别对应 $\hat{I}_{\rm tor}$ 和 $\hat{P}_\rho$:

$$
\begin{aligned}
\alpha_2 &= \int_0^1 \frac{\mu_0 I_{\rm tor}}{2\pi \hat{K}} \mathrm{d}\rho,\quad
I_{\rm tor}=\frac{I_p}{\hat{I}_{\rm tor}(1)}\hat{I}_{\rm tor}\\
\alpha_1 &= \dfrac{\beta_t B_0^2}{2\alpha_2}\dfrac{\displaystyle \int_0^1  V_\rho ~ \mathrm{d} \rho  }{\displaystyle \int_0^1 \hat{P}  V_\rho ~ \mathrm{d} \rho},\quad \hat{P} = \int_1^\rho \hat{P}_\rho \mathrm{d} \rho=
\color{red}\int_1^\rho \hat{P}_\psi \hat{\psi}_\rho \mathrm{d} \rho\\
\end{aligned}
$$

或者直接输入物理剖面:

$$
\begin{aligned}
\alpha_2 &= \int_0^1 \frac{\mu_0 I_{\rm tor}}{2\pi \hat{K}} \mathrm{d}\rho\\
\alpha_1 &= -\dfrac{\mu_0 }{\alpha_2}{ \int_0^1P_\rho ~\mathrm{d}\rho}=
\color{red}-\mu_0 \int_0^1 P_\psi \hat{\psi}_\rho \mathrm{d} \rho\\
\end{aligned}
$$

获得归一化剖面:

$$
\begin{aligned}
\hat{\psi}_\rho &=\frac{\mu_0 I_{\rm tor}}{2\pi \alpha_2\hat{K}}\\[8pt]
\hat{F F_\rho}&=-\frac{1}{\hat{L}_\rho}\left[\frac{\mu_0}{2 \pi \alpha_1} (I_{\text {tor }})_\rho\hat{\psi}_\rho+\frac{V_\rho \hat{P}_\rho}{4 \pi^2}\right]\\
\color{red}{\hat{F F_\psi}}&\color{red}=-\frac{1}{\hat{L}_\rho}\left[\frac{\mu_0}{2 \pi \alpha_1} (I_{\text {tor }})_\rho+\frac{V_\rho {\hat{P}_\psi}}{4 \pi^2}\right]
\end{aligned}
$$

### 4. 环向电流密度 (PJ1-Strict)

- $(\hat{P}_\rho+\beta_t)+(\hat{j}_{\rm tor} +I_p)$
- ${P}_\rho+(\hat{j}_{\rm tor}+I_p)$
- $(\hat{P}_\rho+\beta_t)+{j}_{\rm tor}$
- ${P}_\rho+{j}_{\rm tor}$

**PJ1** 支持使用总电流和比压约束:

$$
\begin{aligned}
\alpha_2 &= \int_0^1 \frac{\mu_0 I_{\rm tor}}{2\pi \hat{K}} \mathrm{d}\rho, \quad
I_{\rm tor} = I_p\frac{\displaystyle\int_0^\rho \hat{j}_{\rm tor} S_\rho~\mathrm{d}\rho}{\displaystyle\int_0^1 \hat{j}_{\rm tor} S_\rho ~\mathrm{d}\rho}\\
\alpha_1 &= \dfrac{\beta_t B_0^2}{2\alpha_2}\dfrac{\displaystyle \int_0^1  V_\rho ~ \mathrm{d} \rho  }{\displaystyle \int_0^1 \hat{P}  V_\rho ~ \mathrm{d} \rho},\quad \hat{P} = \int_1^\rho \hat{P}_\rho \mathrm{d} \rho=
\color{red}\int_1^\rho \hat{P}_\psi \hat{\psi}_\rho \mathrm{d} \rho\\
\end{aligned}
$$

或者直接输入物理剖面:

$$
\begin{aligned}
\alpha_2 &= \int_0^1 \frac{\mu_0 I_{\rm tor}}{2\pi \hat{K}} \mathrm{d}\rho, \quad
I_{\rm tor} = \int_0^\rho  {j}_{\rm tor} S_\rho~\mathrm{d}\rho\\
\alpha_1 &= -\dfrac{\mu_0 }{\alpha_2}{ \int_0^1P_\rho ~\mathrm{d}\rho}=
\color{red}-\mu_0 \int_0^1 P_\psi \hat{\psi}_\rho \mathrm{d} \rho\\
\end{aligned}
$$

获得归一化剖面:

$$
\begin{aligned}
\hat{\psi}_\rho &=\frac{\mu_0 I_{\rm tor}}{2\pi \alpha_2\hat{K}}\\
\hat{F F_\rho}&=-\frac{1}{\hat{L}_\rho}\left[\frac{\mu_0}{2 \pi \alpha_1} j_{\text {tor }}S_\rho\hat{\psi}_\rho+\frac{V_\rho \hat{P}_\rho}{4 \pi^2}\right]\\
\color{red}{\hat{F F_\psi}}&\color{red}=-\frac{1}{\hat{L}_\rho}\left[\frac{\mu_0}{2 \pi \alpha_1} j_{\text {tor }}S_\rho+\frac{V_\rho {\hat{P}_\psi}}{4 \pi^2}\right]
\end{aligned}
$$

### 5. 平行电流密度 (PJ2-Robust)

- $(\hat{P}_\rho+\beta_t)+(\hat{j}_{\|} +I_p)$
- ${P}_\rho+(\hat{j}_{\|}+I_p)$
- $(\hat{P}_\rho+\beta_t)+{j}_{\|}$
- ${P}_\rho+{j}_{\|}$

此时 $F$ 加入参数优化, 支持使用总电流和比压约束:

$$
\begin{aligned}
\alpha_2 &= \int_0^1 \frac{\mu_0 I_{\rm tor}}{2\pi \hat{K}} \mathrm{d}\rho, \quad
I_{\rm tor} = I_p \frac{F(\rho) \displaystyle\int_0^\rho \frac{\hat{L}_\rho \hat{j}_{\|}}{F} \mathrm{d}\rho}{F(1) \displaystyle\int_0^1 \frac{\hat{L}_\rho \hat{j}_{\|}}{F} \mathrm{d}\rho}\\
\alpha_1 &= \dfrac{\beta_t B_0^2}{2\alpha_2}\dfrac{\displaystyle \int_0^1  V_\rho ~ \mathrm{d} \rho  }{\displaystyle \int_0^1 \hat{P}  V_\rho ~ \mathrm{d} \rho},\quad \hat{P} = \int_1^\rho \hat{P}_\rho \mathrm{d} \rho=
\color{red}\int_1^\rho \hat{P}_\psi \hat{\psi}_\rho \mathrm{d} \rho\\
\end{aligned}
$$

或者直接输入物理剖面:

$$
\begin{aligned}
\alpha_2 &= \int_0^1 \frac{\mu_0 I_{\rm tor}}{2\pi \hat{K}} \mathrm{d}\rho, \quad
I_{\rm tor} = 2\pi F(\rho) \int_0^\rho \frac{\hat{L}_\rho j_{\|}}{F} \mathrm{d}\rho\\
\alpha_1 &= -\dfrac{\mu_0 }{\alpha_2}{ \int_0^1P_\rho ~\mathrm{d}\rho}=
\color{red}-\mu_0 \int_0^1 P_\psi \hat{\psi}_\rho \mathrm{d} \rho\\
\end{aligned}
$$

获得归一化剖面:

$$
\begin{aligned}
\hat{\psi}_\rho &=\frac{\mu_0 I_{\rm tor}}{2\pi \alpha_2\hat{K}}\\
\hat{FF_\rho} &= \dfrac{FF_\rho}{\alpha_1\alpha_2}
\end{aligned}
$$

### 6. 安全因子 (PQ-Robust)

- $(\hat{P}_\rho+\beta_t)+(\hat{q} +I_p)$
- ${P}_\rho+(\hat{q} +I_p)$
- $(\hat{P}_\rho+\beta_t)+{q}$
- ${P}_\rho+{q}$

此时 $F$ 加入参数优化, 支持使用总电流和比压约束:

$$
\begin{aligned}
\alpha_2&=\int_0^1 \frac{F \hat{L}_\rho}{q} \mathrm{~d} \rho,\quad q =\hat{q} \frac{2 \pi R_0B_0 }{\mu_0 I_p }\frac{ \hat{K}(1) \hat{L}_\rho(1)}{ \hat{q}(1)} \\
\alpha_1 & =\frac{\beta_t B_0^2}{2 \alpha_2} \frac{\displaystyle\int_0^1 V_\rho \mathrm{d} \rho}{\displaystyle\int_0^1 \hat{P} V_\rho \mathrm{d} \rho}, \quad \hat{P}=\int_1^\rho \hat{P}_\rho \mathrm{d} \rho=\color{red}\int_1^\rho \hat{P}_\psi \hat{\psi}_\rho \mathrm{d} \rho
\end{aligned}
$$

或者直接输入物理剖面:

$$
\begin{aligned}
\alpha_2 & =\int_0^1 \frac{F \hat{L}_\rho}{q} \mathrm{~d} \rho \\
\alpha_1 & =-\frac{\mu_0}{\alpha_2} \int_0^1 P_\rho \mathrm{d} \rho=\color{red}-\mu_0 \int_0^1 P_\psi \hat{\psi}_\rho \mathrm{d} \rho
\end{aligned}
$$

获得归一化剖面:

$$
\begin{aligned}
\hat{\psi}_\rho & =\frac{F \hat{L}_\rho}{\alpha_2 q} \\
\hat{FF_\rho} &= \dfrac{FF_\rho}{\alpha_1\alpha_2}
\end{aligned}
$$
