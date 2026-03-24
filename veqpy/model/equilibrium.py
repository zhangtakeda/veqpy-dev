"""model 层的 Equilibrium 定义.

属于 model 层.
负责持有单网格上的平衡快照, 从 root fields 重新派生 geometry 和 diagnostics,
并提供 plotting, comparison, resample 等 inspection 能力.
不负责 solver runtime ownership, packed state, 或 residual hot path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from veqpy.model.geometry import Geometry
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile
from veqpy.model.reactive import Reactive
from veqpy.model.serial import Serial

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
        "lines.markersize": 4,
        "legend.labelspacing": 0.3,
        "legend.columnspacing": 0.6,
    }
)

PROFILE_INFO = {
    "h": (0.0, 0, "#1f77b4", r"$h$"),
    "v": (0.0, 0, "#17becf", r"$v$"),
    "k": ("k_profile", 0, "#ff7f0e", r"$\kappa$"),
    "c0": ("c0_profile", 0, "#8c564b", r"$c_0$"),
    "c1": ("c1_profile", 1, "#e377c2", r"$c_1$"),
    "s1": ("s1_profile", 1, "#2ca02c", r"$s_1$"),
    "s2": ("s2_profile", 2, "#d62728", r"$s_2$"),
}

BLACK = "black"
BLUE = mcolors.TABLEAU_COLORS["tab:blue"]
ORANGE = mcolors.TABLEAU_COLORS["tab:orange"]
GREEN = mcolors.TABLEAU_COLORS["tab:green"]
RED = mcolors.TABLEAU_COLORS["tab:red"]
PURPLE = mcolors.TABLEAU_COLORS["tab:purple"]


MU0 = 4e-7 * np.pi


class Equilibrium(Reactive, Serial):
    """磁流体平衡态对象.

    Equilibrium 在当前架构中表示单网格上的 materialized diagnostic snapshot.

    它持有已经在当前 grid 上物化好的 root fields:
    - 7 个一维 shape/profile buffer
    - psin_r, psin_rr, FFn_r, Pn_r
    - alpha1, alpha2

    它不是参数化 profile spec 的容器, 也不承诺保留 solver 侧参数化语义.
    resample(...) 的语义是把这些 root snapshot fields 插值到目标 grid,
    然后在目标 grid 上重新派生 geometry 和 diagnostics.

    在当前架构中它属于 model/control plane, 用于 inspection/plotting.
    """

    def __init__(
        self,
        R0: float,
        Z0: float,
        B0: float,
        a: float,
        grid: Grid,
        active_profiles: list[str] | None = None,
        *,
        h_profile: Profile,
        v_profile: Profile,
        k_profile: Profile,
        c0_profile: Profile,
        c1_profile: Profile,
        s1_profile: Profile,
        s2_profile: Profile,
        FFn_r: np.ndarray,
        Pn_r: np.ndarray,
        psin_r: np.ndarray,
        psin_rr: np.ndarray | None,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
    ):
        """初始化平衡快照对象.

        Args:
            R0, Z0: 几何中心坐标, 单位 m.
            B0: 参考磁场, 单位 T.
            a: 小半径尺度, 单位 m.
            grid: 当前快照所属的径向和极向网格.
            active_profiles: 当前快照中视为激活的 profile 名称列表.
            h_profile, v_profile, k_profile, c0_profile, c1_profile, s1_profile, s2_profile:
                当前 grid 上的 profile 快照对象.
            FFn_r, Pn_r, psin_r, psin_rr:
                当前 grid 上的 root fields, shape=(Nr,).
            alpha1, alpha2:
                当前快照关联的 source 缩放系数.
        """
        super().__init__()

        self.R0 = R0
        self.Z0 = Z0
        self.B0 = B0
        self.a = a
        self.grid = grid
        self.active_profiles = list(PROFILE_INFO) if active_profiles is None else list(active_profiles)

        self.h_profile = h_profile
        self.v_profile = v_profile
        self.k_profile = k_profile
        self.c0_profile = c0_profile
        self.c1_profile = c1_profile
        self.s1_profile = s1_profile
        self.s2_profile = s2_profile

        for profile in (
            self.h_profile,
            self.v_profile,
            self.k_profile,
            self.c0_profile,
            self.c1_profile,
            self.s1_profile,
            self.s2_profile,
        ):
            profile.update(grid=self.grid)

        if psin_rr is None:
            psin_rr = grid.differentiate(psin_r)

        self.FFn_r = FFn_r
        self.Pn_r = Pn_r
        self.psin_r = psin_r
        self.psin_rr = psin_rr
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def __rich__(self):
        tree = Tree("[bold blue]Equilibrium[/]")
        tree.add(self.grid)
        tree.add(Text(f"a: {self.a:.3f} [m]"))
        tree.add(Text(f"R0: {self.R0:.3f} [m]"))
        tree.add(Text(f"Z0: {self.Z0:.3f} [m]"))
        tree.add(f"B0: {self.B0:.3f} [T]")
        tree.add(f"Ip: {float(self.Ip):.3e} [A]")
        tree.add(f"beta_t: {float(self.beta_t):.3e}")
        tree.add(f"alpha1: {self.alpha1:.6f}")
        tree.add(f"alpha2: {self.alpha2:.6f}")
        return tree

    def __str__(self) -> str:
        console = Console(color_system=None, force_terminal=False, width=120, record=True, soft_wrap=False)
        with console.capture() as capture:
            console.print(self.__rich__())
        return capture.get().rstrip()

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """声明可序列化的构造根状态.

        Returns:
            仅包含构造函数持有的 root snapshot fields, 不包含派生 diagnostics.
        """
        return {
            "R0": float,
            "Z0": float,
            "B0": float,
            "a": float,
            "grid": Grid,
            "active_profiles": list[str],
            "h_profile": Profile,
            "v_profile": Profile,
            "k_profile": Profile,
            "c0_profile": Profile,
            "c1_profile": Profile,
            "s1_profile": Profile,
            "s2_profile": Profile,
            "FFn_r": np.ndarray,
            "Pn_r": np.ndarray,
            "psin_r": np.ndarray,
            "psin_rr": np.ndarray,
            "alpha1": float,
            "alpha2": float,
        }

    @property
    def rho(self) -> np.ndarray:
        return self.grid.rho

    @property
    def theta(self) -> np.ndarray:
        return self.grid.theta

    @property
    def cos_theta(self) -> np.ndarray:
        return self.grid.cos_theta

    @property
    def sin_theta(self) -> np.ndarray:
        return self.grid.sin_theta

    @property
    def R(self) -> np.ndarray:
        return self.geometry.R

    @property
    def Z(self) -> np.ndarray:
        return self.geometry.Z

    @property
    def geometry(self) -> Geometry:
        """从当前快照 root fields 重新物化 Geometry."""
        geometry = Geometry(grid=self.grid)
        geometry.update(
            self.a,
            self.R0,
            self.Z0,
            self.grid,
            self.h_profile,
            self.v_profile,
            self.k_profile,
            self.c0_profile,
            self.c1_profile,
            self.s1_profile,
            self.s2_profile,
        )
        return geometry

    @property
    def S(self) -> np.ndarray:
        """磁面面积 S = -int R*Z_t dtheta."""
        R, Z_t = self.geometry.R, self.geometry.Z_t
        return -self.grid.quadrature(R * Z_t, axis=1)

    @property
    def S_r(self) -> np.ndarray:
        """磁面面积微分 S_r = int J dtheta."""
        J = self.geometry.J
        return self.grid.quadrature(J, axis=1)

    @property
    def V(self) -> np.ndarray:
        """磁面体积 V = -pi*int R**2*Z_t dtheta."""
        R, Z_t = self.geometry.R, self.geometry.Z_t
        return -np.pi * self.grid.quadrature(R**2 * Z_t, axis=1)

    @property
    def V_r(self) -> np.ndarray:
        """磁面体积微分 V_r = 2pi * int J*R dtheta."""
        R, J = self.geometry.R, self.geometry.J
        return (2 * np.pi) * self.grid.quadrature(J * R, axis=1)

    @property
    def Kn(self) -> np.ndarray:
        """归一化几何因子 Kn = int gttdivJR dtheta/(2pi)."""
        gttdivJR = self.geometry.gttdivJR
        return self.grid.quadrature(gttdivJR, axis=1) / (2 * np.pi)

    @property
    def Kn_r(self) -> np.ndarray:
        """Kn 的径向导数."""
        gttdivJR_r = self.geometry.gttdivJR_r
        return self.grid.quadrature(gttdivJR_r, axis=1) / (2 * np.pi)

    @property
    def Ln_r(self) -> np.ndarray:
        """归一化几何因子 Ln_r = int JdivR dtheta/(2pi)."""
        JdivR = self.geometry.JdivR
        return self.grid.quadrature(JdivR, axis=1) / (2 * np.pi)

    @property
    def FF_r(self) -> np.ndarray:
        """物理 F*F' 剖面, model-side diagnostic."""
        return self.alpha1 * self.alpha2 * self.FFn_r

    @property
    def F2(self) -> np.ndarray:
        """物理 F^2 剖面."""
        return (self.R0 * self.B0) ** 2 + 2.0 * (self.grid.integrate(self.FF_r) - self.grid.quadrature(self.FF_r))

    @property
    def F(self) -> np.ndarray:
        """极向电流函数 F (R*B_phi)."""
        if np.any(self.F2 < 1e-15):
            raise ValueError("Negative F2 encountered, cannot compute F")
        return np.sqrt(self.F2)

    @property
    def P_r(self) -> np.ndarray:
        """物理压强梯度 P', model-side diagnostic."""
        return self.alpha1 * self.alpha2 * self.Pn_r / MU0

    @property
    def P(self) -> np.ndarray:
        """物理压强剖面 P."""
        return self.grid.integrate(self.P_r) - self.grid.quadrature(self.P_r)

    @property
    def beta_t(self) -> np.ndarray:
        """环向比压 beta_t = 2*mu0*<P> / B0^2."""
        P_avg = float(self.grid.quadrature(self.P * self.V_r) / self.grid.quadrature(self.V_r))
        return float(2.0 * MU0 * P_avg / self.B0**2)

    @property
    def Ip(self) -> np.ndarray:
        """总等离子体电流 Ip (Amps)."""
        R, JdivR = self.geometry.R, self.geometry.JdivR
        psin_r_safe = np.maximum(self.psin_r, 1e-10)
        G1n = JdivR * (self.FFn_r[:, None] + R**2 * self.Pn_r[:, None]) / psin_r_safe[:, None]
        return -self.alpha1 * self.grid.quadrature(G1n) / MU0

    @property
    def q(self) -> np.ndarray:
        """安全因子 q, model-side diagnostic."""
        with np.errstate(divide="ignore", invalid="ignore"):
            q = self.F * self.Ln_r / (self.alpha2 * self.psin_r)

        _extrapolate_inplace(self.rho, q, p=2)
        return q

    @property
    def s(self) -> np.ndarray:
        """磁剪切 s, model-side diagnostic."""
        q_r = self.grid.differentiate(self.q)
        q_safe = np.maximum(np.abs(self.q), 1e-15)
        return self.rho * q_r / q_safe

    @property
    def Itor(self) -> np.ndarray:
        """环向电流分布 I_tor(rho), model-side diagnostic."""
        return 2.0 * np.pi * self.Kn * self.alpha2 * self.psin_r / MU0

    @property
    def jtor(self) -> np.ndarray:
        """环向电流密度 j_phi, model-side diagnostic."""
        with np.errstate(divide="ignore", invalid="ignore"):
            jtor = (
                -self.alpha1
                / (MU0 * self.psin_r * self.S_r)
                * (2.0 * np.pi * self.FFn_r * self.Ln_r + self.V_r * self.Pn_r / (2.0 * np.pi))
            )

        _extrapolate_inplace(self.rho, jtor, p=2)
        return jtor

    @property
    def jpara(self) -> np.ndarray:
        """平行电流密度 <j.B>/B0, model-side diagnostic."""
        F_r = self.grid.differentiate(self.F)
        term_r = (
            self.Kn_r * self.psin_r / self.F + self.Kn * self.psin_rr / self.F - self.Kn * self.psin_r * F_r / self.F**2
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            jpara = self.alpha2 / MU0 * self.F / self.Ln_r * term_r

        _extrapolate_inplace(self.rho, jpara, p=2)
        return jpara

    @property
    def jphi(self) -> np.ndarray:
        """局部环向电流密度 j_phi(R, Z)."""
        R = self.geometry.R
        with np.errstate(divide="ignore", invalid="ignore"):
            jphi = -self.alpha1 / (MU0 * self.psin_r[:, None] * R) * (self.FFn_r[:, None] + R**2 * self.Pn_r[:, None])

        _extrapolate_inplace(self.rho, jphi, p=2)
        return jphi

    @property
    def Psi(self) -> np.ndarray:
        """物理极向磁通 Psi."""
        return 2.0 * np.pi * self.alpha2 * self.grid.integrate(self.psin_r)

    @property
    def Phi(self) -> np.ndarray:
        """环向磁通 Phi."""
        return 2.0 * np.pi * self.grid.integrate(self.F * self.Ln_r)

    def plot(
        self,
        outpath: str | None = None,
        *,
        show: bool = False,
        target_grid: Grid | None = None,
        profile_degree: int | None = None,
        native_grid: bool = False,
    ):
        """Render the legacy 6-panel summary figure for this equilibrium."""

        return plot_equilibrium(
            self,
            outpath=outpath,
            show=show,
            target_grid=target_grid,
            profile_degree=profile_degree,
            native_grid=native_grid,
        )

    def compare(
        self,
        other: "Equilibrium",
        outpath: str | Path | None = None,
        *,
        show: bool = False,
        target_grid: Grid | None = None,
        profile_degree: int | None = None,
        native_grid: bool = False,
        label_ref: str = "reference",
        label_other: str = "current",
    ) -> dict[str, float]:
        """Compare this equilibrium against another one on a shared plotting grid."""

        return plot_comparison(
            self,
            other,
            outpath=outpath,
            show=show,
            target_grid=target_grid,
            profile_degree=profile_degree,
            native_grid=native_grid,
            label_ref=label_ref,
            label_other=label_other,
        )

    def resample(
        self,
        *,
        target_grid: Grid | None = None,
        profile_degree: int | None = None,
        native_grid: bool = False,
    ) -> "Equilibrium":
        """将当前平衡快照插值到目标网格.

        Args:
            target_grid: 目标网格. 为 None 时使用当前 grid.
            profile_degree: 可选的 profile 重建阶数.
            native_grid: 为 True 时优先保留原始 profile 样本点语义.

        Returns:
            目标 grid 上重新物化的 Equilibrium 快照.
        """

        return resample_equilibrium(
            self,
            target_grid=target_grid,
            profile_degree=profile_degree,
            native_grid=native_grid,
        )


def _extrapolate_inplace(
    r: np.ndarray,
    y: np.ndarray,
    *,
    p=2,
) -> None:
    if r[0] < 1e-4 and r.size >= 3:
        r1, r2 = r[1] ** p, r[2] ** p
        y[0] = (y[1] * r2 - y[2] * r1) / (r2 - r1)


def plot_equilibrium(
    equilibrium: Equilibrium,
    outpath: str | Path | None = None,
    *,
    show: bool = False,
    target_grid: Grid | None = None,
    profile_degree: int | None = None,
    native_grid: bool = False,
):
    """Render the legacy 6-panel equilibrium summary for one model-side equilibrium."""

    surface_eq = resample_equilibrium(
        equilibrium,
        target_grid=target_grid,
        profile_degree=profile_degree,
        native_grid=native_grid,
    )
    fig = _render_equilibrium_summary(surface_equilibrium=surface_eq, profile_equilibrium=equilibrium)

    if outpath is not None:
        fig.savefig(Path(outpath), dpi=300, facecolor="white")
    if show:
        plt.show()
    elif outpath is not None:
        plt.close(fig)

    return fig


def plot_comparison(
    reference: Equilibrium,
    other: Equilibrium,
    outpath: str | Path | None = None,
    *,
    show: bool = False,
    target_grid: Grid | None = None,
    profile_degree: int | None = None,
    native_grid: bool = False,
    label_ref: str = "reference",
    label_other: str = "current",
) -> dict[str, float]:
    """Render a veqpy-poor-style 3x3 comparison figure."""

    if target_grid is None and not native_grid:
        ref_plot = reference
        other_plot = other
    else:
        ref_plot = resample_equilibrium(
            reference,
            target_grid=target_grid,
            profile_degree=profile_degree,
            native_grid=native_grid,
        )
        other_plot = resample_equilibrium(
            other,
            target_grid=target_grid,
            profile_degree=profile_degree,
            native_grid=native_grid,
        )

    shape_keys = ["h", "k", "s1"]
    groups = [(key, f"{PROFILE_INFO[key][3]}", None) for key in shape_keys]
    groups.extend(
        [
            ("psi_r", r"$\psi_\rho$", None),
            ("FF_r", r"$FF_\rho$", None),
            ("P_r", r"$P_\rho$", None),
            ("Itor", r"$I_{\rm tor}$ [MA]", 1e6),
            ("jtor", r"$j_{\rm tor}$ [MA/m²]", 1e6),
            ("jpara", r"$j_{\|}$ [MA/m²]", 1e6),
        ]
    )
    while len(groups) < 9:
        groups.append(("", "", None))

    def _extract(eq: Equilibrium) -> dict[str, np.ndarray]:
        data = {
            "rho": np.asarray(eq.rho, dtype=np.float64),
            "psi_r": np.asarray(eq.alpha2 * eq.psin_r, dtype=np.float64),
            "FF_r": np.asarray(eq.FF_r, dtype=np.float64),
            "P_r": np.asarray(eq.P_r, dtype=np.float64),
            "Itor": np.asarray(eq.Itor, dtype=np.float64),
            "jtor": np.asarray(eq.jtor, dtype=np.float64),
            "jpara": np.asarray(eq.jpara, dtype=np.float64),
        }
        for key in shape_keys:
            data[key] = np.asarray(getattr(eq, f"{key}_profile").u, dtype=np.float64)
        return data

    d1, d2 = _extract(ref_plot), _extract(other_plot)

    def _aligned_reference(key: str) -> tuple[np.ndarray, np.ndarray]:
        rho1 = np.asarray(d1["rho"], dtype=np.float64)
        rho2 = np.asarray(d2["rho"], dtype=np.float64)
        y1 = np.asarray(d1[key], dtype=np.float64)
        y2 = np.asarray(d2[key], dtype=np.float64)
        if rho1.shape == rho2.shape and np.allclose(rho1, rho2):
            return y1, y2
        return np.interp(rho2, rho1, y1), y2

    errors: dict[str, float] = {}
    fig, axes = plt.subplots(3, 3, figsize=(12, 7.5), sharex=True)

    for i, (ax, (key, ylabel, scale)) in enumerate(zip(axes.flatten(), groups, strict=True)):
        if not key:
            ax.set_visible(False)
            continue

        s = scale or 1.0
        ref_values, cur_values = _aligned_reference(key)
        scale_ref = float(np.max(np.abs(ref_values))) or 1.0
        diff = cur_values - ref_values
        errors[f"rel_{key}_max"] = float(np.max(np.abs(diff)) / scale_ref)
        errors[f"rel_{key}_rms"] = float(np.sqrt(np.mean(diff**2)) / scale_ref)

        ax.plot(d1["rho"], d1[key] / s, color=BLACK, linestyle="-", label=label_ref)
        ax.plot(
            d2["rho"],
            d2[key] / s,
            color=RED,
            linestyle="--",
            marker="o",
            markersize=5,
            zorder=5,
            label=label_other,
        )
        ax.set_title(f"err = {errors[f'rel_{key}_max']:.1e}", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(direction="in", top=True, right=True, labelsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)

        if i == 0:
            ax.legend(loc="best", frameon=False, fontsize=9)
        if i >= 6:
            ax.set_xlabel(r"$\rho$", fontsize=11)

    fig.suptitle(
        rf"Comparison: $\alpha_1={other.alpha1:.4e}$, $\alpha_2={other.alpha2:.4e}$  [{label_ref} vs {label_other}]",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    if outpath is not None:
        fig.savefig(Path(outpath), dpi=300, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return errors


def resample_equilibrium(
    equilibrium: Equilibrium,
    *,
    target_grid: Grid | None = None,
    profile_degree: int | None = None,
    native_grid: bool = False,
) -> Equilibrium:
    """Resample one equilibrium snapshot onto another grid.

    Shape profiles are re-materialized on the target grid from their
    parameterized profile state. Source/root 1D fields are interpolated
    onto the target grid, then geometry and diagnostics are re-derived.
    """
    return _build_plot_equilibrium(
        equilibrium,
        target_grid=target_grid,
        profile_degree=profile_degree,
        native_grid=native_grid,
    )


def _build_plot_equilibrium(
    equilibrium: Equilibrium,
    *,
    target_grid: Grid | None,
    profile_degree: int | None,
    native_grid: bool,
) -> Equilibrium:
    if native_grid:
        return equilibrium

    source_grid = equilibrium.grid
    plot_grid = target_grid or Grid(Nr=64, Nt=32, scheme="uniform", L_max=source_grid.L_max)
    degree = min(source_grid.Nr - 1, 16) if profile_degree is None else int(profile_degree)

    def resample_vector(y_src: np.ndarray) -> np.ndarray:
        return _resample_profile(
            source_grid.rho, np.asarray(y_src, dtype=np.float64), plot_grid.rho, degree=degree, strict=True
        )

    def _resample_profile_triplet(profile: Profile) -> Profile:
        out = profile.copy()
        out.update(grid=plot_grid)
        return out

    psin_r = _resample_profile(
        source_grid.rho,
        np.asarray(equilibrium.psin_r, dtype=np.float64),
        plot_grid.rho,
        left=0.0,
        degree=degree,
        strict=True,
    )
    FFn_r = _resample_profile(
        source_grid.rho,
        np.asarray(equilibrium.FFn_r, dtype=np.float64),
        plot_grid.rho,
        left=0.0,
        right=0.0,
        degree=degree,
        strict=True,
    )
    Pn_r = _resample_profile(
        source_grid.rho,
        np.asarray(equilibrium.Pn_r, dtype=np.float64),
        plot_grid.rho,
        left=0.0,
        right=0.0,
        degree=degree,
        strict=True,
    )

    return Equilibrium(
        R0=equilibrium.R0,
        Z0=equilibrium.Z0,
        B0=equilibrium.B0,
        a=equilibrium.a,
        grid=plot_grid,
        active_profiles=equilibrium.active_profiles,
        h_profile=_resample_profile_triplet(equilibrium.h_profile),
        v_profile=_resample_profile_triplet(equilibrium.v_profile),
        k_profile=_resample_profile_triplet(equilibrium.k_profile),
        c0_profile=_resample_profile_triplet(equilibrium.c0_profile),
        c1_profile=_resample_profile_triplet(equilibrium.c1_profile),
        s1_profile=_resample_profile_triplet(equilibrium.s1_profile),
        s2_profile=_resample_profile_triplet(equilibrium.s2_profile),
        FFn_r=FFn_r,
        Pn_r=Pn_r,
        psin_r=psin_r,
        psin_rr=plot_grid.differentiate(psin_r),
        alpha1=equilibrium.alpha1,
        alpha2=equilibrium.alpha2,
    )


def _render_equilibrium_summary(*, surface_equilibrium: Equilibrium, profile_equilibrium: Equilibrium):
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(
        2,
        7,
        figure=fig,
        width_ratios=[1, 0.35, 1.3, 0.2, 1, 0.4, 1.3],
        height_ratios=[1, 1],
        hspace=0.5,
        wspace=0.0,
        top=0.95,
        bottom=0.1,
        left=0.05,
        right=0.98,
    )

    panel_a = _build_surface_panel_data(surface_equilibrium)
    panel_b = _build_shape_panel_data(profile_equilibrium)
    panel_c = _build_source_panel_data(profile_equilibrium)
    panel_d = _build_jphi_panel_data(surface_equilibrium)
    panel_e = _build_current_panel_data(profile_equilibrium)
    panel_f = _build_safety_panel_data(profile_equilibrium)

    _render_panel_a_surfaces(fig.add_subplot(gs[:, 0]), fig, panel_a)
    _render_panel_b_shapes(fig.add_subplot(gs[0, 2]), panel_b)
    _render_panel_c_sources(fig.add_subplot(gs[1, 2]), panel_c)
    _render_panel_d_jphi(fig.add_subplot(gs[:, 4]), fig, panel_d, panel_a["boundary"])
    _render_panel_e_current_1d(fig.add_subplot(gs[0, 6]), panel_e)
    _render_panel_f_safety(fig.add_subplot(gs[1, 6]), panel_f)
    return fig


def _build_surface_panel_data(equilibrium: Equilibrium) -> dict:
    R = equilibrium.geometry.R
    Z = equilibrium.geometry.Z
    rho = equilibrium.rho
    Nt = equilibrium.grid.Nt

    sample_rho = np.linspace(0.0, 1.0, 12)
    surfaces = []
    for rho_value in sample_rho:
        idx = int(np.argmin(np.abs(rho - rho_value)))
        if rho[idx] <= 0.0:
            continue
        surfaces.append(
            {
                "rho": float(rho[idx]),
                "R": _close_periodic_curve(R[idx, :]),
                "Z": _close_periodic_curve(Z[idx, :]),
            }
        )

    theta_count = min(max(Nt, 1), 16)
    theta_indices = np.unique(np.linspace(0, Nt - 1, theta_count, dtype=int))
    rays = []
    for theta_idx in theta_indices:
        rays.append(
            {
                "theta_index": int(theta_idx),
                "R": np.asarray(R[:, theta_idx], dtype=np.float64),
                "Z": np.asarray(Z[:, theta_idx], dtype=np.float64),
            }
        )

    return {
        "surfaces": surfaces,
        "rays": rays,
        "axis": {"R": float(R[0, 0]), "Z": float(Z[0, 0])},
        "center": {"R": float(equilibrium.R0), "Z": float(equilibrium.Z0)},
        "boundary": {"R": _close_periodic_curve(R[-1, :]), "Z": _close_periodic_curve(Z[-1, :])},
    }


def _build_shape_panel_data(equilibrium: Equilibrium) -> dict:
    profiles = dict(
        h=equilibrium.h_profile,
        v=equilibrium.v_profile,
        k=equilibrium.k_profile,
        c0=equilibrium.c0_profile,
        c1=equilibrium.c1_profile,
        s1=equilibrium.s1_profile,
        s2=equilibrium.s2_profile,
    )

    values: dict[str, np.ndarray] = {}
    rho = equilibrium.rho
    active_set = set(equilibrium.active_profiles)
    for key in PROFILE_INFO:
        if key in active_set:
            values[key] = profiles[key].u
    return {"shape": {"rho": rho, "values": values}}


def _build_source_panel_data(equilibrium: Equilibrium) -> dict:
    return {
        "rho": equilibrium.rho,
        "psi_r": equilibrium.alpha2 * equilibrium.psin_r.copy(),
        "FF_r": equilibrium.FF_r.copy(),
        "mu0_P_r": MU0 * equilibrium.P_r.copy(),
    }


def _build_jphi_panel_data(equilibrium: Equilibrium) -> dict:
    return {
        "R": np.hstack([equilibrium.geometry.R, equilibrium.geometry.R[:, :1]]),
        "Z": np.hstack([equilibrium.geometry.Z, equilibrium.geometry.Z[:, :1]]),
        "jphi": np.hstack([equilibrium.jphi, equilibrium.jphi[:, :1]]) / 1e6,
    }


def _build_current_panel_data(equilibrium: Equilibrium) -> dict:
    return {
        "rho": equilibrium.rho,
        "itor": equilibrium.Itor.copy() / 1e6,
        "jtor": equilibrium.jtor.copy() / 1e6,
        "jpara": equilibrium.jpara.copy() / 1e6,
        "Ip": float(equilibrium.Ip) / 1e6,
    }


def _build_safety_panel_data(equilibrium: Equilibrium) -> dict:
    return {"rho": equilibrium.rho, "q": equilibrium.q.copy(), "s": equilibrium.s.copy()}


def _comparison_profile_values(equilibrium: Equilibrium, key: str) -> np.ndarray:
    if key == "psi_r":
        return equilibrium.alpha2 * equilibrium.psin_r.copy()
    if key == "FF_r":
        return equilibrium.FF_r.copy()
    if key == "P_r":
        return equilibrium.P_r.copy()
    if key == "Itor":
        return equilibrium.Itor.copy()
    if key == "jtor":
        return equilibrium.jtor.copy()
    if key == "jpara":
        return equilibrium.jpara.copy()
    raise KeyError(f"Unsupported comparison profile {key!r}")


def _active_shape_keys(reference: Equilibrium, other: Equilibrium) -> list[str]:
    active_set = set(reference.active_profiles) | set(other.active_profiles)
    return [key for key in PROFILE_INFO if key in active_set]


def _apply_rz_limits(ax: plt.Axes, boundary_data: dict):
    R_bnd, Z_bnd = boundary_data["R"], boundary_data["Z"]
    R_margin = (R_bnd.max() - R_bnd.min()) * 0.1
    Z_margin = (Z_bnd.max() - Z_bnd.min()) * 0.1
    ax.set_xlim(R_bnd.min() - R_margin, R_bnd.max() + R_margin)
    ax.set_ylim(Z_bnd.min() - Z_margin, Z_bnd.max() + Z_margin)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R$ [m]")
    ax.set_ylabel(r"$Z$ [m]")


def _get_trunc_inferno() -> mcolors.LinearSegmentedColormap:
    cmap = plt.get_cmap("inferno")
    return mcolors.LinearSegmentedColormap.from_list("trunc_inferno", cmap(np.linspace(0.15, 0.92, 256)))


def _add_top_headroom(ax: plt.Axes, ratio: float) -> None:
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    if ratio > 0.0:
        ax.set_ylim(y0, y1 + ratio * span)
    else:
        ax.set_ylim(y0 + ratio * span, y1)


def _render_panel_a_surfaces(ax: plt.Axes, fig: plt.Figure, data: dict):
    ax.set_title("(a) Flux Surfaces")
    for ray in data.get("rays", []):
        ax.plot(ray["R"], ray["Z"], color="#9aa0a6", linewidth=0.8, alpha=0.55, zorder=1)

    colors = plt.cm.inferno(np.linspace(0.0, 1.0, max(len(data["surfaces"]), 1)) * 0.77 + 0.15)
    for ci, surf in enumerate(data["surfaces"]):
        ax.plot(surf["R"], surf["Z"], color=colors[min(ci, len(colors) - 1)], zorder=2)

    ax.plot(
        data["axis"]["R"],
        data["axis"]["Z"],
        "ko",
        markersize=3,
        mew=2,
        label=f"Axis ({data['axis']['R']:.2f}, {data['axis']['Z']:.2f})",
    )
    ax.legend(loc="upper right")
    _apply_rz_limits(ax, data["boundary"])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=_get_trunc_inferno(), norm=mcolors.Normalize(vmin=0.0, vmax=1.0))
    fig.colorbar(sm, cax=cax).set_label(r"$\rho$")


def _render_panel_b_shapes(ax: plt.Axes, data: dict):
    ax.set_title("(b) Shape Profiles")
    shape = data["shape"]
    for key, vals in shape["values"].items():
        _, _, color, label = PROFILE_INFO[key]
        ax.plot(shape["rho"], vals, "-", color=color, label=label)

    ax.set(xlabel=r"$\rho$", ylabel="Profile")
    if shape["values"]:
        ax.legend(loc="center left")
    ax.grid(True)


def _render_panel_c_sources(ax: plt.Axes, data: dict):
    ax.set_title("(c) Source Profiles")
    rho = data["rho"]
    if len(rho) < 32:
        ax.plot(rho, data["psi_r"], "--o", color=BLACK, label=r"$\psi_\rho$")
        ax.plot(rho, data["FF_r"], "-o", color=RED, label=r"$FF_\rho$")
        ax.plot(rho, data["mu0_P_r"], "-o", color=PURPLE, label=r"$\mu_0 P_\rho$")
    else:
        ax.plot(rho, data["psi_r"], "--", color=BLACK, label=r"$\psi_\rho$")
        ax.plot(rho, data["FF_r"], "-", color=RED, label=r"$FF_\rho$")
        ax.plot(rho, data["mu0_P_r"], "-", color=PURPLE, label=r"$\mu_0 P_\rho$")

    ax.set(xlabel=r"$\rho$", ylabel="Profile")
    _add_top_headroom(ax, ratio=-0.15)
    ax.legend(loc="lower left")
    ax.grid(True)


def _render_panel_d_jphi(ax: plt.Axes, fig: plt.Figure, data: dict, boundary: dict):
    ax.set_title("(d) Current Density")
    R_plot, Z_plot, j_plot = data["R"], data["Z"], data["jphi"]
    cmap = _get_trunc_inferno()
    vmin = min(float(np.nanmin(j_plot)), 0.0)
    vmax = float(np.nanmax(j_plot))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    ax.set_facecolor(cmap(norm(0.0)))
    pcm = ax.contourf(R_plot, Z_plot, j_plot, levels=np.linspace(vmin, vmax, 128), cmap=cmap, norm=norm)
    _apply_rz_limits(ax, boundary)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(pcm, cax=cax).set_label(r"$j_\phi$ [MA/m$^2$]")


def _render_panel_e_current_1d(ax: plt.Axes, data: dict):
    ax.set_title("(e) Current Profiles")
    rho = data["rho"]
    ax.axhline(data["Ip"], xmin=0.75, xmax=1.0, color=BLACK, linestyle="--", label=r"$I_p$")
    if len(rho) < 32:
        ax.plot(rho, data["itor"], "-o", color=BLUE, label=r"$I_{tor}$")
        ax.plot(rho, data["jtor"], "-o", color=ORANGE, label=r"$j_{tor}$")
        ax.plot(rho, data["jpara"], "-o", color=GREEN, label=r"$j_\parallel$")
    else:
        ax.plot(rho, data["itor"], "-", color=BLUE, label=r"$I_{tor}$")
        ax.plot(rho, data["jtor"], "-", color=ORANGE, label=r"$j_{tor}$")
        ax.plot(rho, data["jpara"], "-", color=GREEN, label=r"$j_\parallel$")

    ax.set(xlabel=r"$\rho$", ylabel="Current [MA]")
    _add_top_headroom(ax, ratio=0.2)
    ax.legend(loc="upper left", ncols=2)
    ax.grid(True)


def _render_panel_f_safety(ax: plt.Axes, data: dict):
    ax.set_title("(f) Safety Factor")
    rho = data["rho"]
    if len(rho) < 32:
        ax.plot(data["rho"], data["q"], "-o", color=BLUE, label=r"$q$")
        ax.plot(data["rho"], data["s"], "-o", color=ORANGE, label=r"$s$")
    else:
        ax.plot(data["rho"], data["q"], "-", color=BLUE, label=r"$q$")
        ax.plot(data["rho"], data["s"], "-", color=ORANGE, label=r"$s$")
    ax.set(xlabel=r"$\rho$", ylabel="Profile")
    ax.legend(loc="upper left")
    ax.grid(True)


def _close_periodic_curve(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D periodic curve, got {arr.shape}")
    return np.concatenate([arr, arr[:1]])


def _resample_profile(
    rho_src: np.ndarray,
    y_src: np.ndarray,
    rho_eval: np.ndarray,
    *,
    left: float | None = None,
    right: float | None = None,
    degree: int | None = None,
    strict: bool = False,
    atol: float = 1e-4,
    even: bool = False,
) -> np.ndarray:
    """Resample one profile with a constrained Legendre fit for plotting."""

    rho_src, y_src, rho_eval = map(lambda x: np.asarray(x, dtype=np.float64), (rho_src, y_src, rho_eval))
    if rho_src.ndim != 1 or y_src.ndim != 1 or len(rho_src) != len(y_src):
        raise ValueError("rho_src and y_src must be 1D arrays with equal length")

    r_src = rho_src**2 if even else rho_src
    r_eval = rho_eval**2 if even else rho_eval

    if left is not None and right is not None:
        base_src = left * (1.0 - r_src) + right * r_src
        base_eval = left * (1.0 - r_eval) + right * r_eval
        factor_src, factor_eval = r_src * (1.0 - r_src), r_eval * (1.0 - r_eval)
    elif left is not None:
        base_src, base_eval = np.full_like(r_src, left), np.full_like(r_eval, left)
        factor_src, factor_eval = r_src, r_eval
    elif right is not None:
        base_src, base_eval = np.full_like(r_src, right), np.full_like(r_eval, right)
        factor_src, factor_eval = 1.0 - r_src, 1.0 - r_eval
    else:
        base_src, base_eval = np.zeros_like(r_src), np.zeros_like(r_eval)
        factor_src, factor_eval = np.ones_like(r_src), np.ones_like(r_eval)

    q_src = np.zeros_like(y_src)
    mask = np.abs(factor_src) > 1e-14
    if np.any(mask):
        q_src[mask] = (y_src[mask] - base_src[mask]) / factor_src[mask]

    m = int(np.count_nonzero(mask))
    if m == 0:
        y_eval = base_eval
    else:
        x_src, x_eval = 2.0 * rho_src[mask] - 1.0, 2.0 * rho_eval - 1.0
        deg = (m - 1) if degree is None else int(max(0, min(degree, m - 1)))
        V_src = np.polynomial.legendre.legvander(x_src, deg)

        if deg == m - 1:
            coeffs = np.linalg.solve(V_src, q_src[mask])
        else:
            coeffs, *_ = np.linalg.lstsq(V_src, q_src[mask], rcond=None)

        q_eval = np.polynomial.legendre.legval(x_eval, coeffs)
        y_eval = base_eval + factor_eval * q_eval

    if strict and degree is None:
        y_back = _resample_profile(rho_src, y_src, rho_src, left=left, right=right, degree=None, strict=False)
        err = float(np.max(np.abs(y_back - y_src)))
        if err > atol:
            raise RuntimeError(f"Legendre resample lost nodal interpolation: max err={err:.3e}")

    return y_eval
