"""
Module: model.equilibrium

Role:
- 负责持有单网格上的平衡快照.
- 负责从 root fields 重新派生 geometry 与 diagnostics.
- 负责提供 plotting, comparison, resample 等 inspection 能力.

Public API:
- Equilibrium

Notes:
- `Equilibrium` 表示 snapshot, 不是 solver runtime 容器.
- 不负责 packed state ownership, 或 residual hot path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebder, chebval, chebvander
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rich.console import Console
from rich.text import Text
from rich.tree import Tree
from scipy.interpolate import CubicSpline

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
SUBPLOT_TITLE_FONTSIZE = plt.rcParams["axes.titlesize"] + 2

SHAPE_PROFILE_PLOT_META = {
    "h": {"color": "#1f77b4", "label": r"$h$", "linestyle": "-", "marker": None},
    "v": {"color": "#ff7f0e", "label": r"$v$", "linestyle": "-", "marker": None},
    "k": {"color": "#2ca02c", "label": r"$\kappa$", "linestyle": "-", "marker": None},
}
SHAPE_PROFILE_NAMES = tuple(SHAPE_PROFILE_PLOT_META)
_EXTRA_SHAPE_PROFILE_COLORS = (
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

BLACK = "black"
BLUE = mcolors.TABLEAU_COLORS["tab:blue"]
ORANGE = mcolors.TABLEAU_COLORS["tab:orange"]
GREEN = mcolors.TABLEAU_COLORS["tab:green"]
RED = mcolors.TABLEAU_COLORS["tab:red"]
PURPLE = mcolors.TABLEAU_COLORS["tab:purple"]


MU0 = 4e-7 * np.pi


class Equilibrium(Reactive, Serial):
    """单网格上的平衡快照对象."""

    def __init__(
        self,
        R0: float,
        Z0: float,
        B0: float,
        a: float,
        grid: Grid,
        active_profiles: dict[str, Profile],
        FFn_psin: np.ndarray,
        Pn_psin: np.ndarray,
        psin: np.ndarray | None,
        psin_r: np.ndarray,
        psin_rr: np.ndarray | None,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
    ):
        """初始化平衡快照对象."""
        super().__init__()

        self.R0 = R0
        self.Z0 = Z0
        self.B0 = B0
        self.a = a
        self.grid = grid
        self.active_profiles = _normalize_shape_profiles(active_profiles)

        for name, profile in self.active_profiles.items():
            setattr(self, f"{name}_profile", profile)
        self.h_profile = self.active_profiles.get("h", _build_default_shape_profile("h"))
        self.v_profile = self.active_profiles.get("v", _build_default_shape_profile("v"))
        self.k_profile = self.active_profiles.get("k", _build_default_shape_profile("k"))

        for profile in _unique_profiles(
            (*self.active_profiles.values(), self.h_profile, self.v_profile, self.k_profile)
        ):
            profile.update(grid=self.grid)

        if psin is None:
            psin = _recover_psin_from_psin_r(grid, psin_r)

        if psin_rr is None:
            psin_rr = grid.corrected_linear_derivative(psin_r)

        self.psin = psin
        self.FFn_psin = FFn_psin
        self.Pn_psin = Pn_psin
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
        """声明可序列化的构造根状态."""
        attrs: dict[str, type] = {
            "R0": float,
            "Z0": float,
            "B0": float,
            "a": float,
            "grid": Grid,
            "active_profiles": dict[str, Profile],
            "psin": np.ndarray,
            "FFn_psin": np.ndarray,
            "Pn_psin": np.ndarray,
            "psin_r": np.ndarray,
            "psin_rr": np.ndarray,
            "alpha1": float,
            "alpha2": float,
        }
        return attrs

    @property
    def rho(self) -> np.ndarray:
        return self.grid.rho

    @property
    def theta(self) -> np.ndarray:
        return self.grid.theta

    @property
    def cos_theta(self) -> np.ndarray:
        return self.grid.cos_ktheta[1]

    @property
    def sin_theta(self) -> np.ndarray:
        return self.grid.sin_ktheta[1]

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
        c_fields = np.zeros((self.grid.M_max + 1, 3, self.grid.Nr), dtype=np.float64)
        s_fields = np.zeros((self.grid.M_max + 1, 3, self.grid.Nr), dtype=np.float64)
        c_active_order = 0
        s_active_order = 0
        for name, profile in _shape_profiles(self).items():
            if name.startswith("c") and name[1:].isdigit():
                order = int(name[1:])
                if order <= self.grid.M_max:
                    c_fields[order] = profile.u_fields
                    c_active_order = max(c_active_order, order)
            elif name.startswith("s") and name[1:].isdigit():
                order = int(name[1:])
                if order <= self.grid.M_max:
                    s_fields[order] = profile.u_fields
                    s_active_order = max(s_active_order, order)
        geometry.update(
            self.a,
            self.R0,
            self.Z0,
            self.grid,
            self.h_profile.u_fields,
            self.v_profile.u_fields,
            self.k_profile.u_fields,
            c_fields,
            s_fields,
            c_active_order=c_active_order,
            s_active_order=s_active_order,
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
    def FFn_r(self) -> np.ndarray:
        ffn_psin = self._diagnostic_FFn_psin
        raw = ffn_psin * self.psin_r

        tail_candidate_ffn_psin = _stabilize_tail_profile_on_rho(
            self.rho,
            ffn_psin,
            fit_end_offset=6,
            fit_count=8,
            replace_count=6,
            degree=2,
        )
        tail_candidate = tail_candidate_ffn_psin * self.psin_r
        if _tail_monotonicity_violations(tail_candidate, tail=8) < _tail_monotonicity_violations(raw, tail=8):
            ffn_psin = tail_candidate_ffn_psin
            raw = tail_candidate

        raw_changes = _axis_derivative_sign_changes(raw)
        if raw_changes == 0:
            return raw

        stabilized_FFn_psin = _stabilize_axis_even_profile(
            self.rho,
            ffn_psin,
            fit_start=6,
            fit_count=12,
            replace_count=12,
            degree=3,
        )
        stabilized = stabilized_FFn_psin * self.psin_r
        if _axis_derivative_sign_changes(stabilized) < raw_changes:
            return stabilized
        return raw

    @property
    def F2(self) -> np.ndarray:
        """物理 F^2 剖面."""
        FF_int = self.grid.integrate(self.FF_r, p=1)
        return (self.R0 * self.B0) ** 2 + 2.0 * (FF_int - FF_int[-1])

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
    def Pn_r(self) -> np.ndarray:
        return self.Pn_psin * self.psin_r

    @property
    def P(self) -> np.ndarray:
        """物理压强剖面 P."""
        P_int = self.grid.integrate(self.P_r, p=1)
        return P_int - P_int[-1]

    @property
    def beta_t(self) -> np.ndarray:
        """环向比压 beta_t = 2*mu0*<P> / B0^2."""
        P_avg = float(self.grid.quadrature(self.P * self.V_r) / self.grid.quadrature(self.V_r))
        return float(2.0 * MU0 * P_avg / self.B0**2)

    @property
    def Gn1(self) -> np.ndarray:
        """GS 算子源项分量 alpha1 前的归一化项."""
        R, JdivR = self.geometry.R, self.geometry.JdivR
        return JdivR * (self._diagnostic_FFn_psin[:, None] + R**2 * self.Pn_psin[:, None])

    @property
    def Gn2(self) -> np.ndarray:
        """GS 算子几何分量 alpha2 前的归一化项."""
        geometry = self.geometry
        return (
            geometry.gttdivJR * self.psin_rr[:, None]
            + (geometry.gttdivJR_r - geometry.grtdivJR_t) * self.psin_r[:, None]
        )

    @property
    def G(self) -> np.ndarray:
        """GS 算子残差场 G = alpha1 * Gn1 + alpha2 * Gn2."""
        return self.alpha1 * self.Gn1 + self.alpha2 * self.Gn2

    @property
    def Ip(self) -> np.ndarray:
        """总等离子体电流 Ip (Amps)."""
        return -self.alpha1 * self.grid.quadrature(self.Gn1) / MU0

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
        q_values = np.asarray(self.q, dtype=np.float64)
        finite = np.isfinite(q_values)
        if np.all(finite):
            q_for_diff = q_values
        elif np.count_nonzero(finite) >= 2:
            q_for_diff = np.interp(self.rho, self.rho[finite], q_values[finite])
        else:
            return np.zeros_like(q_values)

        q_r = self.grid.corrected_even_derivative(q_for_diff)
        q_safe = np.maximum(np.abs(q_for_diff), 1e-15)
        shear = self.rho * q_r / q_safe
        shear[~finite] = 0.0
        shear[0] = 0.0
        return shear

    @property
    def Itor(self) -> np.ndarray:
        """环向电流分布 I_tor(rho), model-side diagnostic."""
        return 2.0 * np.pi * self.Kn * self.alpha2 * self.psin_r / MU0

    @property
    def jtor(self) -> np.ndarray:
        """环向电流密度 j_phi, model-side diagnostic."""
        ffn_psin = self._diagnostic_FFn_psin
        head_changes = _axis_derivative_sign_changes(ffn_psin, head=12)
        if head_changes > 0:
            candidate_ffn = _stabilize_axis_even_profile(
                self.rho,
                ffn_psin,
                fit_start=1,
                fit_count=8,
                replace_count=8,
                degree=2,
            )
            if _axis_derivative_sign_changes(candidate_ffn, head=12) < head_changes:
                ffn_psin = candidate_ffn

        with np.errstate(divide="ignore", invalid="ignore"):
            raw_jtor = (
                -self.alpha1
                / (MU0 * self.S_r)
                * (2.0 * np.pi * ffn_psin * self.Ln_r + self.V_r * self.Pn_psin / (2.0 * np.pi))
            )

        strong_jtor = _stabilize_axis_even_profile(
            self.rho,
            raw_jtor,
            fit_start=6,
            fit_count=16,
            replace_count=20,
            degree=2,
        )
        if _axis_derivative_sign_changes(strong_jtor, head=12) < _axis_derivative_sign_changes(raw_jtor, head=12):
            jtor = strong_jtor
        else:
            jtor = _stabilize_axis_even_profile(
                self.rho,
                raw_jtor,
                fit_start=1,
                fit_count=6,
                replace_count=3,
                degree=2,
            )

        _extrapolate_inplace(self.rho, jtor, p=2)
        return jtor

    @property
    def jpara(self) -> np.ndarray:
        """平行电流密度 <j.B>/B0, model-side diagnostic."""
        psin_r_diag, psin_rr_diag = _smoothed_psin_radial_derivatives(self.rho, self.psin)
        F_r = self.grid.corrected_even_derivative(self.F)
        term_r = (
            self.Kn_r * psin_r_diag / self.F
            + self.Kn * psin_rr_diag / self.F
            - self.Kn * psin_r_diag * F_r / self.F**2
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            jpara = self.alpha2 / MU0 * self.F / self.Ln_r * term_r

        jpara = _stabilize_axis_even_profile(self.rho, jpara, fit_start=1, fit_count=20, replace_count=20, degree=2)
        _extrapolate_inplace(self.rho, jpara, p=2)
        return jpara

    @property
    def jphi(self) -> np.ndarray:
        """局部环向电流密度 j_phi(R, Z)."""
        R = self.geometry.R
        with np.errstate(divide="ignore", invalid="ignore"):
            jphi = -self.alpha1 / (MU0 * R) * (self._diagnostic_FFn_psin[:, None] + R**2 * self.Pn_psin[:, None])

        _extrapolate_inplace(self.rho, jphi, p=2)
        return jphi

    @property
    def Psi(self) -> np.ndarray:
        """物理极向磁通 Psi."""
        return 2.0 * np.pi * self.alpha2 * self.psin

    @property
    def Phi(self) -> np.ndarray:
        """环向磁通 Phi."""
        return 2.0 * np.pi * _integrate_profile_spline(self.rho, self.F * self.Ln_r)

    @property
    def _diagnostic_FFn_psin(self) -> np.ndarray:
        values = np.asarray(self.FFn_psin, dtype=np.float64)
        if _derivative_sign_changes(values) >= 8:
            values = _stabilize_axis_even_profile(
                self.rho,
                _smooth_three_point_profile(values, passes=2),
                fit_start=1,
                fit_count=6,
                replace_count=2,
                degree=2,
            )

        tail_candidate = _stabilize_tail_profile_on_rho(
            self.rho,
            values,
            fit_end_offset=6,
            fit_count=8,
            replace_count=3,
            degree=2,
        )
        tail_changes = _tail_derivative_sign_changes(values, tail=12)
        tail_candidate_changes = _tail_derivative_sign_changes(tail_candidate, tail=12)
        if tail_candidate_changes < tail_changes or _tail_last_jump_ratio(values, tail_candidate) <= 0.5:
            values = tail_candidate

        head_candidate = _stabilize_axis_even_profile(
            self.rho,
            values,
            fit_start=1,
            fit_count=8,
            replace_count=8,
            degree=2,
        )
        if _axis_derivative_sign_changes(head_candidate, head=12) < _axis_derivative_sign_changes(values, head=12):
            values = head_candidate
        return values

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
        label_ref: str = "reference",
        label_other: str = "current",
        target_grid: Grid | None = None,
        profile_degree: int | None = None,
        native_grid: bool = False,
    ) -> dict[str, float]:
        """Compare this equilibrium against another one."""

        return plot_comparison(
            self,
            other,
            outpath=outpath,
            show=show,
            label_ref=label_ref,
            label_other=label_other,
            target_grid=target_grid,
            profile_degree=profile_degree,
            native_grid=native_grid,
        )

    def resample(
        self,
        *,
        target_grid: Grid | None = None,
        profile_degree: int | None = None,
        native_grid: bool = False,
    ) -> "Equilibrium":
        """将当前平衡快照插值到目标网格."""

        return _build_resampled_equilibrium(
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
    r = np.asarray(r, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError(f"Expected r to be 1D, got {r.shape}")
    if y.ndim < 1 or y.shape[0] != r.shape[0]:
        raise ValueError(f"Expected y to have leading shape {r.shape[0]}, got {y.shape}")

    if r[0] < 1e-4 and r.size >= 3 and np.all(np.isfinite(y[1])) and np.all(np.isfinite(y[2])):
        r1, r2 = r[1] ** p, r[2] ** p
        y[0] = (y[1] * r2 - y[2] * r1) / (r2 - r1)


def _normalize_shape_profiles(active_profiles: dict[str, Profile]) -> dict[str, Profile]:
    if not isinstance(active_profiles, dict):
        raise TypeError(f"active_profiles must be dict[str, Profile], got {type(active_profiles).__name__}")
    for name, profile in active_profiles.items():
        if not isinstance(name, str):
            raise TypeError(f"active profile names must be str, got {type(name).__name__}")
        profile_type = type(profile)
        if not (
            isinstance(profile, Profile)
            or (
                profile_type.__name__ == Profile.__name__
                and getattr(profile_type, "__module__", None) == Profile.__module__
            )
        ):
            raise TypeError(f"active profile {name!r} must be Profile, got {type(profile).__name__}")
    return _minimize_shape_profiles(active_profiles)


def _shape_profiles(equilibrium: Equilibrium) -> dict[str, Profile]:
    return equilibrium.active_profiles


def _shape_profile_payload(profiles: dict[str, Profile]) -> dict[str, dict[str, Profile]]:
    return {
        "active_profiles": _minimize_shape_profiles(profiles),
    }


def _build_default_shape_profile(name: str) -> Profile:
    power = 0
    if name.startswith(("c", "s")) and name[1:].isdigit():
        power = int(name[1:])
    return Profile(scale=1.0, power=power, envelope_power=1, offset=0.0, coeff=None)


def _is_default_shape_profile(name: str, profile: Profile) -> bool:
    default = _build_default_shape_profile(name)
    coeff = None if profile.coeff is None else np.asarray(profile.coeff, dtype=np.float64)
    coeff_is_zero = coeff is None or coeff.size == 0 or np.allclose(coeff, 0.0, atol=1e-14, rtol=0.0)
    return (
        np.isclose(profile.scale, default.scale)
        and int(profile.power) == int(default.power)
        and int(profile.envelope_power) == int(default.envelope_power)
        and np.isclose(profile.offset, default.offset)
        and coeff_is_zero
    )


def _minimize_shape_profiles(profiles: dict[str, Profile]) -> dict[str, Profile]:
    minimized: dict[str, Profile] = {}
    for name, profile in profiles.items():
        copied = profile.copy()
        if copied.coeff is not None:
            coeff = np.asarray(copied.coeff, dtype=np.float64)
            copied.coeff = None if coeff.size == 0 or np.allclose(coeff, 0.0, atol=1e-14, rtol=0.0) else coeff.copy()
        if _is_default_shape_profile(name, copied):
            continue
        minimized[name] = copied
    return minimized


def _unique_profiles(profiles) -> list[Profile]:
    unique: dict[int, Profile] = {}
    for profile in profiles:
        unique.setdefault(id(profile), profile)
    return list(unique.values())


def _shape_profile_plot_meta(name: str) -> dict[str, str | None]:
    meta = SHAPE_PROFILE_PLOT_META.get(name)
    if meta is not None:
        return meta

    if name.startswith("c") and name[1:].isdigit():
        label = rf"$c_{int(name[1:])}$"
        style = {"linestyle": "--", "marker": None}
    elif name.startswith("s") and name[1:].isdigit():
        label = rf"$s_{int(name[1:])}$"
        style = {"linestyle": "-", "marker": "x"}
    else:
        label = name
        style = {"linestyle": "-", "marker": None}
    color = _EXTRA_SHAPE_PROFILE_COLORS[sum(ord(ch) for ch in name) % len(_EXTRA_SHAPE_PROFILE_COLORS)]
    return {"color": color, "label": label, **style}


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

    surface_eq = _build_resampled_equilibrium(
        target_grid=target_grid,
        profile_degree=profile_degree,
        native_grid=native_grid,
        equilibrium=equilibrium,
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
    label_ref: str = "reference",
    label_other: str = "current",
    target_grid: Grid | None = None,
    profile_degree: int | None = None,
    native_grid: bool = False,
) -> dict[str, float]:
    """Render a veqpy-poor-style comparison figure with one shared surface panel."""
    ref_plot = reference
    other_plot = other

    shape_keys = ["h", "k", "s1"]
    groups = [(key, _shape_profile_plot_meta(key)["label"], None) for key in shape_keys]
    groups.extend(
        [
            ("psi_r", r"$\psi_\rho$", None),
            ("FF_r", r"$FF_\rho$", None),
            ("mu0_P_r", r"$\mu_0 P_\rho$", None),
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
            "mu0_P_r": MU0 * np.asarray(eq.P_r, dtype=np.float64),
            "Itor": np.asarray(eq.Itor, dtype=np.float64),
            "jtor": np.asarray(eq.jtor, dtype=np.float64),
            "jpara": np.asarray(eq.jpara, dtype=np.float64),
        }
        profiles = _shape_profiles(eq)
        for key in shape_keys:
            profile = profiles.get(key)
            if profile is None:
                data[key] = np.zeros_like(eq.rho, dtype=np.float64)
            else:
                data[key] = np.asarray(profile.u, dtype=np.float64)
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
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(
        3,
        4,
        figure=fig,
        width_ratios=[1.1, 1.0, 1.0, 1.0],
        hspace=0.25,
        wspace=0.3,
        top=0.95,
        bottom=0.1,
        left=0.05,
        right=0.98,
    )

    surface_grid = target_grid or Grid(
        Nr=64,
        Nt=64,
        scheme="uniform",
        L_max=max(reference.grid.L_max, other.grid.L_max),
        M_max=max(reference.grid.M_max, other.grid.M_max),
    )
    ref_surface = _build_resampled_equilibrium(
        target_grid=surface_grid,
        profile_degree=profile_degree,
        native_grid=native_grid,
        equilibrium=ref_plot,
    )
    other_surface = _build_resampled_equilibrium(
        target_grid=surface_grid,
        profile_degree=profile_degree,
        native_grid=native_grid,
        equilibrium=other_plot,
    )
    ref_surface_data = _build_surface_panel_data(ref_surface)
    other_surface_data = _build_surface_panel_data(other_surface)
    shared_boundary = _merge_surface_boundaries(ref_surface_data["boundary"], other_surface_data["boundary"])
    surface_ax = fig.add_subplot(gs[:, 0])
    _render_comparison_surface_overlay_panel(
        surface_ax,
        ref_surface_data,
        other_surface_data,
        shared_boundary,
        label_ref=label_ref,
        label_other=label_other,
    )

    metric_axes = [fig.add_subplot(gs[row, col]) for row in range(3) for col in range(1, 4)]
    for i, (ax, (key, ylabel, scale)) in enumerate(zip(metric_axes, groups, strict=True)):
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
        ax.set_ylabel(ylabel)
        # ax.tick_params(direction="in", top=True, right=True, labelsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        _add_top_headroom(ax, 0.15)
        ax.text(
            0.03,
            0.97,
            f"err = {errors[f'rel_{key}_max']:.1e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

        if i == 0:
            ax.legend(loc="best", frameon=False)
        if i >= 6:
            ax.set_xlabel(r"$\rho$")
        else:
            ax.set_xticklabels([])

    visible_metric_axes = [ax for ax in metric_axes if ax.get_visible()]
    if visible_metric_axes:
        metric_boxes = [ax.get_position() for ax in visible_metric_axes]
        metrics_x0 = min(box.x0 for box in metric_boxes)
        metrics_y1 = max(box.y1 for box in metric_boxes)
        fig.text(
            metrics_x0,
            metrics_y1 + 0.01,
            "(b) Profiles",
            fontsize=SUBPLOT_TITLE_FONTSIZE,
            ha="left",
            va="bottom",
        )

    if outpath is not None:
        fig.savefig(Path(outpath), dpi=300, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return errors


def _build_resampled_equilibrium(
    equilibrium: Equilibrium,
    *,
    target_grid: Grid | None,
    profile_degree: int | None,
    native_grid: bool,
) -> Equilibrium:
    if native_grid:
        return equilibrium

    source_grid = equilibrium.grid
    plot_grid = target_grid or Grid(Nr=64, Nt=64, scheme="uniform", L_max=source_grid.L_max, M_max=source_grid.M_max)
    degree = min(source_grid.Nr - 1, 16) if profile_degree is None else int(profile_degree)

    def resample_vector(y_src: np.ndarray) -> np.ndarray:
        return _resample_profile_spline(
            source_grid.rho, np.asarray(y_src, dtype=np.float64), plot_grid.rho, degree=degree, strict=True
        )

    def _resample_profile_triplet(profile: Profile) -> Profile:
        out = profile.copy()
        out.update(grid=plot_grid)
        return out

    psin_r = _resample_profile_spline(
        source_grid.rho,
        np.asarray(equilibrium.psin_r, dtype=np.float64),
        plot_grid.rho,
        left=0.0,
        degree=degree,
        strict=True,
    )
    psin = _resample_profile_spline(
        source_grid.rho,
        np.asarray(equilibrium.psin, dtype=np.float64),
        plot_grid.rho,
        left=0.0,
        right=1.0,
        degree=degree,
        strict=True,
    )
    FFn_psin = _resample_profile_spline(
        source_grid.rho,
        np.asarray(equilibrium.FFn_psin, dtype=np.float64),
        plot_grid.rho,
        left=0.0,
        right=0.0,
        degree=degree,
        strict=True,
    )
    Pn_psin = _resample_profile_spline(
        source_grid.rho,
        np.asarray(equilibrium.Pn_psin, dtype=np.float64),
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
        **_shape_profile_payload(
            {name: _resample_profile_triplet(profile) for name, profile in _shape_profiles(equilibrium).items()}
        ),
        psin=psin,
        FFn_psin=FFn_psin,
        Pn_psin=Pn_psin,
        psin_r=psin_r,
        psin_rr=plot_grid.corrected_linear_derivative(psin_r),
        alpha1=equilibrium.alpha1,
        alpha2=equilibrium.alpha2,
    )


def _render_equilibrium_summary(*, surface_equilibrium: Equilibrium, profile_equilibrium: Equilibrium):
    fig = plt.figure(figsize=(22, 6.5))
    gs = GridSpec(
        2,
        9,
        figure=fig,
        width_ratios=[1.05, 0.35, 1.28, 0.25, 1.05, 0.35, 1.28, 0.25, 1.05],
        height_ratios=[1, 1],
        hspace=0.42,
        wspace=0.0,
        top=0.95,
        bottom=0.1,
        left=0.025,
        right=0.975,
    )

    panel_a = _build_surface_panel_data(surface_equilibrium)
    panel_b = _build_shape_panel_data(profile_equilibrium)
    panel_c = _build_source_panel_data(profile_equilibrium)
    panel_d = _build_jphi_panel_data(surface_equilibrium)
    panel_e = _build_current_panel_data(profile_equilibrium)
    panel_f = _build_safety_panel_data(profile_equilibrium)
    panel_g = _build_gs_residual_panel_data(surface_equilibrium)

    _render_panel_a_surfaces(fig.add_subplot(gs[:, 0]), fig, panel_a)
    _render_panel_b_shapes(fig.add_subplot(gs[0, 2]), panel_b)
    _render_panel_c_sources(fig.add_subplot(gs[1, 2]), panel_c)
    _render_panel_d_jphi(fig.add_subplot(gs[:, 4]), fig, panel_d, panel_a["boundary"])
    _render_panel_e_current_1d(fig.add_subplot(gs[0, 6]), panel_e)
    _render_panel_f_safety(fig.add_subplot(gs[1, 6]), panel_f)
    _render_panel_g_gs_residual(fig.add_subplot(gs[:, 8]), fig, panel_g, panel_a["boundary"])
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


def _merge_surface_boundaries(*boundaries: dict) -> dict:
    if not boundaries:
        raise ValueError("At least one boundary must be provided")
    R = np.concatenate([np.asarray(boundary["R"], dtype=np.float64) for boundary in boundaries])
    Z = np.concatenate([np.asarray(boundary["Z"], dtype=np.float64) for boundary in boundaries])
    return {"R": R, "Z": Z}


def _build_shape_panel_data(equilibrium: Equilibrium) -> dict:
    profiles = _shape_profiles(equilibrium)

    values = {key: profile.u for key, profile in profiles.items()}
    return {"shape": {"rho": equilibrium.rho, "values": values}}


def _build_source_panel_data(equilibrium: Equilibrium) -> dict:
    return {
        "rho": equilibrium.rho,
        "psi_r": equilibrium.alpha2 * equilibrium.psin_r.copy(),
        "FF_r": equilibrium.FF_r.copy(),
        "mu0_P_r": MU0 * equilibrium.P_r.copy(),
    }


def _build_jphi_panel_data(surface_equilibrium: Equilibrium) -> dict:
    return {
        "R": np.hstack([surface_equilibrium.geometry.R, surface_equilibrium.geometry.R[:, :1]]),
        "Z": np.hstack([surface_equilibrium.geometry.Z, surface_equilibrium.geometry.Z[:, :1]]),
        "jphi": np.hstack([surface_equilibrium.jphi, surface_equilibrium.jphi[:, :1]]) / 1e6,
    }


def _build_gs_residual_panel_data(surface_equilibrium: Equilibrium) -> dict:
    return {
        "R": np.hstack([surface_equilibrium.geometry.R, surface_equilibrium.geometry.R[:, :1]]),
        "Z": np.hstack([surface_equilibrium.geometry.Z, surface_equilibrium.geometry.Z[:, :1]]),
        "G": np.hstack([surface_equilibrium.G, surface_equilibrium.G[:, :1]]),
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


def _resample_profile_linear(
    rho_src: np.ndarray,
    y_src: np.ndarray,
    rho_eval: np.ndarray,
    *,
    left: float | None = None,
    right: float | None = None,
) -> np.ndarray:
    rho_src = np.asarray(rho_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    rho_eval = np.asarray(rho_eval, dtype=np.float64)
    if rho_src.ndim != 1 or y_src.ndim != 1 or rho_eval.ndim != 1 or rho_src.shape != y_src.shape:
        raise ValueError("rho_src, y_src, rho_eval must be 1D arrays and source arrays must share shape")
    left_val = float(y_src[0]) if left is None else float(left)
    right_val = float(y_src[-1]) if right is None else float(right)
    return np.interp(rho_eval, rho_src, y_src, left=left_val, right=right_val)


def _active_shape_keys(reference: Equilibrium, other: Equilibrium) -> list[str]:
    names = list(dict.fromkeys([*_shape_profiles(reference), *_shape_profiles(other)]))
    return [key for key in names if key in _shape_profiles(reference) or key in _shape_profiles(other)]


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


def _get_gs_residual_cmap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "gs_residual",
        [
            (0.0, "#2166ac"),
            (0.5, "#f7f7f7"),
            (1.0, "#b2182b"),
        ],
    )


def _add_top_headroom(ax: plt.Axes, ratio: float) -> None:
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    if ratio > 0.0:
        ax.set_ylim(y0, y1 + ratio * span)
    else:
        ax.set_ylim(y0 + ratio * span, y1)


def _render_panel_a_surfaces(ax: plt.Axes, fig: plt.Figure, data: dict):
    ax.set_title("(a) Flux Surfaces", fontsize=SUBPLOT_TITLE_FONTSIZE)
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
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\rho$")
    cbar.locator = ticker.MaxNLocator(nbins=2)
    cbar.update_ticks()


def _render_comparison_surface_overlay_panel(
    ax: plt.Axes,
    reference_data: dict,
    other_data: dict,
    boundary: dict,
    *,
    label_ref: str,
    label_other: str,
) -> None:
    ax.set_title("(a) Flux Surfaces", fontsize=SUBPLOT_TITLE_FONTSIZE)
    for ray in reference_data.get("rays", []):
        ax.plot(ray["R"], ray["Z"], color="#c5c8ce", linewidth=0.6, alpha=0.35, zorder=1)
    for ray in other_data.get("rays", []):
        ax.plot(ray["R"], ray["Z"], color="#f0b3b3", linewidth=0.6, alpha=0.25, zorder=1)

    for ci, surf in enumerate(reference_data["surfaces"]):
        ax.plot(
            surf["R"],
            surf["Z"],
            color=BLACK,
            linewidth=1.15 if ci < len(reference_data["surfaces"]) - 1 else 1.6,
            alpha=0.85,
            zorder=2,
            label=label_ref if ci == 0 else None,
        )
    for ci, surf in enumerate(other_data["surfaces"]):
        ax.plot(
            surf["R"],
            surf["Z"],
            color=RED,
            linewidth=1.15 if ci < len(other_data["surfaces"]) - 1 else 1.6,
            alpha=0.75,
            zorder=3,
            label=label_other if ci == 0 else None,
        )

    ax.plot(
        reference_data["axis"]["R"],
        reference_data["axis"]["Z"],
        marker="o",
        color=BLACK,
        markersize=3,
        linestyle="None",
        zorder=4,
    )
    ax.plot(
        other_data["axis"]["R"],
        other_data["axis"]["Z"],
        marker="o",
        color=RED,
        markersize=3,
        linestyle="None",
        zorder=4,
    )
    _apply_rz_limits(ax, boundary)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="upper right", frameon=False)


def _render_panel_b_shapes(ax: plt.Axes, data: dict):
    ax.set_title("(b) Shape Profiles", fontsize=SUBPLOT_TITLE_FONTSIZE)
    shape = data["shape"]
    for key, vals in shape["values"].items():
        meta = _shape_profile_plot_meta(key)
        ax.plot(
            shape["rho"],
            vals,
            linestyle=meta["linestyle"],
            marker=meta["marker"],
            color=meta["color"],
            label=meta["label"],
        )

    ax.set(xlabel=r"$\rho$", ylabel="Profile")
    if shape["values"]:
        if len(shape["values"]) < 5:
            ax.legend(loc="center left")
        elif len(shape["values"]) > 8:
            ax.legend(loc="center left", ncols=3)
        else:
            ax.legend(loc="center left", ncols=2)
    ax.grid(True)


def _render_panel_c_sources(ax: plt.Axes, data: dict):
    ax.set_title("(c) Source Profiles", fontsize=SUBPLOT_TITLE_FONTSIZE)
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
    ax.set_title("(d) Current Density", fontsize=SUBPLOT_TITLE_FONTSIZE)
    R_plot, Z_plot, j_plot = data["R"], data["Z"], data["jphi"]
    cmap = _get_trunc_inferno()
    vmin = min(float(np.nanmin(j_plot)), 0.0)
    vmax = float(np.nanmax(j_plot))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    ax.set_facecolor(cmap(norm(0.0)))
    pcm = ax.contourf(R_plot, Z_plot, j_plot, levels=np.linspace(vmin, vmax, 128), cmap=cmap, norm=norm)
    _apply_rz_limits(ax, boundary)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label(r"$j_\phi$ [MA/m$^2$]")
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()


def _render_panel_g_gs_residual(ax: plt.Axes, fig: plt.Figure, data: dict, boundary: dict):
    ax.set_title("(g) GS Residual", fontsize=SUBPLOT_TITLE_FONTSIZE)
    R_plot, Z_plot, G_plot = data["R"], data["Z"], data["G"]
    finite_abs = np.abs(G_plot[np.isfinite(G_plot)])
    vmax = float(np.quantile(finite_abs, 0.99)) if finite_abs.size else 0.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    vmin = -vmax
    cmap = _get_gs_residual_cmap()
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    ax.set_facecolor(cmap(norm(0.0)))
    pcm = ax.contourf(R_plot, Z_plot, G_plot, levels=np.linspace(vmin, vmax, 129), cmap=cmap, norm=norm)
    _apply_rz_limits(ax, boundary)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label(r"$G$")
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()


def _render_panel_e_current_1d(ax: plt.Axes, data: dict):
    ax.set_title("(e) Current Profiles", fontsize=SUBPLOT_TITLE_FONTSIZE)
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
    ax.set_title("(f) Safety Factor", fontsize=SUBPLOT_TITLE_FONTSIZE)
    rho = data["rho"]
    if len(rho) < 32:
        ax.plot(data["rho"], data["q"], "-o", color=BLUE, label=r"$q$")
        ax.plot(data["rho"], data["s"], "--o", color=ORANGE, label=r"$s$")
    else:
        ax.plot(data["rho"], data["q"], "-", color=BLUE, label=r"$q$")
        ax.plot(data["rho"], data["s"], "--", color=ORANGE, label=r"$s$")
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


def _resample_profile_spline(
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
    """Resample one profile with a natural cubic spline for smooth plotting."""

    rho_src, y_src, rho_eval = map(lambda x: np.asarray(x, dtype=np.float64), (rho_src, y_src, rho_eval))
    if rho_src.ndim != 1 or y_src.ndim != 1 or len(rho_src) != len(y_src):
        raise ValueError("rho_src and y_src must be 1D arrays with equal length")

    x_src = rho_src**2 if even else rho_src
    x_eval = rho_eval**2 if even else rho_eval
    y_fit = y_src.copy()
    if left is not None:
        y_fit[0] = float(left)
    if right is not None:
        y_fit[-1] = float(right)

    spline = CubicSpline(x_src, y_fit, bc_type="natural", extrapolate=True)
    y_eval = np.asarray(spline(x_eval), dtype=np.float64)

    if left is not None:
        y_eval[0] = float(left)
    if right is not None:
        y_eval[-1] = float(right)

    if strict:
        y_back = np.asarray(spline(x_src), dtype=np.float64)
        if left is not None:
            y_back[0] = float(left)
        if right is not None:
            y_back[-1] = float(right)
        err = float(np.max(np.abs(y_back - y_fit)))
        if err > atol:
            raise RuntimeError(f"Spline resample lost nodal interpolation: max err={err:.3e}")

    return y_eval


def _integrate_profile_spline(rho: np.ndarray, values: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if rho.ndim != 1 or values.ndim != 1 or rho.shape != values.shape:
        raise ValueError(f"Expected rho/values to share a 1D shape, got {rho.shape} and {values.shape}")
    if rho.size == 0:
        return values.copy()
    if rho.size == 1:
        return np.zeros_like(values)

    spline = CubicSpline(rho, values, bc_type="natural", extrapolate=True)
    anti = spline.antiderivative()
    out = np.asarray(anti(rho) - anti(rho[0]), dtype=np.float64)
    out[0] = 0.0
    return out


def _integrate_axis_linear_profile(rho: np.ndarray, values: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if rho.ndim != 1 or values.ndim != 1 or rho.shape != values.shape:
        raise ValueError(f"Expected rho/values to share a 1D shape, got {rho.shape} and {values.shape}")
    if rho.size == 0:
        return values.copy()
    if rho.size == 1:
        return np.zeros_like(values)

    rho_safe = np.where(rho > 1e-12, rho, 1.0)
    reduced = values / rho_safe
    reduced[0] = reduced[1]
    reduced = _stabilize_axis_even_profile(rho, reduced, fit_start=1, fit_count=6, replace_count=2, degree=2)

    x = rho * rho
    spline = CubicSpline(x, reduced, bc_type="natural", extrapolate=True)
    anti = spline.antiderivative()
    out = 0.5 * np.asarray(anti(x) - anti(x[0]), dtype=np.float64)
    out[0] = 0.0
    return out


def _recover_psin_from_psin_r(grid: Grid, psin_r: np.ndarray) -> np.ndarray:
    psin = grid.integrate(psin_r, p=1)
    scale = float(psin[-1] - psin[0])
    if abs(scale) < 1e-12:
        raise ValueError("psin_r does not span a valid normalized flux interval")
    psin = (psin - psin[0]) / scale
    psin[0] = 0.0
    psin[-1] = 1.0
    return psin


def _differentiate_even_profile_on_rho2(rho: np.ndarray, values: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if rho.ndim != 1 or values.ndim != 1 or rho.shape != values.shape:
        raise ValueError(f"Expected rho/values to share a 1D shape, got {rho.shape} and {values.shape}")
    if rho.size < 2:
        return np.zeros_like(values)

    smooth = _stabilize_axis_even_profile(rho, values, fit_start=1, fit_count=6, replace_count=2, degree=2)
    x = rho * rho
    finite = np.isfinite(smooth)
    if not np.all(finite):
        if np.count_nonzero(finite) < 2:
            return np.zeros_like(values)
        smooth = np.interp(x, x[finite], smooth[finite])
    spline = CubicSpline(x, smooth, bc_type="natural", extrapolate=True)
    out = 2.0 * rho * np.asarray(spline(x, 1), dtype=np.float64)
    out[0] = 0.0
    return out


def _smoothed_psin_radial_derivatives(rho: np.ndarray, psin: np.ndarray, *, degree: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(rho, dtype=np.float64)
    psin = np.asarray(psin, dtype=np.float64)
    if rho.ndim != 1 or psin.ndim != 1 or rho.shape != psin.shape:
        raise ValueError(f"Expected rho/psin to share a 1D shape, got {rho.shape} and {psin.shape}")
    if rho.size < 4:
        psin_r = np.gradient(psin, rho, edge_order=1)
        psin_rr = np.gradient(psin_r, rho, edge_order=1)
        return psin_r, psin_rr

    x = rho * rho
    xi = 2.0 * x - 1.0
    fit_degree = min(int(degree), rho.size - 1)
    vandermonde = chebvander(xi, fit_degree)
    coeff, *_ = np.linalg.lstsq(vandermonde, psin, rcond=None)
    coeff_r = chebder(coeff)
    coeff_rr = chebder(coeff_r)

    dpsin_dx = 2.0 * chebval(xi, coeff_r)
    d2psin_dx2 = 4.0 * chebval(xi, coeff_rr)
    psin_r = 2.0 * rho * dpsin_dx
    psin_rr = 2.0 * dpsin_dx + 4.0 * x * d2psin_dx2
    return np.asarray(psin_r, dtype=np.float64), np.asarray(psin_rr, dtype=np.float64)


def _axis_derivative_sign_changes(values: np.ndarray, *, head: int = 12) -> int:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected values to be 1D, got {values.shape}")
    window = values[: min(int(head), values.shape[0])]
    delta = np.diff(window)
    signs = np.sign(delta)
    nonzero = signs[signs != 0.0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))


def _tail_derivative_sign_changes(values: np.ndarray, *, tail: int = 12) -> int:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected values to be 1D, got {values.shape}")
    window = values[-min(int(tail), values.shape[0]) :]
    delta = np.diff(window)
    signs = np.sign(delta)
    nonzero = signs[signs != 0.0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))


def _tail_monotonicity_violations(values: np.ndarray, *, tail: int = 8, tol: float = 1.0e-8) -> int:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected values to be 1D, got {values.shape}")
    window = values[-min(int(tail), values.shape[0]) :]
    return int(np.count_nonzero(np.diff(window) < -abs(float(tol))))


def _derivative_sign_changes(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected values to be 1D, got {values.shape}")
    delta = np.diff(values)
    signs = np.sign(delta)
    nonzero = signs[signs != 0.0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))


def _smooth_three_point_profile(values: np.ndarray, *, passes: int = 1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected values to be 1D, got {values.shape}")
    out = values.copy()
    if out.size < 3:
        return out
    for _ in range(max(int(passes), 0)):
        prev = out.copy()
        out[1:-1] = 0.25 * prev[:-2] + 0.5 * prev[1:-1] + 0.25 * prev[2:]
    return out


def _stabilize_axis_even_profile(
    rho: np.ndarray,
    profile: np.ndarray,
    *,
    fit_start: int = 6,
    fit_count: int = 12,
    replace_count: int = 10,
    degree: int = 3,
) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    if rho.ndim != 1 or profile.ndim != 1 or rho.shape != profile.shape:
        raise ValueError(f"Expected rho/profile to share a 1D shape, got {rho.shape} and {profile.shape}")

    start = max(int(fit_start), 0)
    stop = min(start + int(fit_count), profile.shape[0])
    replace_stop = min(int(replace_count), stop)
    fit_degree = min(int(degree), stop - start - 1)
    if replace_stop <= 0 or stop - start < 2 or fit_degree <= 0:
        return profile.copy()

    x_fit = rho[start:stop] ** 2
    vandermonde = np.vander(x_fit, N=fit_degree + 1, increasing=True)
    coeff, *_ = np.linalg.lstsq(vandermonde, profile[start:stop], rcond=None)

    stabilized = profile.copy()
    x_replace = rho[:replace_stop] ** 2
    stabilized[:replace_stop] = np.vander(x_replace, N=fit_degree + 1, increasing=True) @ coeff
    return stabilized


def _stabilize_tail_profile_on_rho(
    rho: np.ndarray,
    profile: np.ndarray,
    *,
    fit_end_offset: int = 6,
    fit_count: int = 8,
    replace_count: int = 3,
    degree: int = 2,
) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    if rho.ndim != 1 or profile.ndim != 1 or rho.shape != profile.shape:
        raise ValueError(f"Expected rho/profile to share a 1D shape, got {rho.shape} and {profile.shape}")

    stop = max(profile.shape[0] - int(fit_end_offset), 0)
    start = max(stop - int(fit_count), 0)
    replace_start = max(profile.shape[0] - int(replace_count), 0)
    fit_degree = min(int(degree), stop - start - 1)
    if stop - start < 2 or replace_start >= profile.shape[0] or fit_degree <= 0:
        return profile.copy()

    x_fit = rho[start:stop]
    vandermonde = np.vander(x_fit, N=fit_degree + 1, increasing=True)
    coeff, *_ = np.linalg.lstsq(vandermonde, profile[start:stop], rcond=None)

    stabilized = profile.copy()
    x_replace = rho[replace_start:]
    stabilized[replace_start:] = np.vander(x_replace, N=fit_degree + 1, increasing=True) @ coeff
    return stabilized


def _tail_last_jump_ratio(values: np.ndarray, candidate: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if values.ndim != 1 or candidate.ndim != 1 or values.shape != candidate.shape:
        raise ValueError(f"Expected values/candidate to share a 1D shape, got {values.shape} and {candidate.shape}")
    if values.size < 2:
        return 0.0
    baseline = abs(float(values[-1] - values[-2]))
    if baseline <= 1.0e-15:
        return 0.0
    return abs(float(candidate[-1] - candidate[-2])) / baseline
