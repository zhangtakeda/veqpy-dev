"""
Module: model.equilibrium

Role:
- Hold an equilibrium snapshot on one grid.
- Re-derive geometry and diagnostics from root fields.
- Provide plotting, comparison, resampling, and other inspection capabilities.

Public API:
- Equilibrium

Notes:
- `Equilibrium` is a snapshot, not a solver runtime container.
- Does not own packed state or the residual hot path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from veqpy.base import Reactive, Serial
from veqpy.model.geometry import Geometry
from veqpy.model.geqdsk import Geqdsk
from veqpy.model.grid import Grid
from veqpy.model.profile import Profile

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.size": 10,
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


def _regularize_axis_linear_profile(
    values: np.ndarray,
    rho: np.ndarray,
    *,
    copy: bool = False,
) -> np.ndarray:
    values = np.array(values, dtype=np.float64, copy=copy)
    rho = np.asarray(rho, dtype=np.float64)
    if values.ndim != 1 or rho.ndim != 1 or values.shape != rho.shape:
        raise ValueError(
            f"Expected values/rho to share a 1D shape, got {values.shape} and {rho.shape}"
        )
    if values.size < 3 or abs(rho[0]) >= 1e-10:
        return values

    rho1 = float(rho[1])
    rho2 = float(rho[2])
    if abs(rho2 - rho1) < 1e-14:
        return values
    if not np.isfinite(values[1]) or not np.isfinite(values[2]):
        return values

    slope = (values[2] - values[1]) / (rho2 - rho1)
    values[0] = values[1] + slope * (rho[0] - rho1)
    return values


def _regularize_axis_linear_surface(
    values: np.ndarray,
    rho: np.ndarray,
    *,
    copy: bool = False,
) -> np.ndarray:
    values = np.array(values, dtype=np.float64, copy=copy)
    rho = np.asarray(rho, dtype=np.float64)
    if values.ndim != 2 or rho.ndim != 1 or values.shape[0] != rho.shape[0]:
        raise ValueError(f"values/rho shape mismatch: {values.shape} vs {rho.shape}")
    if values.shape[0] < 3 or abs(rho[0]) >= 1e-10:
        return values

    rho1 = float(rho[1])
    rho2 = float(rho[2])
    if abs(rho2 - rho1) < 1e-14:
        return values

    slope = (values[2] - values[1]) / (rho2 - rho1)
    values[0] = values[1] + slope * (rho[0] - rho1)
    return values


class Equilibrium(Reactive, Serial):
    """Equilibrium snapshot object on one grid."""

    root_properties = {
        "R0",
        "Z0",
        "B0",
        "a",
        "grid",
        "shape_profiles",
        "FFn_psin",
        "Pn_psin",
        "psin",
        "psin_r",
        "psin_rr",
        "alpha1",
        "alpha2",
    }

    def __init__(
        self,
        R0: float,
        Z0: float,
        B0: float,
        a: float,
        grid: Grid,
        shape_profiles: dict[str, Profile],
        FFn_psin: np.ndarray,
        Pn_psin: np.ndarray,
        psin: np.ndarray,
        psin_r: np.ndarray,
        psin_rr: np.ndarray,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
    ):
        """Initialize the equilibrium snapshot object."""
        super().__init__()

        self.R0 = R0
        self.Z0 = Z0
        self.B0 = B0
        self.a = a
        self.grid = grid
        self.shape_profiles = _normalize_shape_profiles(shape_profiles)

        for name, profile in self.shape_profiles.items():
            setattr(self, f"{name}_profile", profile)
        self.h_profile = self.shape_profiles.get("h", _build_default_shape_profile("h", self.grid))
        self.v_profile = self.shape_profiles.get("v", _build_default_shape_profile("v", self.grid))
        self.k_profile = self.shape_profiles.get("k", _build_default_shape_profile("k", self.grid))

        for profile in _unique_profiles(
            (*self.shape_profiles.values(), self.h_profile, self.v_profile, self.k_profile)
        ):
            profile.update(grid=self.grid)

        self.psin = np.asarray(psin, dtype=np.float64)
        self.FFn_psin = _regularize_axis_linear_profile(FFn_psin, grid.rho, copy=True)
        self.Pn_psin = _regularize_axis_linear_profile(Pn_psin, grid.rho, copy=True)
        self.psin_r = np.asarray(psin_r, dtype=np.float64)
        self.psin_rr = np.asarray(psin_rr, dtype=np.float64)
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
        console = Console(
            color_system=None, force_terminal=False, width=120, record=True, soft_wrap=False
        )
        with console.capture() as capture:
            console.print(self.__rich__())
        return capture.get().rstrip()

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        """Declare serializable construction root state."""
        attrs: dict[str, type] = {
            "R0": float,
            "Z0": float,
            "B0": float,
            "a": float,
            "grid": Grid,
            "shape_profiles": dict[str, Profile],
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
        return self.grid.cos_mtheta[1]

    @property
    def sin_theta(self) -> np.ndarray:
        return self.grid.sin_mtheta[1]

    @property
    def R(self) -> np.ndarray:
        return self.geometry.R

    @property
    def Z(self) -> np.ndarray:
        return self.geometry.Z

    @property
    def geometry(self) -> Geometry:
        """Re-materialize Geometry from current snapshot root fields."""
        geometry = Geometry(grid=self.grid)
        c_fields = np.zeros((self.grid.M_max + 1, 3, self.grid.Nr), dtype=np.float64)
        s_fields = np.zeros((self.grid.M_max + 1, 3, self.grid.Nr), dtype=np.float64)
        c_active_order = 0
        s_active_order = 0
        for name, profile in self.shape_profiles.items():
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
        """Flux-surface area S = -int R*Z_t dtheta."""
        R, Z_t = self.geometry.R, self.geometry.Z_t
        return -self.grid.integrate(R * Z_t, axis=1)

    @property
    def S_r(self) -> np.ndarray:
        """Flux-surface area derivative S_r = int J dtheta."""
        J = self.geometry.J
        return self.grid.integrate(J, axis=1)

    @property
    def V(self) -> np.ndarray:
        """Flux-surface volume V = -pi*int R**2*Z_t dtheta."""
        R, Z_t = self.geometry.R, self.geometry.Z_t
        return -np.pi * self.grid.integrate(R**2 * Z_t, axis=1)

    @property
    def V_r(self) -> np.ndarray:
        """Flux-surface volume derivative V_r = 2pi * int J*R dtheta."""
        R, J = self.geometry.R, self.geometry.J
        return (2 * np.pi) * self.grid.integrate(J * R, axis=1)

    @property
    def Kn(self) -> np.ndarray:
        """Normalized geometry factor Kn = int gttdivJR dtheta/(2pi)."""
        gttdivJR = self.geometry.gttdivJR
        return self.grid.integrate(gttdivJR, axis=1) / (2 * np.pi)

    @property
    def Kn_r(self) -> np.ndarray:
        """Radial derivative of Kn."""
        gttdivJR_r = self.geometry.gttdivJR_r
        return self.grid.integrate(gttdivJR_r, axis=1) / (2 * np.pi)

    @property
    def Ln_r(self) -> np.ndarray:
        """Normalized geometry factor Ln_r = int JdivR dtheta/(2pi)."""
        JdivR = self.geometry.JdivR
        return self.grid.integrate(JdivR, axis=1) / (2 * np.pi)

    @property
    def FF_r(self) -> np.ndarray:
        """Physical F*F' profile, model-side diagnostic."""
        return self.alpha1 * self.alpha2 * self.FFn_r

    @property
    def FFn_r(self) -> np.ndarray:
        return self.FFn_psin * self.psin_r

    @property
    def F2(self) -> np.ndarray:
        """Physical F^2 profile."""
        FF_int = self.grid.accumulate(self.FF_r)
        return (self.R0 * self.B0) ** 2 + 2.0 * (FF_int - FF_int[-1])

    @property
    def F(self) -> np.ndarray:
        """Poloidal current function F (R*B_phi)."""
        if np.any(self.F2 < 1e-6):
            raise ValueError("Negative F2 encountered, cannot compute F")
        return np.sqrt(self.F2)

    @property
    def P_r(self) -> np.ndarray:
        """Physical pressure gradient P', model-side diagnostic."""
        return self.alpha1 * self.alpha2 * self.Pn_r / MU0

    @property
    def Pn_r(self) -> np.ndarray:
        return self.Pn_psin * self.psin_r

    @property
    def P(self) -> np.ndarray:
        """Physical pressure profile P."""
        P_int = self.grid.accumulate(self.P_r)
        return P_int - P_int[-1]

    @property
    def beta_t(self) -> np.ndarray:
        """Toroidal beta beta_t = 2*mu0*<P> / B0^2."""
        P_avg = float(self.grid.integrate(self.P * self.V_r) / self.grid.integrate(self.V_r))
        return float(2.0 * MU0 * P_avg / self.B0**2)

    @property
    def Gn1(self) -> np.ndarray:
        """Normalized source term before alpha1 in the GS operator."""
        R, JdivR = self.geometry.R, self.geometry.JdivR
        return JdivR * (self.FFn_psin[:, None] + R**2 * self.Pn_psin[:, None])

    @property
    def Gn2(self) -> np.ndarray:
        """Normalized geometry term before alpha2 in the GS operator."""
        geometry = self.geometry
        return (
            geometry.gttdivJR * self.psin_rr[:, None]
            + (geometry.gttdivJR_r - geometry.grtdivJR_t) * self.psin_r[:, None]
        )

    @property
    def G(self) -> np.ndarray:
        """GS operator residual field G = alpha1 * Gn1 + alpha2 * Gn2."""
        return self.alpha1 * self.Gn1 + self.alpha2 * self.Gn2

    @property
    def Ip(self) -> np.ndarray:
        """Total plasma current Ip (Amps)."""
        return -self.alpha1 * self.grid.integrate(self.Gn1) / MU0

    @property
    def q(self) -> np.ndarray:
        """Safety factor q, model-side diagnostic."""
        with np.errstate(divide="ignore", invalid="ignore"):
            q = self.F * self.Ln_r / (self.alpha2 * self.psin_r)
        return _regularize_axis_linear_profile(q, self.rho)

    @property
    def s(self) -> np.ndarray:
        """Magnetic shear s, model-side diagnostic."""
        q_r = self.grid.differentiate(self.q)
        return self.rho * q_r / self.q

    @property
    def Itor(self) -> np.ndarray:
        """Toroidal current distribution I_tor(rho), model-side diagnostic."""
        return 2.0 * np.pi * self.Kn * self.alpha2 * self.psin_r / MU0

    @property
    def jtor(self) -> np.ndarray:
        """Toroidal current density j_phi, model-side diagnostic."""
        with np.errstate(divide="ignore", invalid="ignore"):
            jtor = (
                -self.alpha1
                / (MU0 * self.S_r)
                * (
                    2.0 * np.pi * self.FFn_psin * self.Ln_r
                    + self.V_r * self.Pn_psin / (2.0 * np.pi)
                )
            )
        return _regularize_axis_linear_profile(jtor, self.rho)

    @property
    def jpara(self) -> np.ndarray:
        """Parallel current density <j.B>/B0, model-side diagnostic."""
        F_r = self.grid.differentiate(self.F)
        term_r = (
            self.Kn_r * self.psin_r / self.F
            + self.Kn * self.psin_rr / self.F
            - self.Kn * self.psin_r * F_r / self.F**2
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            jpara = self.alpha2 / MU0 * self.F / self.Ln_r * term_r
        return _regularize_axis_linear_profile(jpara, self.rho)

    @property
    def jphi(self) -> np.ndarray:
        """Local toroidal current density j_phi(R, Z)."""
        R = self.geometry.R
        with np.errstate(divide="ignore", invalid="ignore"):
            jphi = (
                -self.alpha1 / (MU0 * R) * (self.FFn_psin[:, None] + R**2 * self.Pn_psin[:, None])
            )
        return _regularize_axis_linear_surface(jphi, self.rho)

    @property
    def Psi(self) -> np.ndarray:
        """Physical poloidal flux Psi."""
        return 2.0 * np.pi * self.alpha2 * self.psin

    @property
    def Phi(self) -> np.ndarray:
        """Toroidal flux Phi."""
        return 2.0 * np.pi * self.grid.accumulate(self.F * self.Ln_r)

    def plot(
        self,
        outpath: str | None = None,
        *,
        show: bool = False,
        plot_residual: bool = False,
        grid: Grid | None = None,
    ):
        """Render the legacy 6-panel summary figure for this equilibrium."""

        return _plot_equilibrium(
            self,
            outpath=outpath,
            show=show,
            plot_residual=plot_residual,
            grid=grid,
        )

    def compare(
        self,
        other: Self,
        outpath: str | Path | None = None,
        *,
        show: bool = False,
        label_ref: str = "reference",
        label_other: str = "current",
        grid: Grid | None = None,
    ) -> dict[str, float]:
        """Compare this equilibrium against another one."""

        return _compare_equilibrium(
            self,
            other,
            outpath=outpath,
            show=show,
            label_ref=label_ref,
            label_other=label_other,
            grid=grid,
        )

    def resample(
        self,
        grid: Grid,
    ) -> Self:
        """Interpolate the current equilibrium snapshot to a target grid."""
        return _build_resampled_equilibrium(
            self,
            grid=grid,
        )

    def to_geqdsk(
        self,
        *,
        R_range: tuple[float, float],
        Z_range: tuple[float, float],
        NR: int,
        NZ: int | None = None,
        header: str = "",
        limiter: np.ndarray | None = None,
        psi_axis: float = 0.0,
        psi_outside: float | None = None,
    ) -> Geqdsk:
        """Export a GEQDSK snapshot written in physical psi."""
        geometry = self.geometry
        R_nodes, Z_nodes, Rmin, Rmax, Zmin, Zmax = _build_geqdsk_rectilinear_grid(
            geometry,
            R_range=R_range,
            Z_range=Z_range,
            NR=NR,
            NZ=NZ,
        )
        psin_uniform = np.linspace(0.0, 1.0, int(NR), dtype=np.float64)
        psi_axis = float(psi_axis)
        psi_scale = float(self.alpha2)
        if abs(psi_scale) <= 1.0e-14:
            raise ValueError("alpha2 is zero")
        psi_bound = psi_axis + psi_scale
        psi_outside_value = psi_bound if psi_outside is None else float(psi_outside)
        boundary = np.column_stack((geometry.R[-1], geometry.Z[-1])).astype(np.float64, copy=False)
        limiter_points = _coerce_optional_point_array(limiter, name="limiter")

        geqdsk = Geqdsk(
            header=str(header),
            NR=int(NR),
            NZ=int(Z_nodes.size),
            R0=float(self.R0),
            Z0=float(self.Z0),
            Rmin=Rmin,
            Rmax=Rmax,
            Zmin=Zmin,
            Zmax=Zmax,
            boundary=boundary.copy(),
            limiter=limiter_points,
            Bt0=float(self.B0),
            Raxis=float(geometry.R[0, 0]),
            Zaxis=float(geometry.Z[0, 0]),
            Ip=float(self.Ip),
            psi_axis=psi_axis,
            psi_bound=psi_bound,
            F=_sample_profile_on_uniform_psin(self.psin, self.F, psin_uniform),
            P=_sample_profile_on_uniform_psin(self.psin, self.P, psin_uniform),
            FF_psi=_sample_profile_on_uniform_psin(
                self.psin, self.alpha1 * self.FFn_psin, psin_uniform
            ),
            P_psi=_sample_profile_on_uniform_psin(
                self.psin, self.alpha1 * self.Pn_psin / MU0, psin_uniform
            ),
            q=_sample_profile_on_uniform_psin(self.psin, self.q, psin_uniform),
            psi=_interpolate_psin_to_rectilinear_grid(
                geometry,
                self.psin,
                np.square(np.asarray(self.rho, dtype=np.float64)),
                R_nodes=R_nodes,
                Z_nodes=Z_nodes,
                psi_axis=psi_axis,
                psi_scale=psi_scale,
                psi_outside=psi_outside_value,
            ),
        )
        geqdsk.dR = float(R_nodes[1] - R_nodes[0]) if R_nodes.size > 1 else 0.0
        geqdsk.dZ = float(Z_nodes[1] - Z_nodes[0]) if Z_nodes.size > 1 else 0.0
        return geqdsk


def _normalize_shape_profiles(shape_profiles: dict[str, Profile]) -> dict[str, Profile]:
    if not isinstance(shape_profiles, dict):
        raise TypeError(
            f"shape_profiles must be dict[str, Profile], got {type(shape_profiles).__name__}"
        )
    for name, profile in shape_profiles.items():
        if not isinstance(name, str):
            raise TypeError(f"shape profile names must be str, got {type(name).__name__}")
        profile_type = type(profile)
        if not (
            isinstance(profile, Profile)
            or (
                profile_type.__name__ == Profile.__name__
                and getattr(profile_type, "__module__", None) == Profile.__module__
            )
        ):
            raise TypeError(f"shape profile {name!r} must be Profile, got {type(profile).__name__}")
    return {name: profile.copy() for name, profile in shape_profiles.items()}


def _build_default_shape_profile(name: str, grid: Grid) -> Profile:
    power = 0
    if name.startswith(("c", "s")) and name[1:].isdigit():
        power = int(grid.K_values[int(name[1:])])
    return Profile(scale=1.0, power=power, envelope_power=1, offset=0.0, coeff=None)


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
    color = _EXTRA_SHAPE_PROFILE_COLORS[
        sum(ord(ch) for ch in name) % len(_EXTRA_SHAPE_PROFILE_COLORS)
    ]
    return {"color": color, "label": label, **style}


def _plot_equilibrium(
    equilibrium: Equilibrium,
    outpath: str | Path | None = None,
    *,
    show: bool = False,
    plot_residual: bool = False,
    grid: Grid | None = None,
):
    """Render the legacy 6-panel equilibrium summary for one model-side equilibrium."""
    surface_equilibrium = _build_resampled_equilibrium(equilibrium, grid=grid)
    fig = _render_equilibrium_summary(
        surface_equilibrium=surface_equilibrium,
        profile_equilibrium=equilibrium,
        plot_residual=plot_residual,
    )

    if outpath is not None:
        fig.savefig(Path(outpath), dpi=300, facecolor="white")
    if show:
        plt.show()
    elif outpath is not None:
        plt.close(fig)

    return fig


def _compare_equilibrium(
    reference: Equilibrium,
    other: Self,
    outpath: str | Path | None = None,
    *,
    show: bool = False,
    label_ref: str = "reference",
    label_other: str = "current",
    grid: Grid | None = None,
) -> dict[str, float]:
    """Render a compact 3-column comparison figure with shared surface overlay."""
    compare_grid = grid or Grid(
        Nr=64,
        Nt=64,
        quadrature_scheme="uniform",
        L_max=max(reference.grid.L_max, other.grid.L_max),
        M_max=max(reference.grid.M_max, other.grid.M_max),
        K_max=reference.grid.K_max if reference.grid.K_max == other.grid.K_max else None,
    )
    ref_surface = _build_resampled_equilibrium(reference, grid=compare_grid)
    other_surface = _build_resampled_equilibrium(other, grid=compare_grid)

    shape_keys = [
        key
        for key in ["h", "k", "s1"]
        if key in reference.shape_profiles or key in other.shape_profiles
    ]
    source_groups = [
        ("psi_r", r"$\psi_\rho$", None),
        ("FF_psi", r"$FF_\psi$", None),
        ("mu0_P_psi", r"$\mu_0 P_\psi$", None),
    ]
    d1 = _build_comparison_profile_data(reference, shape_keys=shape_keys)
    d2 = _build_comparison_profile_data(other, shape_keys=shape_keys)

    errors: dict[str, float] = {}
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1.2, 0.9, 0.9],
        hspace=0.25,
        wspace=0.3,
        top=0.95,
        bottom=0.1,
        left=0.05,
        right=0.98,
    )

    ref_surface_data = _build_surface_panel_data(ref_surface)
    other_surface_data = _build_surface_panel_data(other_surface)
    shared_boundary = _merge_surface_boundaries(
        ref_surface_data["boundary"], other_surface_data["boundary"]
    )
    surface_ax = fig.add_subplot(gs[:, 0])
    _render_comparison_surface_overlay_panel(
        surface_ax,
        ref_surface_data,
        other_surface_data,
        shared_boundary,
        label_ref=label_ref,
        label_other=label_other,
    )

    shape_axes = [fig.add_subplot(gs[row, 1]) for row in range(3)]
    source_axes = [fig.add_subplot(gs[row, 2]) for row in range(3)]

    for i, (ax, key) in enumerate(zip(shape_axes, shape_keys, strict=True)):
        ylabel = _shape_profile_plot_meta(key)["label"]
        ref_values = np.asarray(d1[key], dtype=np.float64)
        cur_values = np.asarray(d2[key], dtype=np.float64)
        rel_max, rel_rms = _profile_errors_on_coarser_grid(
            d1["rho"],
            ref_values,
            d2["rho"],
            cur_values,
        )
        errors[f"rel_{key}_max"] = rel_max
        errors[f"rel_{key}_rms"] = rel_rms

        ax.plot(d1["rho"], d1[key], color=BLACK, linestyle="-", label=label_ref)
        ax.plot(d2["rho"], d2[key], color=RED, linestyle="--", label=label_other)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.5)
        _add_top_headroom(ax, 0.15)
        ax.text(
            0.03,
            0.97,
            f"err = {_format_profile_error(errors[f'rel_{key}_max'])}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        if i == 0:
            ax.set_title("(b) Shape Parameters", fontsize=SUBPLOT_TITLE_FONTSIZE)
            ax.legend(loc="best", frameon=False)
        if i == len(shape_keys) - 1:
            ax.set_xlabel(r"$\rho$")
        else:
            ax.set_xticklabels([])

    for i, (ax, (key, ylabel, scale)) in enumerate(zip(source_axes, source_groups, strict=True)):
        s = scale or 1.0
        ref_values = np.asarray(d1[key], dtype=np.float64)
        cur_values = np.asarray(d2[key], dtype=np.float64)
        rel_max, rel_rms = _profile_errors_on_coarser_grid(
            d1["rho"],
            ref_values,
            d2["rho"],
            cur_values,
        )
        errors[f"rel_{key}_max"] = rel_max
        errors[f"rel_{key}_rms"] = rel_rms

        ax.plot(d1["rho"], d1[key] / s, color=BLACK, linestyle="-", label=label_ref)
        ax.plot(d2["rho"], d2[key] / s, color=RED, linestyle="--", label=label_other)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.5)
        _add_top_headroom(ax, 0.15)
        ax.text(
            0.03,
            0.97,
            f"err = {_format_profile_error(errors[f'rel_{key}_max'])}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        if i == 0:
            ax.set_title("(c) Source Profiles", fontsize=SUBPLOT_TITLE_FONTSIZE)
        if i == len(source_groups) - 1:
            ax.set_xlabel(r"$\rho$")
        else:
            ax.set_xticklabels([])

    if outpath is not None:
        fig.savefig(Path(outpath), dpi=300, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return errors


def _profile_errors_on_coarser_grid(
    reference_rho: np.ndarray,
    reference_values: np.ndarray,
    current_rho: np.ndarray,
    current_values: np.ndarray,
) -> tuple[float, float]:
    """Return profile errors by interpolating only for error calculation.

    The plotted 1D profiles stay on their native ``Equilibrium`` grids.  When
    the two grids differ, compare on the coarser grid so diagnostics remain
    available without densifying or mutating either profile.
    """

    reference_rho = np.asarray(reference_rho, dtype=np.float64)
    current_rho = np.asarray(current_rho, dtype=np.float64)
    reference_values = np.asarray(reference_values, dtype=np.float64)
    current_values = np.asarray(current_values, dtype=np.float64)
    if reference_rho.ndim != 1 or current_rho.ndim != 1:
        raise ValueError("Expected 1D rho grids for profile comparison")
    if reference_values.shape != reference_rho.shape:
        raise ValueError(
            f"reference values/rho shape mismatch: "
            f"{reference_values.shape} vs {reference_rho.shape}"
        )
    if current_values.shape != current_rho.shape:
        raise ValueError(
            f"current values/rho shape mismatch: {current_values.shape} vs {current_rho.shape}"
        )

    if reference_rho.size <= current_rho.size:
        target_rho = reference_rho
        reference_on_target = reference_values
        current_on_target = _resample_profile_linear(current_rho, current_values, target_rho)
    else:
        target_rho = current_rho
        reference_on_target = _resample_profile_linear(reference_rho, reference_values, target_rho)
        current_on_target = current_values

    scale_ref = float(np.max(np.abs(reference_on_target))) or 1.0
    diff = current_on_target - reference_on_target
    return (
        float(np.max(np.abs(diff)) / scale_ref),
        float(np.sqrt(np.mean(diff**2)) / scale_ref),
    )


def _format_profile_error(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1e}"


def _build_resampled_equilibrium(
    equilibrium: Equilibrium,
    *,
    grid: Grid | None,
) -> Equilibrium:
    source_grid = equilibrium.grid
    plot_grid = grid or Grid(
        Nr=64,
        Nt=64,
        quadrature_scheme="uniform",
        L_max=source_grid.L_max,
        M_max=source_grid.M_max,
        K_max=source_grid.K_max,
    )

    psin_r = _resample_profile_linear(
        source_grid.rho,
        np.asarray(equilibrium.psin_r, dtype=np.float64),
        plot_grid.rho,
        left=0.0,
    )
    psin = _resample_profile_linear(
        source_grid.rho,
        np.asarray(equilibrium.psin, dtype=np.float64),
        plot_grid.rho,
    )
    FFn_psin = _resample_profile_linear(
        source_grid.rho,
        np.asarray(equilibrium.FFn_psin, dtype=np.float64),
        plot_grid.rho,
        right=0.0,
    )
    Pn_psin = _resample_profile_linear(
        source_grid.rho,
        np.asarray(equilibrium.Pn_psin, dtype=np.float64),
        plot_grid.rho,
        right=0.0,
    )

    shape_profiles: dict[str, Profile] = {}
    for name, profile in equilibrium.shape_profiles.items():
        copied = profile.copy()
        copied.update(grid=plot_grid)
        shape_profiles[name] = copied

    return Equilibrium(
        R0=equilibrium.R0,
        Z0=equilibrium.Z0,
        B0=equilibrium.B0,
        a=equilibrium.a,
        grid=plot_grid,
        shape_profiles=shape_profiles,
        psin=psin,
        FFn_psin=FFn_psin,
        Pn_psin=Pn_psin,
        psin_r=psin_r,
        psin_rr=plot_grid.differentiate(psin_r),
        alpha1=equilibrium.alpha1,
        alpha2=equilibrium.alpha2,
    )


def _build_comparison_profile_data(
    equilibrium: Equilibrium,
    *,
    shape_keys: list[str],
) -> dict[str, np.ndarray]:
    data = {
        "rho": np.asarray(equilibrium.rho, dtype=np.float64),
        "psi_r": np.asarray(equilibrium.alpha2 * equilibrium.psin_r, dtype=np.float64),
        "FF_psi": np.asarray(equilibrium.alpha1 * equilibrium.FFn_psin, dtype=np.float64),
        "mu0_P_psi": np.asarray(equilibrium.alpha1 * equilibrium.Pn_psin, dtype=np.float64),
    }
    for key in shape_keys:
        profile = equilibrium.shape_profiles.get(key)
        if profile is None:
            data[key] = np.zeros_like(equilibrium.rho, dtype=np.float64)
        else:
            data[key] = np.asarray(profile.u, dtype=np.float64)
    return data


def _render_equilibrium_summary(
    *,
    surface_equilibrium: Equilibrium,
    profile_equilibrium: Equilibrium | None = None,
    plot_residual: bool = False,
):
    if profile_equilibrium is None:
        profile_equilibrium = surface_equilibrium

    if plot_residual:
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
    else:
        fig = plt.figure(figsize=(19.5, 6.5))
        gs = GridSpec(
            2,
            7,
            figure=fig,
            width_ratios=[1.05, 0.35, 1.28, 0.25, 1.05, 0.35, 1.28],
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
    _render_panel_a_surfaces(fig.add_subplot(gs[:, 0]), fig, panel_a)
    _render_panel_b_shapes(fig.add_subplot(gs[0, 2]), panel_b)
    _render_panel_c_sources(fig.add_subplot(gs[1, 2]), panel_c)
    _render_panel_d_jphi(fig.add_subplot(gs[:, 4]), fig, panel_d, panel_a["boundary"])
    _render_panel_e_current_1d(fig.add_subplot(gs[0, 6]), panel_e)
    _render_panel_f_safety(fig.add_subplot(gs[1, 6]), panel_f)
    if plot_residual:
        panel_g = _build_gs_residual_panel_data(surface_equilibrium)
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


def _close_periodic_curve(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D periodic curve, got {arr.shape}")
    return np.concatenate([arr, arr[:1]])


def _merge_surface_boundaries(*boundaries: dict) -> dict:
    if not boundaries:
        raise ValueError("At least one boundary must be provided")
    R = np.concatenate([np.asarray(boundary["R"], dtype=np.float64) for boundary in boundaries])
    Z = np.concatenate([np.asarray(boundary["Z"], dtype=np.float64) for boundary in boundaries])
    return {"R": R, "Z": Z}


def _build_shape_panel_data(equilibrium: Equilibrium) -> dict:
    values = {
        key: profile.u
        for key, profile in equilibrium.shape_profiles.items()
        if _include_shape_panel_profile(key)
    }
    return {"shape": {"rho": equilibrium.rho, "values": values}}


def _include_shape_panel_profile(name: str) -> bool:
    if name in SHAPE_PROFILE_PLOT_META:
        return True
    if name.startswith("c") and name[1:].isdigit():
        return int(name[1:]) <= 2
    if name.startswith("s") and name[1:].isdigit():
        return int(name[1:]) <= 3
    return False


def _build_source_panel_data(equilibrium: Equilibrium) -> dict:
    return {
        "rho": equilibrium.rho,
        "psi_r": equilibrium.alpha2 * equilibrium.psin_r.copy(),
        "FF_psi": equilibrium.alpha1 * equilibrium.FFn_psin.copy(),
        "mu0_P_psi": equilibrium.alpha1 * equilibrium.Pn_psin.copy(),
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
        raise ValueError("Expected 1D rho_src/y_src/rho_eval with matching source shape")
    left_val = float(y_src[0]) if left is None else float(left)
    right_val = float(y_src[-1]) if right is None else float(right)
    return np.interp(rho_eval, rho_src, y_src, left=left_val, right=right_val)


def _build_geqdsk_rectilinear_grid(
    geometry: Geometry,
    *,
    R_range: tuple[float, float],
    Z_range: tuple[float, float],
    NR: int,
    NZ: int | None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    NR = int(NR)
    if NR < 2:
        raise ValueError(f"NR must be at least 2, got {NR}")

    Zmin, Zmax = map(float, Z_range)
    Rmin, Rmax = map(float, R_range)

    if not np.isfinite(Rmin) or not np.isfinite(Rmax) or Rmax <= Rmin:
        raise ValueError(f"R_range must be finite and increasing, got {R_range!r}")

    if not np.isfinite(Zmin) or not np.isfinite(Zmax) or Zmax <= Zmin:
        raise ValueError(f"Z_range must be finite and increasing, got {Z_range!r}")

    if NZ is None:
        NZ = NR

    if NZ < 2:
        raise ValueError(f"NZ must be at least 2, got {NZ}")

    R_nodes = np.linspace(Rmin, Rmax, NR, dtype=np.float64)
    Z_nodes = np.linspace(Zmin, Zmax, NZ, dtype=np.float64)
    return R_nodes, Z_nodes, Rmin, Rmax, Zmin, Zmax


def _prepare_profile_interp_axis(
    psin_src: np.ndarray, values_src: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    psin_arr = np.asarray(psin_src, dtype=np.float64)
    values_arr = np.asarray(values_src, dtype=np.float64)
    if psin_arr.ndim != 1 or values_arr.ndim != 1 or psin_arr.shape != values_arr.shape:
        raise ValueError("psin_src and values_src must be 1D arrays with the same shape")

    order = np.argsort(psin_arr)
    psin_sorted = psin_arr[order]
    values_sorted = values_arr[order]
    psin_unique, unique_indices = np.unique(psin_sorted, return_index=True)
    values_unique = values_sorted[unique_indices]
    if psin_unique.size < 2:
        raise ValueError("Need at least two distinct psin samples to export Geqdsk profiles")
    return psin_unique, values_unique


def _sample_profile_on_uniform_psin(
    psin_src: np.ndarray,
    values_src: np.ndarray,
    psin_eval: np.ndarray,
) -> np.ndarray:
    psin_axis, values_axis = _prepare_profile_interp_axis(psin_src, values_src)
    psin_eval = np.asarray(psin_eval, dtype=np.float64)
    return np.interp(
        psin_eval, psin_axis, values_axis, left=float(values_axis[0]), right=float(values_axis[-1])
    )


def _interpolate_psin_to_rectilinear_grid(
    geometry: Geometry,
    psin: np.ndarray,
    rho2_src: np.ndarray,
    *,
    R_nodes: np.ndarray,
    Z_nodes: np.ndarray,
    psi_axis: float,
    psi_scale: float,
    psi_outside: float,
) -> np.ndarray:
    R_surfaces = np.asarray(geometry.R, dtype=np.float64)
    Z_surfaces = np.asarray(geometry.Z, dtype=np.float64)
    psin = np.asarray(psin, dtype=np.float64)
    rho2_src = np.asarray(rho2_src, dtype=np.float64)
    if R_surfaces.shape != Z_surfaces.shape:
        raise ValueError(f"Geometry R/Z shape mismatch: {R_surfaces.shape} vs {Z_surfaces.shape}")
    if psin.ndim != 1 or psin.shape[0] != R_surfaces.shape[0]:
        raise ValueError(f"psin must have shape ({R_surfaces.shape[0]},), got {psin.shape}")
    if rho2_src.ndim != 1 or rho2_src.shape[0] != R_surfaces.shape[0]:
        raise ValueError(f"rho2_src must have shape ({R_surfaces.shape[0]},), got {rho2_src.shape}")

    R_grid, Z_grid = np.meshgrid(R_nodes, Z_nodes, indexing="ij")
    rho2_grid = _interpolate_rho2_to_rectilinear_grid(
        R_surfaces,
        Z_surfaces,
        rho2_src,
        R_grid,
        Z_grid,
    )

    psi_grid = np.full(R_grid.shape, float(psi_outside), dtype=np.float64)
    inside = np.isfinite(rho2_grid)
    if np.any(inside):
        psi_grid[inside] = float(psi_axis) + float(psi_scale) * np.interp(
            rho2_grid[inside], rho2_src, psin
        )
    return psi_grid


def _interpolate_rho2_to_rectilinear_grid(
    R_surfaces: np.ndarray,
    Z_surfaces: np.ndarray,
    rho2_surfaces: np.ndarray,
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
) -> np.ndarray:
    if R_surfaces.ndim != 2 or Z_surfaces.ndim != 2 or R_surfaces.shape != Z_surfaces.shape:
        raise ValueError(
            f"Expected R_surfaces/Z_surfaces to share a 2D shape, "
            f"got {R_surfaces.shape} and {Z_surfaces.shape}"
        )
    if rho2_surfaces.ndim != 1 or rho2_surfaces.shape[0] != R_surfaces.shape[0]:
        raise ValueError(
            f"rho2_surfaces must have shape ({R_surfaces.shape[0]},), got {rho2_surfaces.shape}"
        )

    points_R, points_Z, point_values, triangles = _build_flux_mesh_triangulation(
        R_surfaces, Z_surfaces, rho2_surfaces
    )
    triangle_mask = _build_degenerate_triangle_mask(points_R, points_Z, triangles)
    rho2_grid = np.full(R_grid.shape, np.nan, dtype=np.float64)
    R_nodes = np.asarray(R_grid[:, 0], dtype=np.float64)
    Z_nodes = np.asarray(Z_grid[0, :], dtype=np.float64)

    for tri, masked in zip(triangles, triangle_mask, strict=True):
        if bool(masked):
            continue
        _rasterize_triangle_to_grid(
            rho2_grid,
            R_grid,
            Z_grid,
            R_nodes,
            Z_nodes,
            points_R[tri],
            points_Z[tri],
            point_values[tri],
        )
    return rho2_grid


def _build_flux_mesh_triangulation(
    R_surfaces: np.ndarray,
    Z_surfaces: np.ndarray,
    rho2_surfaces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nr, nt = R_surfaces.shape
    if nr < 2 or nt < 3:
        raise ValueError(f"Need at least a 2x3 flux mesh, got {(nr, nt)}")

    point_count = 1 + (nr - 1) * nt
    points_R = np.empty(point_count, dtype=np.float64)
    points_Z = np.empty(point_count, dtype=np.float64)
    point_values = np.empty(point_count, dtype=np.float64)

    points_R[0] = float(R_surfaces[0, 0])
    points_Z[0] = float(Z_surfaces[0, 0])
    point_values[0] = float(rho2_surfaces[0])

    for i in range(1, nr):
        start = 1 + (i - 1) * nt
        end = start + nt
        points_R[start:end] = R_surfaces[i]
        points_Z[start:end] = Z_surfaces[i]
        point_values[start:end] = float(rho2_surfaces[i])

    triangle_count = nt + 2 * (nr - 2) * nt
    triangles = np.empty((triangle_count, 3), dtype=np.int32)
    cursor = 0

    def vertex_index(i: int, j: int) -> int:
        if i == 0:
            return 0
        return 1 + (i - 1) * nt + (j % nt)

    for j in range(nt):
        triangles[cursor] = [vertex_index(0, 0), vertex_index(1, j), vertex_index(1, j + 1)]
        cursor += 1

    for i in range(1, nr - 1):
        for j in range(nt):
            triangles[cursor] = [
                vertex_index(i, j),
                vertex_index(i + 1, j),
                vertex_index(i + 1, j + 1),
            ]
            cursor += 1
            triangles[cursor] = [
                vertex_index(i, j),
                vertex_index(i + 1, j + 1),
                vertex_index(i, j + 1),
            ]
            cursor += 1

    return points_R, points_Z, point_values, triangles


def _build_degenerate_triangle_mask(
    points_R: np.ndarray,
    points_Z: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    p0 = triangles[:, 0]
    p1 = triangles[:, 1]
    p2 = triangles[:, 2]
    twice_area = (points_R[p1] - points_R[p0]) * (points_Z[p2] - points_Z[p0]) - (
        points_R[p2] - points_R[p0]
    ) * (points_Z[p1] - points_Z[p0])
    scale = np.maximum(
        np.maximum(np.abs(points_R[p0]), np.abs(points_R[p1])),
        np.maximum(np.abs(points_Z[p0]), np.abs(points_Z[p1])),
    )
    scale = np.maximum(scale, 1.0)
    return np.abs(twice_area) <= 1.0e-14 * scale * scale


def _rasterize_triangle_to_grid(
    rho2_grid: np.ndarray,
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    R_nodes: np.ndarray,
    Z_nodes: np.ndarray,
    tri_R: np.ndarray,
    tri_Z: np.ndarray,
    tri_values: np.ndarray,
) -> None:
    r_min = float(np.min(tri_R))
    r_max = float(np.max(tri_R))
    z_min = float(np.min(tri_Z))
    z_max = float(np.max(tri_Z))
    i0 = int(np.searchsorted(R_nodes, r_min, side="left"))
    i1 = int(np.searchsorted(R_nodes, r_max, side="right"))
    j0 = int(np.searchsorted(Z_nodes, z_min, side="left"))
    j1 = int(np.searchsorted(Z_nodes, z_max, side="right"))
    if i0 >= i1 or j0 >= j1:
        return

    x0, x1, x2 = map(float, tri_R)
    y0, y1, y2 = map(float, tri_Z)
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) <= 1.0e-20:
        return

    sub_R = R_grid[i0:i1, j0:j1]
    sub_Z = Z_grid[i0:i1, j0:j1]
    l0 = ((y1 - y2) * (sub_R - x2) + (x2 - x1) * (sub_Z - y2)) / denom
    l1 = ((y2 - y0) * (sub_R - x2) + (x0 - x2) * (sub_Z - y2)) / denom
    l2 = 1.0 - l0 - l1
    inside = (l0 >= -1.0e-12) & (l1 >= -1.0e-12) & (l2 >= -1.0e-12)
    if not np.any(inside):
        return

    values = l0 * float(tri_values[0]) + l1 * float(tri_values[1]) + l2 * float(tri_values[2])
    target = rho2_grid[i0:i1, j0:j1]
    target[inside] = values[inside]


def _coerce_optional_point_array(value, *, name: str) -> np.ndarray:
    arr = np.asarray(
        value if value is not None else np.empty((0, 2), dtype=np.float64), dtype=np.float64
    )
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {arr.shape}")
    return arr.copy()


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
    return mcolors.LinearSegmentedColormap.from_list(
        "trunc_inferno", cmap(np.linspace(0.15, 0.92, 256))
    )


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
    sm = plt.cm.ScalarMappable(
        cmap=_get_trunc_inferno(), norm=mcolors.Normalize(vmin=0.0, vmax=1.0)
    )
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

    ax.set(xlabel=r"$\rho$", ylabel=Profile)
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
        ax.plot(rho, data["FF_psi"], "-o", color=RED, label=r"$FF_\psi$")
        ax.plot(rho, data["mu0_P_psi"], "-o", color=PURPLE, label=r"$\mu_0 P_\psi$")
    else:
        ax.plot(rho, data["psi_r"], "--", color=BLACK, label=r"$\psi_\rho$")
        ax.plot(rho, data["FF_psi"], "-", color=RED, label=r"$FF_\psi$")
        ax.plot(rho, data["mu0_P_psi"], "-", color=PURPLE, label=r"$\mu_0 P_\psi$")

    ax.set(xlabel=r"$\rho$", ylabel=Profile)
    _add_top_headroom(ax, ratio=0.35)
    ax.legend(loc="upper left")
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
    pcm = ax.contourf(
        R_plot, Z_plot, j_plot, levels=np.linspace(vmin, vmax, 128), cmap=cmap, norm=norm
    )
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
    pcm = ax.contourf(
        R_plot, Z_plot, G_plot, levels=np.linspace(vmin, vmax, 129), cmap=cmap, norm=norm
    )
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
    ax.set(xlabel=r"$\rho$", ylabel=Profile)
    ax.legend(loc="upper left")
    ax.grid(True)
