"""
Module: model.geqdsk

Role:
- 负责读取, 持有, 插值与绘制 GEQDSK 风格平衡数据.

Public API:
- Geqdsk

Notes:
- 这个文件偏 legacy utility.
- 当前注释优先保留业务可读性, 暂不按 runtime hot-path 风格重写内部实现.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from scipy.integrate import simpson
from scipy.interpolate import interp1d, splev, splprep
from scipy.optimize import least_squares
from shapely.geometry import Point, Polygon
from units import base

from veqpy.model.serial import Serial


class Geqdsk(Serial):
    def __init__(self, path=None):
        # 网格
        self.nr = self.nz = 0
        self.R0 = self.Z0 = self.dr = self.dz = 0.0
        self.Rmin = self.Rmax = self.Zmin = self.Zmax = 0.0

        # 边界
        self.boundary = self.limiter = np.empty((0, 2), dtype=float)

        # 0-D 参数
        self.Bt0 = self.Raxis = self.Zaxis = self.I_total = 0.0
        self.psi_axis = self.psi_bound = 0.0

        # 1-D 参数
        self.f = self.p = self.fdf = self.dp = np.empty(0, dtype=float)
        self.q = self.phi = self.rho = self.xi = np.empty(0, dtype=float)

        # 2-D 参数
        self.psi = np.empty((0, 0), dtype=float)

        # 读取
        if path:
            self.read(path)

    def read(self, path):
        """从文件读取 UniTS.Geqdsk"""

        if isinstance(path, dict):
            self.from_dict(path)

        elif isinstance(path, str):
            if path.endswith(".json"):
                self.from_json(path)
            else:
                self.from_chease(path)

        else:
            raise ValueError("Input must be a path or dict")

    def write(self, path=None):
        """向文件写入 UniTS.geometry"""

        if path is None:
            return self.to_dict()

        elif isinstance(path, str):
            if path.endswith(".json"):
                self.to_json(path)
            else:
                raise ValueError("Output must be a JSON file")

        else:
            raise ValueError("Output must be a path")

    def from_chease(self, chease):
        """从 CHEASE 文件读取 UniTS.Geqdsk"""

        with open(chease, "r") as file:
            self._read_header(file)
            self._read_geometry(file)
            self._read_axfig_and_current(file)
            self._read_profile(file)

    def interp(self, mesh=51, mode="linear"):
        """
        该函数用于将 1-D 参数插值到新的 xi 网格上
        - mesh:         xi 网格，可以是整数或浮点数列表
        - mode:         插值模式 'linear'（线性）、'cubic'（三次样条）
        """

        self.xi = base.get_mesh(mesh)
        for attr in ["f", "fdf", "p", "dp", "q", "phi", "rho"]:
            self._interp_and_update(attr, mode)

    def plot(
        self,
        contour_value=True,
        fig_show=True,
        fig_save=False,
    ):
        if not fig_show and not fig_save:
            raise ValueError("At least one of fig_show or fig_save must be True")

        plt.rcParams["figure.figsize"] = (8, 6)

        # 创建网格
        R = np.linspace(self.Rmin, self.Rmax, self.nr)
        Z = np.linspace(self.Zmin, self.Zmax, self.nz)
        R2D, Z2D = np.meshgrid(R, Z)

        # 创建图形并设置标题
        fig = plt.figure()

        # 绘制 psi 和边界、限制器
        ax1 = fig.add_subplot(121)
        contf = ax1.contourf(R2D, Z2D, self.psi, 40, cmap="jet", alpha=0.25)
        cont = ax1.contour(R2D, Z2D, self.psi, 20, cmap="jet")

        if contour_value:
            ax1.clabel(cont, inline=True, fontsize=6)

        ax1.plot(
            self.boundary[:, 0],
            self.boundary[:, 1],
            "--",
            color="#ff7f0e",
            linewidth=2,
            label="Boundary",
        )
        ax1.plot(
            self.limiter[:, 0],
            self.limiter[:, 1],
            color="#000000",
            linewidth=2,
            label="Limiter",
        )

        ax1.set_aspect("equal")
        ax1.set_xlabel("$R$ (m)")
        ax1.set_ylabel("$Z$ (m)")
        ax1.legend(loc="upper right")
        fig.colorbar(contf, ax=ax1)

        # 绘制 q
        ax2 = fig.add_subplot(222)
        self._plot_profile(ax2, self.q, "$q$", r"$\psi$ (normalized)", color="#2a7dc1")

        # 绘制 p
        ax3 = fig.add_subplot(224)
        self._plot_profile(
            ax3,
            self.p,
            "$P$ (Pa)",
            r"$\psi$ (normalized)",
            color="#ff7f0e",
            sci=True,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if fig_save:
            plt.savefig("Geqdsk.png", dpi=500)
            print("Saved as Geqdsk.png")

        if fig_show:
            plt.show()
        else:
            plt.close()

    def contour(
        self,
        mesh=51,
        count=lambda xi: int(100 * np.sqrt(xi) + 20),
        method="mask",
        fig_show=True,
        fig_save=False,
    ):
        """
        该函数用于绘制等值线图，并输出等离子体边界内的等势点集合
        - mesh:         xi 网格，可以是整数或浮点数列表
        - count:        xi 网格上的点数，可以是整数或函数
        - method:       筛选方法 'polygon'（多边形）、'mask'（掩膜法）
        - fig_show:     是否显示图像
        - fig_save:     是否保存图像到 Geqdsk.png
        """

        # 通过 mesh 参数获得网格
        mesh = base.get_mesh(mesh)

        # 通过 count 参数获得函数
        if isinstance(count, int):

            def count(xi):
                return count
        elif not callable(count):
            raise ValueError("Num must be an integer or a function")

        # 通过 method 参数获得方法
        if method in ["polygon", "p"]:
            method = "p"
        elif method in ["mask", "m"]:
            method = "m"
        else:
            raise ValueError("Method must be 'polygon', 'mask' or their abbreviations")

        psi1D = np.linspace(self.psi_axis, self.psi_bound - 0.01, self.nr)
        psi1D_mesh = interp1d(self.xi, psi1D, kind="linear")(mesh)

        # 生成网格 (X, Y)
        R = np.linspace(self.Rmin, self.Rmax, self.nr)
        Z = np.linspace(self.Zmin, self.Zmax, self.nz)
        X, Y = np.meshgrid(R, Z)

        # 创建边界的 Path 对象
        if method == "p":
            boundary_path = Polygon(self.boundary)
        elif method == "m":
            boundary_path = Path(self.boundary)  # 将边界转换为 Path 对象

        plt.rcParams["figure.figsize"] = (6, 6)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        contours = ax.contour(
            X,
            Y,
            self.psi,
            levels=psi1D_mesh[1:] if mesh[0] == 0 else psi1D_mesh,
            linewidths=0.75,
        )

        if mesh[0] == 0:
            all_points = [np.array([[self.Raxis, self.Zaxis]])]
        else:
            all_points = []

        # 收集每个磁面上的点（边界内）
        for i, contour in enumerate(contours.collections):
            points = []
            for path in contour.get_paths():
                path_points = path.vertices

                if method == "p":
                    # 多边形逐点检查
                    for point in path_points:
                        if boundary_path.contains(Point(point)):
                            points.append(point)
                else:
                    # 掩膜法
                    mask = boundary_path.contains_points(path_points)
                    points.extend(path_points[mask])

            points = np.array(points)

            # 根据 count 参数获得每个磁面上的点数
            number = count(mesh[i])

            if len(points) < number:
                tck, u = splprep(points.T, s=0, per=True)
                u_new = np.linspace(u.min(), u.max(), number)
                points = np.column_stack(splev(u_new, tck))
            elif len(points) > number:
                indices = np.round(np.linspace(0, len(points) - 1, number)).astype(int)
                points = points[indices]

            all_points.append(points)

        # ax.clear()

        # 绘制等值线（多边形内点）
        if fig_save or fig_show:
            if all_points:
                for points in all_points:
                    ax.plot(
                        points[:, 0],
                        points[:, 1],
                        color="r",
                        alpha=0.75,
                        zorder=10,
                        linewidth=0.5,
                    )
            if mesh[0] == 0:
                ax.plot(self.Raxis, self.Zaxis, "ro", markersize=3)

            ax.set_xlabel("R (m)")
            ax.set_ylabel("Z (m)")

            if fig_save:
                plt.savefig("Geqdsk.png", dpi=300)
                print("Saved as Geqdsk.png")
            if fig_show:
                plt.show()

        if not fig_show:
            plt.close()

        return all_points


    def boundary_shape_params(self, *, R0=None, Z0=None, a=None):
        """
        从 LCFS 边界点估计边界形状参数 (ka, c0a, c1a, s1a, s2a).

        Notes
        -----
        模型采用
        R = R0 + a * cos(theta_bar),
        Z = Z0 - a * ka * sin(theta),
        theta_bar = theta + c0a + c1a*cos(theta) + s1a*sin(theta) + s2a*sin(2*theta).

        该实现使用几何初值 + 最小二乘拟合, 适合从 GEQDSK 边界直接反推
        operator case 所需的 edge 参数。
        """

        if self.boundary.size == 0:
            raise ValueError("Boundary is empty. Read GEQDSK first.")

        R = self.boundary[:, 0].astype(float)
        Z = self.boundary[:, 1].astype(float)

        r_min = float(np.nanmin(R))
        r_max = float(np.nanmax(R))
        z_min = float(np.nanmin(Z))
        z_max = float(np.nanmax(Z))
        r_mid = 0.5 * (r_max + r_min)
        z_mid = 0.5 * (z_max + z_min)
        span_r = r_max - r_min
        span_z = z_max - z_min

        initial_R0 = float(R0) if R0 is not None else float(self.R0 if np.isfinite(self.R0) else r_mid)
        initial_Z0 = float(Z0) if Z0 is not None else float(self.Z0 if np.isfinite(self.Z0) else z_mid)
        initial_a = float(a) if a is not None else 0.5 * span_r

        if initial_a <= 0:
            raise ValueError("a must be positive")

        ka0 = float(0.5 * span_z / initial_a)
        ka0 = max(ka0, 1e-6)

        def ordered_boundary_variants():
            start = int(np.argmin(Z))
            r_ordered = np.roll(R, -start)
            z_ordered = np.roll(Z, -start)
            yield r_ordered, z_ordered
            yield np.concatenate(([r_ordered[0]], r_ordered[:0:-1])), np.concatenate(([z_ordered[0]], z_ordered[:0:-1]))

        def infer_theta(z_points, z0, a_value, ka):
            sin_theta = np.clip(-(z_points - z0) / (a_value * max(float(ka), 1e-6)), -1.0, 1.0)
            theta = np.empty_like(sin_theta)
            theta[0] = 0.5 * np.pi
            previous = theta[0]
            step = 2.0 * np.pi / max(len(z_points), 1)

            for index in range(1, len(z_points)):
                alpha = np.arcsin(sin_theta[index])
                candidates = []
                for candidate in (alpha, np.pi - alpha):
                    while candidate < previous - 1e-12:
                        candidate += 2.0 * np.pi
                    candidates.extend((candidate, candidate + 2.0 * np.pi))
                target = previous + step
                theta[index] = min(candidates, key=lambda value: abs(value - target))
                previous = theta[index]

            return theta

        def estimate_phase_coeffs(theta, r_points, r0, a_value):
            x = np.clip((r_points - r0) / a_value, -1.0, 1.0)
            base = np.arccos(x)
            theta_bar = np.empty_like(base)
            theta_bar[0] = 0.5 * np.pi
            previous = theta_bar[0]
            step = 2.0 * np.pi / max(len(r_points), 1)

            for index in range(1, len(r_points)):
                candidates = []
                for candidate in (base[index], -base[index]):
                    while candidate < previous - np.pi:
                        candidate += 2.0 * np.pi
                    while candidate > previous + np.pi:
                        candidate -= 2.0 * np.pi
                    candidates.extend((candidate - 2.0 * np.pi, candidate, candidate + 2.0 * np.pi))
                target = previous + step
                theta_bar[index] = min(candidates, key=lambda value: abs(value - target))
                previous = theta_bar[index]

            dphi = np.unwrap(theta_bar - theta)
            basis = np.column_stack((
                np.ones_like(theta),
                np.cos(theta),
                np.sin(theta),
                np.sin(2.0 * theta),
            ))
            return np.linalg.lstsq(basis, dphi, rcond=None)[0]

        base_params = {
            "R0": initial_R0,
            "Z0": initial_Z0,
            "a": initial_a,
            "ka": ka0,
            "c0a": 0.0,
            "c1a": 0.0,
            "s1a": 0.0,
            "s2a": 0.0,
        }

        free_names = []
        if R0 is None:
            free_names.append("R0")
        if Z0 is None:
            free_names.append("Z0")
        if a is None:
            free_names.append("a")
        free_names.extend(["ka", "c0a", "c1a", "s1a", "s2a"])

        lower_bounds_by_name = {
            "R0": r_min - 0.25 * span_r,
            "Z0": z_min - 0.25 * span_z,
            "a": max(1e-6, 0.25 * initial_a),
            "ka": 1e-6,
            "c0a": -10.0,
            "c1a": -10.0,
            "s1a": -10.0,
            "s2a": -10.0,
        }
        upper_bounds_by_name = {
            "R0": r_max + 0.25 * span_r,
            "Z0": z_max + 0.25 * span_z,
            "a": max(4.0 * initial_a, span_z, 1.0),
            "ka": 10.0,
            "c0a": 10.0,
            "c1a": 10.0,
            "s1a": 10.0,
            "s2a": 10.0,
        }

        def pack_params(params):
            return np.array([params[name] for name in free_names], dtype=float)

        def unpack_params(vector):
            params = dict(base_params)
            for name, value in zip(free_names, vector):
                params[name] = float(value)
            return params

        best_fit = None
        for r_points, z_points in ordered_boundary_variants():
            theta0 = infer_theta(z_points, initial_Z0, initial_a, ka0)
            c0_init, c1_init, s1_init, s2_init = estimate_phase_coeffs(theta0, r_points, initial_R0, initial_a)
            starts = [
                {
                    "R0": initial_R0,
                    "Z0": initial_Z0,
                    "a": initial_a,
                    "ka": ka0,
                    "c0a": c0_init,
                    "c1a": c1_init,
                    "s1a": s1_init,
                    "s2a": s2_init,
                },
                {
                    "R0": r_mid,
                    "Z0": z_mid,
                    "a": 0.5 * span_r,
                    "ka": max(0.5 * span_z / max(0.5 * span_r, 1e-6), 1e-6),
                    "c0a": 0.0,
                    "c1a": 0.0,
                    "s1a": 0.0,
                    "s2a": 0.0,
                },
                {
                    "R0": initial_R0,
                    "Z0": initial_Z0,
                    "a": initial_a * 0.9,
                    "ka": ka0,
                    "c0a": 0.0,
                    "c1a": 0.0,
                    "s1a": 0.0,
                    "s2a": 0.0,
                },
                {
                    "R0": initial_R0,
                    "Z0": initial_Z0,
                    "a": initial_a * 1.1,
                    "ka": ka0,
                    "c0a": 0.0,
                    "c1a": 0.0,
                    "s1a": 0.0,
                    "s2a": 0.0,
                },
                {
                    "R0": initial_R0,
                    "Z0": initial_Z0,
                    "a": initial_a,
                    "ka": ka0 * 0.9,
                    "c0a": 0.0,
                    "c1a": 0.2,
                    "s1a": 0.0,
                    "s2a": 0.0,
                },
                {
                    "R0": initial_R0,
                    "Z0": initial_Z0,
                    "a": initial_a,
                    "ka": ka0 * 1.1,
                    "c0a": 0.0,
                    "c1a": -0.2,
                    "s1a": 0.0,
                    "s2a": 0.0,
                },
                {
                    "R0": r_mid,
                    "Z0": z_mid,
                    "a": initial_a * 0.85,
                    "ka": ka0 * 1.1,
                    "c0a": 0.1,
                    "c1a": 0.2,
                    "s1a": 0.0,
                    "s2a": 0.0,
                },
                {
                    "R0": r_mid,
                    "Z0": z_mid,
                    "a": initial_a * 1.15,
                    "ka": ka0 * 0.9,
                    "c0a": -0.1,
                    "c1a": -0.2,
                    "s1a": 0.0,
                    "s2a": 0.0,
                },
            ]

            def residual(vector):
                params = unpack_params(vector)
                theta = infer_theta(z_points, params["Z0"], params["a"], params["ka"])
                tb = (
                    theta
                    + params["c0a"]
                    + params["c1a"] * np.cos(theta)
                    + params["s1a"] * np.sin(theta)
                    + params["s2a"] * np.sin(2.0 * theta)
                )
                r_res = r_points - (params["R0"] + params["a"] * np.cos(tb))
                z_res = z_points - (params["Z0"] - params["a"] * params["ka"] * np.sin(theta))
                return np.concatenate([r_res, z_res])

            bounds = (
                np.array([lower_bounds_by_name[name] for name in free_names], dtype=float),
                np.array([upper_bounds_by_name[name] for name in free_names], dtype=float),
            )

            for start in starts:
                start["a"] = max(start["a"], lower_bounds_by_name["a"])
                start["ka"] = max(start["ka"], lower_bounds_by_name["ka"])
                x0 = pack_params(start)
                fit = least_squares(
                    residual,
                    x0=x0,
                    bounds=bounds,
                    method="trf",
                )
                rms = float(np.sqrt(np.mean(fit.fun**2)))
                if best_fit is None or rms < best_fit["rms"]:
                    best_fit = {"fit": fit, "rms": rms, "params": unpack_params(fit.x)}

        fitted = best_fit["params"]
        R0 = fitted["R0"]
        Z0 = fitted["Z0"]
        a = fitted["a"]
        ka = fitted["ka"]
        c0a = fitted["c0a"]
        c1a = fitted["c1a"]
        s1a = fitted["s1a"]
        s2a = fitted["s2a"]
        c0a = float((c0a + np.pi) % (2.0 * np.pi) - np.pi)

        return {
            "R0": float(R0),
            "Z0": float(Z0),
            "a": float(a),
            "ka": float(ka),
            "c0a": c0a,
            "c1a": float(c1a),
            "s1a": float(s1a),
            "s2a": float(s2a),
            "rms": best_fit["rms"],
        }

    def _read_header(self, file):
        line = file.readline()
        header_regex = re.compile(r"^\s*(.*)\s+(\d+)\s+(\d+)\s*$", re.I)
        match = header_regex.match(line)
        if match:
            _, self.nr, self.nz = (
                match.groups(1),
                int(match.group(2)),
                int(match.group(3)),
            )
        else:
            raise ValueError(f"Error reading header from line: {line}")
        print("Geqdsk")
        print(f" - nr, nz: {self.nr}, {self.nz}")

    def _read_geometry(self, file):
        line = self._sanitize_line(file.readline())
        geometry_regex = re.compile(r"^\s*(\S+)\s*(\S+)\s*(\S+)\s*(\S+)\s*(\S+)")
        match = geometry_regex.match(line)
        if match:
            Rboxlen, Zboxlen, self.R0, self.Rmin, self.Z0 = map(float, match.groups())
            self.Rmax = self.Rmin + Rboxlen
            self.Zmin = self.Z0 - 0.5 * Zboxlen
            self.Zmax = self.Z0 + 0.5 * Zboxlen
            self.dr = Rboxlen / (self.nr - 1)
            self.dz = Zboxlen / (self.nz - 1)
        else:
            raise ValueError(f"Error reading geometry from line: {line}")

    def _read_axfig_and_current(self, file):
        for i in range(2):  # Read 2 lines: axis and current
            line = self._sanitize_line(file.readline())
            values = self._extract_get_mesh_from_line(line)
            if i == 0:
                self.Raxis, self.Zaxis, self.psi_axis, self.psi_bound, self.Bt0 = values
            else:
                self.I_total = values[0]

    def _read_profile(self, file):
        file.readline()  # Skip useless line
        lines = file.read()
        lines = re.sub("\n", " ", lines)
        lines = re.sub(r"(\d)-", r"\1 -", lines)
        fields = re.split(r"\s+", lines)

        all_datas = np.array([self._safe_float_conversion(n) for n in fields if n])

        # Read profile and psi values
        self.f = all_datas[: self.nr]
        self.p = all_datas[self.nr : 2 * self.nr]
        self.fdf = all_datas[2 * self.nr : 3 * self.nr]
        self.dp = all_datas[3 * self.nr : 4 * self.nr]
        self.psi = np.array(
            [all_datas[4 * self.nr + i * self.nz : 4 * self.nr + (i + 1) * self.nz] for i in range(self.nr)]
        )

        # Read q, phi, rho, xi profile
        index = 4 * self.nr + self.nr * self.nz
        self.q = all_datas[index : index + self.nr]
        self._calculate_phi_rho_xi()

        # Read boundary and limiter geometry
        nbound = int(all_datas[index + self.nr])
        nlimiter = int(all_datas[index + self.nr + 1])
        index += self.nr + 2

        Rbound, Zbound = self._read_geometry_pairs(all_datas, index, nbound)
        index += 2 * nbound

        Rlimiter, Zlimiter = self._read_geometry_pairs(all_datas, index, nlimiter)

        self.boundary = np.column_stack((Rbound, Zbound))
        self.limiter = np.column_stack((Rlimiter, Zlimiter))

    def _sanitize_line(self, line):
        return re.sub(r"([^Ee])-", r"\1 -", line)

    def _extract_get_mesh_from_line(self, line):
        return list(map(float, re.split(r"\s+", line.strip())))

    def _safe_float_conversion(self, value):
        try:
            return float(value)
        except ValueError:
            return np.nan  # Use np.nan to denote invalid values

    def _calculate_phi_rho_xi(self):
        psi_array = np.linspace(self.psi_axis, self.psi_bound, self.nr)
        self.phi = np.zeros(self.nr)
        for i in range(self.nr - 1):
            self.phi[i + 1] = simpson(y=self.q[: i + 2], x=psi_array[: i + 2]) * 2 * np.pi

        self.rho = np.zeros(self.nr)
        for i in range(1, self.nr):
            self.rho[i] = (self.phi[i] / (np.pi * self.Bt0)) ** 0.5

        self.xi = np.zeros(self.nr)
        for i in range(1, self.nr):
            self.xi[i] = self.rho[i] / self.rho[-1]

    def _read_geometry_pairs(self, data, start_index, n_pairs):
        R = data[start_index : start_index + 2 * n_pairs : 2]
        Z = data[start_index + 1 : start_index + 2 * n_pairs : 2]
        return R, Z

    def _plot_profile(self, ax, data, ylabel, xlabel, color, sci=False):
        ax.plot(self.xi, data, color=color)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if sci:
            ax.ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")

    def _interp_and_update(self, attr, mode="linear"):
        interp_func = interp1d(self.xi, getattr(self, attr), kind=mode)
        setattr(self, attr, interp_func(self.xi))
