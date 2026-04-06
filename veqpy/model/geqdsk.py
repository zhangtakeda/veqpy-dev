"""
Passive GEQDSK payload container plus canonical text read/write helpers.

Boundary fitting is owned by `veqpy.model.boundary`; `Geqdsk` only stores
GEQDSK data and serializes it.
"""

from __future__ import annotations

import os
import re
from dataclasses import InitVar, dataclass, field

import numpy as np

from veqpy.model.serial import Serial, read_serializer, write_serializer


@dataclass(slots=True)
class Geqdsk(Serial):
    path: InitVar[str | os.PathLike[str] | None] = None

    header: str = ""

    nr: int = 0
    nz: int = 0

    R0: float = 0.0
    Z0: float = 0.0
    dr: float = 0.0
    dz: float = 0.0
    Rmin: float = 0.0
    Rmax: float = 0.0
    Zmin: float = 0.0
    Zmax: float = 0.0

    boundary: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    limiter: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))

    Bt0: float = 0.0
    Raxis: float = 0.0
    Zaxis: float = 0.0
    I_total: float = 0.0
    psi_axis: float = 0.0
    psi_bound: float = 0.0

    f: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    p: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    fdf: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    dp: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    q: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    phi: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    rho: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    xi: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    psi: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))

    @classmethod
    def serial_attributes(cls) -> dict[str, type]:
        return {
            "header": str,
            "nr": int,
            "nz": int,
            "R0": float,
            "Z0": float,
            "dr": float,
            "dz": float,
            "Rmin": float,
            "Rmax": float,
            "Zmin": float,
            "Zmax": float,
            "boundary": np.ndarray,
            "limiter": np.ndarray,
            "Bt0": float,
            "Raxis": float,
            "Zaxis": float,
            "I_total": float,
            "psi_axis": float,
            "psi_bound": float,
            "f": np.ndarray,
            "p": np.ndarray,
            "fdf": np.ndarray,
            "dp": np.ndarray,
            "q": np.ndarray,
            "phi": np.ndarray,
            "rho": np.ndarray,
            "xi": np.ndarray,
            "psi": np.ndarray,
        }

    def __post_init__(self, path: str | os.PathLike[str] | None) -> None:
        self.header = str(self.header)
        self.nr = int(self.nr)
        self.nz = int(self.nz)

        for name in (
            "R0",
            "Z0",
            "dr",
            "dz",
            "Rmin",
            "Rmax",
            "Zmin",
            "Zmax",
            "Bt0",
            "Raxis",
            "Zaxis",
            "I_total",
            "psi_axis",
            "psi_bound",
        ):
            setattr(self, name, float(getattr(self, name)))

        self.boundary = _coerce_point_array(self.boundary, name="boundary")
        self.limiter = _coerce_point_array(self.limiter, name="limiter")
        self.f = _coerce_vector(self.f, name="f")
        self.p = _coerce_vector(self.p, name="p")
        self.fdf = _coerce_vector(self.fdf, name="fdf")
        self.dp = _coerce_vector(self.dp, name="dp")
        self.q = _coerce_vector(self.q, name="q")
        self.phi = _coerce_vector(self.phi, name="phi")
        self.rho = _coerce_vector(self.rho, name="rho")
        self.xi = _coerce_vector(self.xi, name="xi")
        self.psi = _coerce_matrix(self.psi, name="psi")

        if path is not None:
            self.read(os.fspath(path))
            return

        self._refresh_spacing()
        if (
            self.q.size
            and self.q.shape[0] == self.nr
            and self.phi.size == 0
            and self.rho.size == 0
            and self.xi.size == 0
        ):
            self.refresh_flux_coordinates()

    def check(self) -> None:
        if self.nr < 0 or self.nz < 0:
            raise ValueError("nr and nz must be non-negative")

        for name in ("boundary", "limiter"):
            arr = getattr(self, name)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"{name} must have shape (N, 2), got {arr.shape}")

        for name in ("f", "p", "fdf", "dp", "q", "phi", "rho", "xi"):
            arr = getattr(self, name)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1D, got {arr.shape}")

        if self.psi.ndim != 2:
            raise ValueError(f"psi must be 2D, got {self.psi.shape}")

        if self.nr == 0 or self.nz == 0:
            return

        expected_radial = self.nr
        expected_psi = (self.nr, self.nz)
        for name in ("f", "p", "fdf", "dp", "q"):
            arr = getattr(self, name)
            if arr.size != expected_radial:
                raise ValueError(f"{name} must have length {expected_radial}, got {arr.size}")

        for name in ("phi", "rho", "xi"):
            arr = getattr(self, name)
            if arr.size not in (0, expected_radial):
                raise ValueError(f"{name} must have length {expected_radial} or be empty, got {arr.size}")

        if self.psi.shape != expected_psi:
            raise ValueError(f"psi must have shape {expected_psi}, got {self.psi.shape}")

    def refresh_flux_coordinates(self) -> None:
        if self.q.size == 0:
            self.phi = np.empty(0, dtype=np.float64)
            self.rho = np.empty(0, dtype=np.float64)
            self.xi = np.empty(0, dtype=np.float64)
            return
        if self.nr <= 0 or self.q.size != self.nr:
            raise ValueError("q must have length nr before computing phi/rho/xi")

        psi_array = np.linspace(self.psi_axis, self.psi_bound, self.nr, dtype=np.float64)
        phi = np.zeros(self.nr, dtype=np.float64)
        if self.nr > 1:
            delta = np.diff(psi_array)
            phi[1:] = 2.0 * np.pi * np.cumsum(0.5 * (self.q[:-1] + self.q[1:]) * delta)

        rho = np.zeros(self.nr, dtype=np.float64)
        if abs(self.Bt0) > 0.0:
            scaled = np.clip(phi / (np.pi * self.Bt0), a_min=0.0, a_max=None)
            rho[1:] = np.sqrt(scaled[1:])

        xi = np.zeros(self.nr, dtype=np.float64)
        if rho.size > 0 and abs(rho[-1]) > 0.0:
            xi[1:] = rho[1:] / rho[-1]

        self.phi = phi
        self.rho = rho
        self.xi = xi

    @read_serializer("txt", "geqdsk", "gfile")
    def read_geqdsk(self, file: str) -> Geqdsk:
        with open(file, "r", encoding="utf-8") as handle:
            self._read_header(handle)
            self._read_geometry(handle)
            self._read_axfig_and_current(handle)
            self._read_profiles_and_boundary(handle)
        self._refresh_spacing()
        self.refresh_flux_coordinates()
        return self

    @write_serializer("txt", "geqdsk", "gfile")
    def write_geqdsk(self, file: str) -> None:
        self.check()
        with open(file, "w", encoding="utf-8") as handle:
            handle.write(_format_header_line(self.header, self.nr, self.nz))
            handle.write(
                _format_float_line([self.Rmax - self.Rmin, self.Zmax - self.Zmin, self.R0, self.Rmin, self.Z0])
            )
            handle.write(_format_float_line([self.Raxis, self.Zaxis, self.psi_axis, self.psi_bound, self.Bt0]))
            handle.write(_format_float_line([self.I_total, 0.0, 0.0, 0.0, 0.0]))
            handle.write("\n")
            handle.write(_format_float_block(self.f))
            handle.write(_format_float_block(self.p))
            handle.write(_format_float_block(self.fdf))
            handle.write(_format_float_block(self.dp))
            handle.write(_format_float_block(self.psi.reshape(-1)))
            handle.write(_format_float_block(self.q))
            handle.write(f"{int(self.boundary.shape[0])} {int(self.limiter.shape[0])}\n")
            handle.write(_format_float_block(self.boundary.reshape(-1)))
            handle.write(_format_float_block(self.limiter.reshape(-1)))

    def _read_header(self, file) -> None:
        line = file.readline()
        match = re.match(r"^\s*(.*)\s+(\d+)\s+(\d+)\s*$", line, re.I)
        if match is None:
            raise ValueError(f"Error reading header from line: {line}")
        self.header = match.group(1).rstrip()
        self.nr = int(match.group(2))
        self.nz = int(match.group(3))

    def _read_geometry(self, file) -> None:
        line = _sanitize_line(file.readline())
        match = re.match(r"^\s*(\S+)\s*(\S+)\s*(\S+)\s*(\S+)\s*(\S+)", line)
        if match is None:
            raise ValueError(f"Error reading geometry from line: {line}")

        r_box_len, z_box_len, self.R0, self.Rmin, self.Z0 = map(float, match.groups())
        self.Rmax = self.Rmin + r_box_len
        self.Zmin = self.Z0 - 0.5 * z_box_len
        self.Zmax = self.Z0 + 0.5 * z_box_len

    def _read_axfig_and_current(self, file) -> None:
        for index in range(2):
            line = _sanitize_line(file.readline())
            values = list(map(float, re.split(r"\s+", line.strip())))
            if index == 0:
                self.Raxis, self.Zaxis, self.psi_axis, self.psi_bound, self.Bt0 = values
            else:
                self.I_total = values[0]

    def _read_profiles_and_boundary(self, file) -> None:
        file.readline()
        payload = _sanitize_line(file.read().replace("\n", " "))
        fields = re.split(r"\s+", payload.strip())
        data = np.array([_safe_float_conversion(value) for value in fields if value], dtype=np.float64)

        nr = self.nr
        nz = self.nz
        index = 0

        def take(count: int, *, label: str) -> np.ndarray:
            nonlocal index
            end = index + count
            if data.size < end:
                raise ValueError(f"GEQDSK payload ended before {label}.")
            out = data[index:end]
            index = end
            return out

        self.f = take(nr, label="f profile").copy()
        self.p = take(nr, label="p profile").copy()
        self.fdf = take(nr, label="fdf profile").copy()
        self.dp = take(nr, label="dp profile").copy()
        self.psi = take(nr * nz, label="psi grid").reshape(nr, nz).copy()
        self.q = take(nr, label="q profile").copy()

        counts = take(2, label="boundary metadata")
        nbound = int(counts[0])
        nlimiter = int(counts[1])

        boundary_flat = take(2 * nbound, label="LCFS boundary points")
        limiter_flat = take(2 * nlimiter, label="limiter points")
        self.boundary = boundary_flat.reshape(nbound, 2).copy()
        self.limiter = limiter_flat.reshape(nlimiter, 2).copy()

    def _refresh_spacing(self) -> None:
        if self.nr > 1 and np.isfinite(self.Rmax) and np.isfinite(self.Rmin):
            self.dr = (self.Rmax - self.Rmin) / (self.nr - 1)
        else:
            self.dr = 0.0

        if self.nz > 1 and np.isfinite(self.Zmax) and np.isfinite(self.Zmin):
            self.dz = (self.Zmax - self.Zmin) / (self.nz - 1)
        else:
            self.dz = 0.0


def _coerce_vector(value, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}")
    return arr.copy()


def _coerce_matrix(value, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {arr.shape}")
    return arr.copy()


def _coerce_point_array(value, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {arr.shape}")
    return arr.copy()


def _sanitize_line(line: str) -> str:
    return re.sub(r"([^Ee])-", r"\1 -", line)


def _safe_float_conversion(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return np.nan


def _format_header_line(header: str, nr: int, nz: int) -> str:
    title = (header or "veqpy GEQDSK").strip()
    return f"{title} {int(nr)} {int(nz)}\n"


def _format_float_line(values: list[float] | tuple[float, ...]) -> str:
    return "".join(f"{float(value):16.9E}" for value in values) + "\n"


def _format_float_block(values: np.ndarray, *, columns: int = 5) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return ""
    lines = []
    for start in range(0, arr.size, columns):
        chunk = arr[start : start + columns]
        lines.append("".join(f"{float(value):16.9E}" for value in chunk))
    return "\n".join(lines) + "\n"
