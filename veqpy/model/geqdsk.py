"""
Compact CHEASE GEQDSK reader for LCFS boundary extraction.

This type is intentionally narrow:
- read CHEASE-style GEQDSK files
- retain only boundary-related geometry metadata
"""

from __future__ import annotations

import os
import re

import numpy as np


class Geqdsk:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.nr = 0
        self.nz = 0

        self.R0 = 0.0
        self.Z0 = 0.0
        self.dr = 0.0
        self.dz = 0.0
        self.Rmin = 0.0
        self.Rmax = 0.0
        self.Zmin = 0.0
        self.Zmax = 0.0

        self.boundary = np.empty((0, 2), dtype=float)

        self.Bt0 = 0.0
        self.Raxis = 0.0
        self.Zaxis = 0.0
        self.I_total = 0.0
        self.psi_axis = 0.0
        self.psi_bound = 0.0

        if path is not None:
            self.read(path)

    def read(self, path: str | os.PathLike[str]) -> None:
        with open(os.fspath(path), "r", encoding="utf-8") as file:
            self._read_header(file)
            self._read_geometry(file)
            self._read_axfig_and_current(file)
            self._read_boundary(file)

    def _read_header(self, file) -> None:
        line = file.readline()
        match = re.match(r"^\s*(.*)\s+(\d+)\s+(\d+)\s*$", line, re.I)
        if match is None:
            raise ValueError(f"Error reading header from line: {line}")
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
        self.dr = r_box_len / (self.nr - 1)
        self.dz = z_box_len / (self.nz - 1)

    def _read_axfig_and_current(self, file) -> None:
        for index in range(2):
            line = _sanitize_line(file.readline())
            values = list(map(float, re.split(r"\s+", line.strip())))
            if index == 0:
                self.Raxis, self.Zaxis, self.psi_axis, self.psi_bound, self.Bt0 = values
            else:
                self.I_total = values[0]

    def _read_boundary(self, file) -> None:
        file.readline()
        payload = _sanitize_line(file.read().replace("\n", " "))
        fields = re.split(r"\s+", payload.strip())
        data = np.array([_safe_float_conversion(value) for value in fields if value], dtype=float)

        index = 4 * self.nr + self.nr * self.nz + self.nr
        if data.size < index + 2:
            raise ValueError("CHEASE payload ended before boundary metadata.")

        nbound = int(data[index])
        index += 2

        if nbound <= 0:
            self.boundary = np.empty((0, 2), dtype=float)
            return

        end = index + 2 * nbound
        if data.size < end:
            raise ValueError("CHEASE payload ended before LCFS boundary points.")

        self.boundary = data[index:end].reshape(nbound, 2)


def _sanitize_line(line: str) -> str:
    return re.sub(r"([^Ee])-", r"\1 -", line)


def _safe_float_conversion(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return np.nan
