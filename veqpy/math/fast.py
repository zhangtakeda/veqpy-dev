"""
Module: math.fast

Role:
- Provide small Numba kernels for reusable dense math operations.

Public API:
- copy_vector
- dot1d
- dot2d_axis0_into
- dot2d_axis1_into
- fill_pointwise_product
- fill_product_ratio
- fill_scaled_product
- fill_scaled_ratio
- fill_scaled_vector
- matmul_into
- matvec_into
- maximum_floor
- maximum_floor_out
- project_rows_to_packed
- quadrature
- quadrature_product
- quadrature_product_ratio
- sum2d_axis1_into
- theta_reduction
- weighted_sum2d_axis0_into
- weighted_sum2d_axis1_into
"""

import numba as nb
import numpy as np

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------


@nb.njit(
    nb.void(nb.float64[:, ::1], nb.float64[:, ::1], nb.float64[:, ::1]),
    cache=True,
    nogil=True,
    fastmath=True,
    inline="always",
)
def matmul_into(
    out: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> None:
    """
    Compute out = a @ b.

    Requirements:
        a, b, out are float64 C-contiguous arrays.
        a.shape == (m, k)
        b.shape == (k, n)
        out.shape == (m, n)
        out must not alias a or b.
    """
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]

    for i in range(m):
        for j in range(n):
            out[i, j] = 0.0

        for p in range(k):
            aip = a[i, p]
            for j in range(n):
                out[i, j] += aip * b[p, j]


@nb.njit(
    nb.void(
        nb.float64[::1],
        nb.float64[:, ::1],
        nb.float64[::1],
    ),
    cache=True,
    nogil=True,
    inline="always",
)
def matvec_into(
    out: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
) -> None:
    rows = a.shape[0]
    cols = a.shape[1]

    for i in range(rows):
        total = 0.0

        for j in range(cols):
            total += a[i, j] * x[j]

        out[i] = total


@nb.njit(
    nb.float64(
        nb.float64[::1],
        nb.float64[::1],
    ),
    cache=True,
    nogil=True,
    fastmath=True,
    inline="always",
)
def dot1d(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Compute:

        sum_i a[i] * b[i]

    Requirements:
        a, b are float64 C-contiguous 1D arrays.
        a.shape == b.shape.
    """
    total = 0.0

    for i in range(a.shape[0]):
        total += a[i] * b[i]

    return total


@nb.njit(
    nb.void(
        nb.float64[::1],
        nb.float64[:, ::1],
        nb.float64[:, ::1],
    ),
    cache=True,
    nogil=True,
    fastmath=True,
    inline="always",
)
def dot2d_axis0_into(
    out: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> None:
    """
    Compute:

        out[j] = sum_i a[i, j] * b[i, j]

    This compresses axis 0.
    Result shape: (ncols,)
    """
    nrows = a.shape[0]
    ncols = a.shape[1]

    for j in range(ncols):
        out[j] = 0.0

    # Row-major friendly:
    # a[i, j], b[i, j], out[j] are accessed with contiguous j.
    for i in range(nrows):
        for j in range(ncols):
            out[j] += a[i, j] * b[i, j]


@nb.njit(
    nb.void(
        nb.float64[::1],
        nb.float64[:, ::1],
        nb.float64[:, ::1],
    ),
    cache=True,
    nogil=True,
    fastmath=True,
    inline="always",
)
def dot2d_axis1_into(
    out: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> None:
    """
    Compute:

        out[i] = sum_j a[i, j] * b[i, j]

    This compresses axis 1.
    Result shape: (nrows,)
    """
    nrows = a.shape[0]
    ncols = a.shape[1]

    for i in range(nrows):
        total = 0.0

        for j in range(ncols):
            total += a[i, j] * b[i, j]

        out[i] = total


@nb.njit(
    nb.void(
        nb.float64[::1],
        nb.float64[:, ::1],
    ),
    cache=True,
    fastmath=True,
    nogil=True,
    inline="always",
)
def sum2d_axis1_into(out: np.ndarray, a: np.ndarray) -> None:
    nrows = a.shape[0]
    ncols = a.shape[1]

    for i in range(nrows):
        total = 0.0

        for j in range(ncols):
            total += a[i, j]

        out[i] = total


@nb.njit(
    nb.void(
        nb.float64[::1],
        nb.float64[:, ::1],
        nb.float64[::1],
    ),
    cache=True,
    fastmath=True,
    nogil=True,
    inline="always",
)
def weighted_sum2d_axis1_into(out: np.ndarray, a: np.ndarray, weights: np.ndarray) -> None:
    nrows = a.shape[0]
    ncols = a.shape[1]

    for i in range(nrows):
        total = 0.0

        for j in range(ncols):
            total += a[i, j] * weights[j]

        out[i] = total


@nb.njit(
    nb.void(
        nb.float64[::1],
        nb.float64[:, ::1],
        nb.float64[::1],
    ),
    cache=True,
    fastmath=True,
    nogil=True,
    inline="always",
)
def weighted_sum2d_axis0_into(out: np.ndarray, a: np.ndarray, weights: np.ndarray) -> None:
    nrows = a.shape[0]
    ncols = a.shape[1]

    for j in range(ncols):
        total = 0.0

        for i in range(nrows):
            total += weights[i] * a[i, j]

        out[j] = total


@nb.njit(cache=True, nogil=True)
def theta_reduction(out: np.ndarray, arr: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    if axis == 0:
        for j in range(arr.shape[1]):
            total = 0.0

            for i in range(arr.shape[0]):
                total += weights[i] * arr[i, j]

            out[j] = total
        return out

    if axis == 1:
        scale = 2.0 * np.pi / arr.shape[1]

        for i in range(arr.shape[0]):
            total = 0.0

            for j in range(arr.shape[1]):
                total += arr[i, j]

            out[i] = total * scale
        return out

    raise ValueError(f"Unsupported quadrature axis {axis}")


@nb.njit(cache=True, nogil=True)
def quadrature(arr: np.ndarray, weights: np.ndarray) -> float:
    if arr.ndim == 1:
        total = 0.0

        for i in range(arr.shape[0]):
            total += arr[i] * weights[i]

        return total

    radial_sum = np.empty(arr.shape[1], dtype=arr.dtype)

    for j in range(arr.shape[1]):
        radial_total = 0.0

        for i in range(arr.shape[0]):
            radial_total += weights[i] * arr[i, j]

        radial_sum[j] = radial_total

    total = 0.0

    for j in range(radial_sum.shape[0]):
        total += radial_sum[j]

    return (2.0 * np.pi / arr.shape[1]) * total


@nb.njit(cache=True, nogil=True, inline="always")
def quadrature_product(lhs: np.ndarray, rhs: np.ndarray, weights: np.ndarray) -> float:
    total = 0.0

    for i in range(lhs.shape[0]):
        total += weights[i] * lhs[i] * rhs[i]

    return total


@nb.njit(cache=True, nogil=True, inline="always")
def quadrature_product_ratio(
    lhs: np.ndarray, rhs: np.ndarray, den: np.ndarray, weights: np.ndarray
) -> float:
    total = 0.0

    for i in range(lhs.shape[0]):
        total += weights[i] * lhs[i] * rhs[i] / den[i]

    return total


@nb.njit(cache=True, fastmath=True, nogil=True, inline="always")
def project_rows_to_packed(
    out_packed: np.ndarray,
    coeff_indices: np.ndarray,
    matrix: np.ndarray,
    vector: np.ndarray,
) -> None:
    rows = coeff_indices.shape[0]
    cols = vector.shape[0]

    for i in range(rows):
        total = 0.0

        for j in range(cols):
            total += matrix[i, j] * vector[j]

        out_packed[coeff_indices[i]] = total


@nb.njit(cache=True, fastmath=True, nogil=True, inline="always")
def copy_vector(out: np.ndarray, src: np.ndarray) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = src[i]
    return out


@nb.njit(cache=True, fastmath=True, nogil=True, inline="always")
def fill_scaled_vector(out: np.ndarray, src: np.ndarray, scale: float) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = scale * src[i]
    return out


@nb.njit(cache=True, fastmath=True, nogil=True, inline="always")
def fill_pointwise_product(out: np.ndarray, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = lhs[i] * rhs[i]
    return out


@nb.njit(cache=True, fastmath=True, nogil=True, inline="always")
def fill_scaled_product(
    out: np.ndarray, lhs: np.ndarray, rhs: np.ndarray, scale: float
) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = scale * lhs[i] * rhs[i]
    return out


@nb.njit(cache=True, fastmath=True, nogil=True, inline="always")
def fill_scaled_ratio(
    out: np.ndarray, num: np.ndarray, den: np.ndarray, scale: float
) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = scale * num[i] / den[i]
    return out


@nb.njit(cache=True, fastmath=True, nogil=True, inline="always")
def fill_product_ratio(
    out: np.ndarray,
    lhs: np.ndarray,
    rhs: np.ndarray,
    den: np.ndarray,
    scale: float,
) -> np.ndarray:
    for i in range(out.shape[0]):
        out[i] = scale * lhs[i] * rhs[i] / den[i]
    return out


@nb.njit(cache=True, nogil=True, inline="always")
def maximum_floor(arr: np.ndarray, floor: float) -> np.ndarray:
    out = np.empty_like(arr)

    for i in range(arr.shape[0]):
        value = arr[i]
        out[i] = value if value > floor else floor

    return out


@nb.njit(cache=True, nogil=True, inline="always")
def maximum_floor_out(out: np.ndarray, arr: np.ndarray, floor: float) -> np.ndarray:
    for i in range(arr.shape[0]):
        value = arr[i]
        out[i] = value if value > floor else floor

    return out


# -----------------------------------------------------------------------------
# New neutral names (Phase 1d)
# -----------------------------------------------------------------------------

copy_into = copy_vector
scale_into = fill_scaled_vector
product_into = fill_pointwise_product
scaled_product_into = fill_scaled_product
scaled_ratio_into = fill_scaled_ratio
scaled_product_ratio_into = fill_product_ratio
maximum_floor_into = maximum_floor_out
indexed_matvec_into = project_rows_to_packed
weighted_product_sum = quadrature_product
weighted_product_ratio_sum = quadrature_product_ratio
