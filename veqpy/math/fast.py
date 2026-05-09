import numba as nb
import numpy as np


@nb.njit(
    nb.void(
        nb.float64[:, ::1],
        nb.float64[:, ::1],
        nb.float64[:, ::1],
    ),
    cache=True,
    nogil=True,
    fastmath=True,
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
