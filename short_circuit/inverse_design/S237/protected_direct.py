"""Reflection symmetry preconditioner for the three slowest frequencies of the 50×50 operator."""
from functools import lru_cache

import numpy as np

from model import C, DIAG, FB, FREQ, conv


_SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_SELECTED = frozenset(np.flatnonzero(np.isin(FREQ, (3.15, 3.55, 4.0))))


def use_direct(k):
    return k in _SELECTED


@lru_cache(maxsize=3)
def _inverses(k):
    y, x = (axis.ravel() for axis in np.indices((25, 25)))
    kernel = np.fft.ifft2(C[k])
    blocks = []
    for sy, sx in _SIGNS:
        block = np.zeros((625, 625), complex)
        for wy, yy in ((1, y), (sy, 49-y)):
            for wx, xx in ((1, x), (sx, 49-x)):
                block += wy * wx * kernel[y[:, None] - yy, x[:, None] - xx]
        # The constant part commutes with reflections. Iterative refinement
        # below applies the true cell-dependent diagonal in double precision.
        block.flat[::626] += DIAG[k] + .5 * FB[k]
        blocks.append(np.linalg.inv(block).astype(np.complex64))
    return blocks


def _apply(rhs, blocks):
    solution = np.zeros(rhs.shape, complex)
    tl = rhs[:25, :25]
    tr = rhs[:25, 25:][:, ::-1]
    bl = rhs[25:, :25][::-1]
    br = rhs[25:, 25:][::-1, ::-1]
    for (sy, sx), block in zip(_SIGNS, blocks):
        component = ((tl + sx*tr + sy*bl + sy*sx*br) * .25).astype(np.complex64)
        v = (block @ component.ravel()).reshape(25, 25)
        solution[:25, :25] += v
        solution[:25, 25:] += sx*v[:, ::-1]
        solution[25:, :25] += sy*v[::-1]
        solution[25:, 25:] += sy*sx*v[::-1, ::-1]
    return solution


def try_direct(mask, k, rhs, adjoint=False):
    # A is complex symmetric. Conjugating both sides also solves A* x = rhs.
    scale = np.max(np.abs(rhs))
    if scale == 0:
        return np.zeros_like(rhs, dtype=complex)
    b = (rhs.conj() if adjoint else rhs) / scale
    try:
        blocks = _inverses(k)
    except (np.linalg.LinAlgError, MemoryError):
        return None
    diagonal = DIAG[k] + (1-mask)*FB[k]
    x = _apply(b, blocks)
    x += _apply(b - diagonal*x - conv(C[k], x), blocks)
    residual = np.linalg.norm(b - diagonal*x - conv(C[k], x))
    if not np.isfinite(residual) or residual > 1e-11 * np.linalg.norm(b):
        return None
    return (x.conj() if adjoint else x) * scale
