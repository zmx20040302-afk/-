"""Circulant-preconditioned iteration on the native PMUT equation during search."""
from functools import lru_cache

import numpy as np

from model import C, DIAG, FB, G, N, conv, solve as reference_solve


@lru_cache(maxsize=23)
def preconditioner(k):
    # Frobenius projection of finite 2-D Toeplitz convolution onto BCCB.
    native = np.fft.ifft2(C[k]).reshape(2, 50, 2, 50)
    index = np.arange(100)
    weight = (1 - np.minimum(index, 100-index) / 50).reshape(2, 50)
    projected = np.einsum("iajb,ia,jb->ab", native, weight, weight)
    symbol = np.fft.fft2(projected) + DIAG[k]
    return 1 / symbol


def solve(mask, k, rhs, adjoint=False):
    from protected_direct import use_direct, try_direct
    if use_direct(k):
        direct = try_direct(mask, k, rhs, adjoint)
        if direct is not None:
            return direct
    diagonal = DIAG[k] + (1 - mask) * FB[k]
    kernel = C[k]
    inv = preconditioner(k)
    if adjoint:
        diagonal, kernel, inv = diagonal.conj(), kernel.conj(), inv.conj()
    def op(v):
        return diagonal*v + conv(kernel, v)

    def inverse(v):
        return np.fft.ifft2(inv * np.fft.fft2(v))

    norm = np.linalg.norm(rhs)
    x = np.zeros_like(rhs, dtype=complex)
    if norm == 0:
        return x
    r = rhs.copy()
    shadow = r.copy()
    rho = alpha = omega = 1.
    p = np.zeros_like(x)
    v = np.zeros_like(x)
    for it in range(600):
        new = np.vdot(shadow, r)
        if new == 0 or omega == 0:
            break
        p = r + (new / rho) * (alpha / omega) * (p-omega*v)
        phat = inverse(p)
        v = op(phat)
        den = np.vdot(shadow, v)
        if den == 0:
            break
        alpha = new / den
        s = r-alpha*v
        x += alpha*phat
        if np.linalg.norm(s) <= norm*1e-11:
            break
        shat = inverse(s)
        t = op(shat)
        den = np.vdot(t, t)
        if den == 0:
            break
        omega = np.vdot(t, s) / den
        x += omega*shat
        r = s - omega*t
        rho = new
        if np.linalg.norm(r) <= norm*1e-11:
            break
    rel = np.linalg.norm(op(x)-rhs)/norm
    if not np.isfinite(rel) or rel>1e-9:
        return reference_solve(mask, k, rhs, adjoint)
    return x


def forward(mask, frequencies):
    flow = np.stack([solve(mask, k, N[k] * mask) for k in frequencies])
    field = np.stack([conv(G[k], v) for k, v in zip(frequencies, flow)])
    return field, flow


@lru_cache(maxsize=12)
def cached_field(packed, frequencies):
    mask = np.unpackbits(np.frombuffer(packed, np.uint8), count=2500).reshape(50, 50).astype(float)
    return forward(mask, frequencies)


def seed_field(mask, frequencies):
    return cached_field(np.packbits(mask.astype(np.uint8)).tobytes(), tuple(frequencies))
