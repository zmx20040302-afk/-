"""Infer fixed-count candidate masks from the supplied full complex observation plane."""
import numpy as np
from model import C, DIAG, FB, FREQ, G, N, conv


def drive_score(target):
    """Regularized acoustic inversion, followed by the open-electrode drive equation."""
    scores = []
    for mhz in (8.0, 12.5):
        k = int(np.flatnonzero(FREQ == mhz)[0])
        kernel = G[k] / np.abs(G[k]).max()
        field = target[k] / np.abs(G[k]).max()
        rhs = conv(kernel.conj(), field)
        flow = np.zeros_like(rhs)
        residual = rhs.copy()
        direction = residual.copy()
        power = np.vdot(residual, residual).real
        initial_power = power
        for _ in range(45):
            if power <= initial_power * 1e-16:
                break
            applied = conv(kernel.conj(), conv(kernel, direction)) + 1e-5 * direction
            step = power / np.vdot(direction, applied).real
            flow += step * direction
            residual -= step * applied
            new_power = np.vdot(residual, residual).real
            direction = residual + new_power / power * direction
            power = new_power
        # (D + C + FB)u = mask * (N + FB*u). A real least-squares score
        # estimates the local drive on the blurred, regularized velocity map.
        loaded = (DIAG[k] + FB[k]) * flow + conv(C[k], flow)
        drive = N[k] + FB[k] * flow
        projected = (loaded * drive.conj()).real / np.maximum(abs(drive)**2, np.finfo(float).tiny)
        scale = np.std(projected)
        if scale > 0:
            scores.append((projected - projected.mean()) / scale)
    return sum(scores) if scores else np.zeros((50, 50))


def ranked_masks(score, seed):
    """Try direct thresholding and increasingly conservative same-count edits."""
    active = np.flatnonzero(seed)
    inactive = np.flatnonzero(1 - seed)
    low = active[np.argsort(score.flat[active], kind="stable")]
    high = inactive[np.argsort(-score.flat[inactive], kind="stable")]
    count = int(np.sum(score.flat[high[:min(low.size, high.size)]] >
                       score.flat[low[:min(low.size, high.size)]]))
    for n in dict.fromkeys([count, count // 2, count // 5, count // 12, 4, 1]):
        if 0 < n <= count:
            candidate = seed.copy()
            candidate.flat[low[:n]] = 0
            candidate.flat[high[:n]] = 1
            yield candidate
