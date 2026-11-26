"""Propose simultaneous binary exchanges using seven pressure error sensitivities."""
import numpy as np

from common import H, L, P, roi
from model import DIAG, FB, G, N, conv
from protected_linear import solve


def _phase(z):
    mag = abs(z)
    return np.divide(z, mag, out=np.zeros_like(z), where=mag > 0)


def responses(mask, obj, p, u):
    """Linearized toggle effects, in the native units of each acceptance test."""
    delta = 1 - 2 * mask
    a = abs(p)
    err = a - obj.a
    local_p = roi(p[[P, H]])
    local = abs(local_p) - obj.b[[P, H]]
    local_rms = np.sqrt(np.mean(local**2, axis=(1, 2)))
    whole = 100 * np.linalg.norm(err) / obj.norm
    high = 100 * np.linalg.norm(err[H]) / obj.high_norm
    # Mean bias and the center errors keep their sign until after prediction.
    values = np.array([whole, err[P].mean(), *local[:, 2, 2], *local_rms, high])
    columns = np.zeros((7, mask.size))

    def add(j, k, receiver):
        adj = solve(mask, k, conv(G[k].conj(), receiver), True)
        diag = DIAG[k] + (1 - mask) * FB[k]
        # Exact toggle for the uncoupled diagonal; off-diagonal interaction
        # between flips is recovered by the subsequent full binary solve.
        toggle = delta * (N[k] + FB[k] * u[k]) / (1 - delta * FB[k] / diag)
        columns[j] += np.real(adj.conj() * toggle).ravel()

    # For speed approximate the broadband derivative by the frequencies
    # carrying most error energy, retaining their fraction of the total.
    energy = np.sum(err**2, axis=(1, 2))
    important = np.argsort(energy)[-3:]
    total_sampled = energy[important].sum()
    if whole > 0 and total_sampled > 0:
        for k in important:
            add(0, k, err[k] * _phase(p[k]) * whole / total_sampled)
    add(1, P, _phase(p[P]) / mask.size)
    center = L[2, :, None] * L[2, None, :]
    for j, k in enumerate((P, H)):
        add(2 + j, k, center * _phase(local_p[j, 2:3, 2:3]))
        if local_rms[j] > 0:
            add(4 + j, k, L.T @ (local[j] * _phase(local_p[j])) @ L / (25 * local_rms[j]))
    if high > 0:
        add(6, H, err[H] * _phase(p[H]) * (100 / obj.high_norm)**2 / high)
    return values, columns.T


def proposals(mask, obj, p, u, family, rng):
    """Rank feasible-count edits on signed sensitivities before any full solve."""
    values, slopes = responses(mask, obj, p, u)
    bounds = np.r_[obj.bounds, obj.before]
    scale = np.maximum(bounds, np.maximum(1., abs(values)) * 1e-9)
    values /= scale
    slopes /= scale
    bounds /= scale
    on, off = np.flatnonzero(mask), np.flatnonzero(1 - mask)
    if not len(on) or not len(off):
        return
    if family in ("patch", "stripe", "cluster", "edge"):
        # Exchange along interfaces, where small edits preserve the observed
        # structure. Retain a sample of unrestricted edits for diffuse masks.
        extended = np.pad(mask, 1, mode="edge")
        neighbors = sum(extended[y:y+50, x:x+50] for y in range(3) for x in range(3)).ravel()
        border_on = on[neighbors[on] < 9]
        border_off = off[neighbors[off] > 0]
    else:
        border_on, border_off = on, off
    down = np.r_[rng.choice(on, 1536), rng.choice(border_on, 4096)]
    up = np.r_[rng.choice(off, 1536), rng.choice(border_off, 4096)]
    steps = slopes[down] + slopes[up]
    # The extra clearance absorbs linearization error. Exact acceptance uses
    # the user's unmodified limits on fields recomputed by the native solver.
    goal = bounds.copy()
    goal[-1] *= .97
    def costs(prediction):
        margin = abs(prediction) - goal
        excess = np.maximum(margin, 0)
        return margin.max(axis=1) + .35 * excess.sum(axis=1)

    current = values.copy()
    pool = np.ones(len(up), bool)
    choices = []
    removed, added = [], []
    for n in range(1, min(36, len(on), len(off)) + 1):
        costs_all = np.where(pool, costs(current + steps), np.inf)
        i = int(np.argmin(costs_all))
        if not np.isfinite(costs_all[i]):
            break
        removed.append(int(down[i]))
        added.append(int(up[i]))
        current += steps[i]
        if n in (1, 2, 4, 8, 16, 24, 36) or costs_all[i] < -.04:
            choices.append((costs_all[i], tuple(removed), tuple(added)))
        if costs_all[i] < -.04:
            break
        pool &= (down != down[i]) & (up != up[i])
    # Include alternate single moves, especially at a sharp class boundary.
    single = costs(values + steps)
    for i in np.argpartition(single, 5)[:5]:
        choices.append((single[i], (int(down[i]),), (int(up[i]),)))
    seen = set()
    for cost, rem, add in sorted(choices):
        if (rem, add) in seen:
            continue
        seen.add((rem, add))
        candidate = mask.copy()
        candidate.flat[list(rem)] = 0
        candidate.flat[list(add)] = 1
        yield candidate
