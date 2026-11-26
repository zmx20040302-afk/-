"""Screen fixed-count exchanges against the complex response at each requested point."""
import numpy as np

from model import DIAG, FB, G, N, conv, solve


def point_proposals(mask, objective, field, flow, family, rng):
    positions = np.argwhere(objective.used)
    # Each observed frequency-position requires a separate receiver solve.
    changes = []
    delta = 1 - 2 * mask
    for j, y, x in positions:
        k = objective.ks[j]
        receiver = np.zeros_like(mask)
        receiver[y, x] = 1
        adjoint = solve(mask, k, conv(G[k].conj(), receiver), True)
        diagonal = DIAG[k] + (1 - mask) * FB[k]
        correction = 1 - delta * FB[k] / diagonal
        changes.append((adjoint.conj() * delta * (N[k] + FB[k] * flow[j]) / correction).ravel())
    changes = np.stack(changes, axis=1)
    current = field[objective.used]
    target = objective.target[objective.used]
    weights = objective.weight[positions[:, 0], 0, 0]
    on = np.flatnonzero(mask)
    off = np.flatnonzero(1 - mask)
    # Candidate pools preserve local structure in coherent families; certification
    # and all final pressure values are always evaluated on the full binary mask.
    pools = [(on, off)]
    if family in ("cluster", "stripe", "patch"):
        extended = np.pad(mask, 1, mode="edge")
        neighbors = sum(extended[i:i+50, j:j+50] for i in range(3) for j in range(3)).ravel()
        border_on = on[neighbors[on] < 9]
        border_off = off[neighbors[off] > 0]
        if len(border_on) and len(border_off):
            pools.insert(0, (border_on, border_off))
    if family == "edge":
        y, x = np.indices(mask.shape)
        depth = np.minimum.reduce([x, y, 49-x, 49-y]).ravel()
        near = np.percentile(depth[on], 95)
        far = np.percentile(depth[off], 5)
        pools.insert(0, (on[depth[on] >= far-1], off[depth[off] <= near+1]))

    def scores(predicted):
        errors = abs(abs(predicted) - target)
        # All modes share the same absolute per-item tolerance.
        return np.max(errors, axis=1), np.sum(weights * errors**2, axis=1)

    choices = []
    for active, idle in pools:
        if not len(active) or not len(idle):
            continue
        if len(active)*len(idle) <= 14000:
            down, up = np.repeat(active, len(idle)), np.tile(idle, len(active))
        else:
            down = rng.choice(active, size=14000)
            up = rng.choice(idle, size=14000)
        step = changes[down] + changes[up]
        approx = current.copy()
        removed, added = [], []
        usable = np.ones(len(up), bool)
        for _ in range(min(24, len(active), len(idle))):
            maximum, cost = scores(approx + step)
            # Emphasize distance to the acceptance region in Pa. Use the
            # requested objective to break ties, including within tolerance.
            rank = np.where(usable, maximum, np.inf)
            best = int(np.argmin(rank))
            if not np.isfinite(rank[best]) or maximum[best] >= np.max(abs(abs(approx)-target)):
                break
            removed.append(down[best])
            added.append(up[best])
            approx += step[best]
            choices.append((float(maximum[best]), float(cost[best]), tuple(removed), tuple(added)))
            if maximum[best] < 3:
                break
            usable &= (down != down[best]) & (up != up[best])
        # Several alternatives to a single swap help when the best numerical
        # move does not meet the family constraint.
        maximum, cost = scores(current + step)
        for i in np.argpartition(maximum, min(len(maximum)-1, 8))[:8]:
            choices.append((float(maximum[i]), float(cost[i]), (down[i],), (up[i],)))
    seen = set()
    for maximum, cost, removed, added in sorted(choices, key=lambda c: (c[0] > 10, c[0], c[1])):
        key = (tuple(sorted(removed)), tuple(sorted(added)))
        if key in seen:
            continue
        seen.add(key)
        candidate = mask.copy()
        candidate.flat[list(removed)] = 0
        candidate.flat[list(added)] = 1
        yield candidate
