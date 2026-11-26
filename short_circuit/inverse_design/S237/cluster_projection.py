"""Project sparse continuation parents across the frozen cluster boundary."""
import numpy as np

from cluster_geometry import signed_sparse


def _lowest(scores, count):
    count = min(len(scores), count)
    return np.argpartition(scores, count-1)[:count]


def _neighborhood(parent, parent_ratios, rank, margin, checker, expired, penalty):
    y, x = np.divmod(parent, 50)
    dy, dx = (np.indices((9, 9)) - 4).reshape(2, -1)
    yy = y[:, None] + dy
    xx = x[:, None] + dx
    source, offset = np.where((yy >= 0) & (xx >= 0) & (yy < 50) & (xx < 50) &
                              ((dy != 0) | (dx != 0)))
    dest = 50 * yy[source, offset] + xx[source, offset]
    free = ~np.isin(dest, parent)
    source, dest = source[free], dest[free]
    if not len(dest):
        return None
    singles = np.broadcast_to(parent, (len(dest), len(parent))).copy()
    singles[np.arange(len(singles)), source] = dest

    rank.parents = parent[None]
    margin.adapt(parent[None])
    base_margin = margin.margins(parent[None])[0]
    single_margin = margin.margins(singles)
    base_ratios = rank.levels(parent[None], False)[0]
    single_ratios = rank.levels(singles, False) + parent_ratios - base_ratios
    cost = penalty(single_ratios)
    keep = [_lowest(single_margin + weight*cost, 18) for weight in (.35, 1., 3., 10.)]
    candidates = [singles[np.unique(np.concatenate(keep))]]

    for cell in range(len(parent)-1):
        if expired():
            return None
        left = np.flatnonzero(source == cell)
        right = np.flatnonzero(source > cell)
        first = np.repeat(left, len(right))
        second = np.tile(right, len(left))
        distinct = dest[first] != dest[second]
        first, second = first[distinct], second[distinct]
        if not len(first):
            continue
        # The sum of single-step changes ranks pairs. Each retained pair
        # is scored again using its combined complex field and geometry.
        ratio = single_ratios[first] + single_ratios[second] - parent_ratios
        geom = single_margin[first] + single_margin[second] - base_margin
        cost = penalty(ratio)
        keep = [_lowest(geom + weight*cost, 16) for weight in (.3, 1, 3, 10, 30)]
        keep.extend(_lowest(np.maximum(geom, 0) + weight*cost + .01*geom, 24)
                    for weight in (.5, 4))
        indices = np.unique(np.concatenate(keep))
        double = singles[first[indices]].copy()
        selected = second[indices]
        double[np.arange(len(double)), source[selected]] = dest[selected]
        candidates.append(double)

    candidates = np.unique(np.sort(np.vstack(candidates), axis=1), axis=0)
    if expired():
        return None
    geom = margin.margins(candidates)
    ratios = rank.levels(candidates, True)
    ratios[:, 0] += parent_ratios[0] - base_ratios[0]
    cost = penalty(ratios)
    keep = [_lowest(geom + weight*cost, 8) for weight in (.2, .7, 2, 8, 32)]
    keep.extend(_lowest(np.maximum(geom, 0) + weight*cost + .01*geom, 14)
                for weight in (.25, 1, 10))
    keep.append(_lowest(np.maximum(geom, 0) + cost, 20))
    indices = np.unique(np.concatenate(keep))
    candidates, ratios = candidates[indices], ratios[indices]
    if expired():
        return None
    signed, legal, _ = signed_sparse(candidates, checker)
    ratios[:, 0] = rank.whole(candidates)
    return candidates, ratios, signed, legal


def proposals(parents, archive, rank, margin, checker, expired, *, steps=14):
    """Yield acoustic and geometry finalists, using only this request's parents."""
    from cluster_continuation import acoustic_cost
    if not len(parents) or expired():
        return
    local = {tuple(parent): archive[tuple(parent)] for parent in parents}
    visited = set()
    original_parents = rank.parents
    try:
        for step in range(steps):
            if expired():
                return
            available = [key for key in local if key not in visited]
            if not available:
                return
            weight = (2, 8, .45, 1, 4)[step % 5]
            ratios = np.stack([local[key][0] for key in available])
            signed = np.array([local[key][1] for key in available])
            score = np.maximum(signed, 0) + weight*acoustic_cost(ratios) + .04*signed
            key = available[np.argmin(score)]
            visited.add(key)
            result = _neighborhood(np.asarray(key), local[key][0], rank, margin,
                                   checker, expired, acoustic_cost)
            if result is None:
                return
            candidates, ratios, signed, legal = result
            for row, values, geom in zip(candidates, ratios, signed):
                local[tuple(row)] = (values, geom)
            for index in np.flatnonzero(legal & (ratios.max(axis=1) < 1.000001)):
                mask = np.zeros((50, 50))
                mask.flat[candidates[index]] = 1
                yield mask
    finally:
        rank.parents = original_parents
