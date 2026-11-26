"""Lower the high-frequency plane error by moving neighbouring drive positions together."""
from itertools import combinations

import numpy as np

from cluster_repair import Ranker


def shifted_pairs(mask):
    cells = np.flatnonzero(mask)
    pairs = np.array(list(combinations(range(len(cells)), 2)))
    y, x = np.divmod(cells[pairs], 50)
    groups = []
    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)):
        yy, xx = y+dy, x+dx
        legal = ((yy>=0)&(yy<50)&(xx>=0)&(xx<50)).all(axis=1)
        slots = pairs[legal]
        proposed = np.broadcast_to(cells, (len(slots),len(cells))).copy()
        proposed[np.arange(len(slots))[:,None],slots] = 50*yy[legal]+xx[legal]
        proposed.sort(axis=1)
        groups.append(proposed[(np.diff(proposed,axis=1)!=0).all(axis=1)])
    return np.unique(np.concatenate(groups),axis=0)


def proposals(objective, mask, pressure, metrics, expired):
    ranker = Ranker(objective,pressure)
    ranker.parents = np.flatnonzero(mask)[None]
    ids = shifted_pairs(mask)
    if not len(ids) or expired():
        return
    # The native P/H planes and interpolated regions cheaply filter the family
    # of joint moves before evaluating the 23-frequency plane error.
    ratio = ranker.levels(ids,True)
    eligible = np.flatnonzero((ratio[:,1:6].max(axis=1)<=1.00001)
                               &(ratio[:,6]<metrics["after"]/ranker.scale[-1] + 1e-6))
    if not len(eligible) or expired():
        return
    whole = ranker.whole(ids[eligible])
    eligible = eligible[whole<=1.00001]
    order = eligible[np.argsort(ratio[eligible,6],kind="stable")]
    for index in order:
        if expired():
            return
        candidate = np.zeros_like(mask)
        candidate.flat[ids[index]] = 1
        yield candidate
