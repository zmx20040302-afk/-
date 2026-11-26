"""Screen fixed-count exchanges around target-independent sparse cluster shapes."""
import numpy as np

from cluster_search import Screen
from cluster_transfer import field as estimate_pressure
from common import P


def _cost(ratios):
    excess = np.maximum(ratios-1, 0.)
    return excess.max(axis=-1) + .22*excess.sum(axis=-1) + .025*np.minimum(ratios,2.).mean(axis=-1)


def _proposals(screen, obj, current, metrics, native_mean, atoms, certified, seen, expired):
    on = np.flatnonzero(current)
    off = np.flatnonzero(1-current)
    field = atoms[on].sum(axis=0)
    amp = abs(field)
    estimated = screen.ratio(amp[None], np.array([native_mean]))[0]
    observed = np.r_[metrics["limits_after"], metrics["after"]] / np.maximum(np.r_[obj.bounds,obj.before],1e-10)
    # The full-plane & high-frequency error use a sparse quadrature only for
    # proposing moves. Anchor it at the exact score of the current shape.
    correction = np.zeros(7)
    correction[[0,6]] = (observed-estimated)[[0,6]]
    mean_offset = native_mean - amp[0,:screen.dim].mean()
    off_fields = atoms[off]
    scores, pairs = [], []
    for remove in on:
        if expired():
            return
        trial = abs(off_fields + (field - atoms[remove]))
        means = trial[:,0,:screen.dim].mean(axis=1) + mean_offset
        ratios = np.maximum(screen.ratio(trial, means) + correction, 0.)
        scores.append(_cost(ratios))
        pairs.append(ratios[:,:2].max(axis=1))
    costs = np.concatenate(scores)
    pair = np.concatenate(pairs)
    oy, ox = np.divmod(on,50)
    fy, fx = np.divmod(off,50)
    distance = np.hypot(oy[:,None]-fy, ox[:,None]-fx).ravel()
    # A short exchange tends to preserve certification at a shape boundary.
    # Interleave four rankings to keep both long-range and local repairs.
    orders = [np.argsort(s,kind="stable")[:96] for s in
              (costs, costs+.014*distance, costs+.05*distance, .65*pair+.35*costs)]
    indices = set()
    found = 0
    for index in np.asarray(orders).T.ravel():
        if expired() or len(indices)>=160 or found>=4:
            break
        if index in indices:
            continue
        indices.add(index)
        i,j = divmod(index,len(off))
        trial = current.copy()
        trial.flat[on[i]] = 0
        trial.flat[off[j]] = 1
        packed = np.packbits(trial.astype(bool)).tobytes()
        if packed not in seen:
            seen.add(packed)
            if certified(trial,"cluster"):
                found+=1
                yield trial


def sparse_refinements(objective, initial_p, seeds, certified, expired):
    """Search several local basins; all accepted outputs still use the native solver."""
    if not seeds or not 6<=seeds[0].sum()<=24:
        return
    screen = Screen(objective,initial_p)
    re,im = np.split(screen.rows,2,axis=1)
    atoms = (re+1j*im).reshape(2500,3,-1).astype(np.complex64)
    scale = np.maximum(np.r_[objective.bounds,objective.before],1e-10)
    seen=set()
    queue=[]
    def assess(m):
        # Scores only order the queue. The caller verifies proposed designs
        # against the native equations before they can replace its incumbent.
        field=estimate_pressure(m)
        _,metrics=objective.evaluate(field)
        ratio=np.r_[metrics["limits_after"],metrics["after"]]/scale
        return float(_cost(ratio)),m,metrics,float(abs(field[P]).mean())
    for seed in seeds[:4]:
        if expired():
            return
        key=np.packbits(seed.astype(bool)).tobytes()
        if key not in seen and certified(seed,"cluster"):
            seen.add(key)
            queue.append(assess(seed))
    for _ in range(8):
        if expired() or not queue:
            break
        queue.sort(key=lambda node:node[0])
        cost,mask,metrics,native_mean=queue.pop(0)
        yield mask
        if metrics["protected_ok"]:
            return
        for trial in _proposals(screen,objective,mask,metrics,native_mean,atoms,certified,seen,expired):
            if expired():
                break
            node=assess(trial)
            if node[2]["protected_ok"]:
                yield trial
                return
            if node[0] <= cost+.35:
                queue.append(node)
