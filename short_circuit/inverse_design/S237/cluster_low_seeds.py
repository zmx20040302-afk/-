"""Rank a target-independent low-pressure K=12 cluster atlas for a protected request."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from cluster_search import pose
from cluster_transfer import field
from common import F, H, P


@lru_cache(maxsize=1)
def atlas(path=None):
    with np.load(path or Path(__file__).with_suffix(".npz")) as data:
        return {key:data[key] for key in data.files}


def preserves(candidate, incumbent):
    old=np.asarray(incumbent["limits_after"][:2])
    new=np.asarray(candidate["limits_after"][:2])
    return (candidate["protected_ok"] or
            (np.all(new<=old) and np.any(new<old) and
             np.all(~np.asarray(incumbent["limits_ok"])|candidate["limits_ok"]) and
             (not incumbent["peak_ok"] or candidate["peak_ok"]) and
             (incumbent["after"]>=incumbent["before"] or candidate["after"]<candidate["before"])))


def proposals(objective, incumbent, metrics, expired, *, data=None):
    """Yield up to two promising masks. The caller verifies each with the native solver."""
    if int(incumbent.sum())!=12 or metrics["limits_ok"][1] or expired():
        return
    data=atlas() if data is None else data
    old=np.maximum(metrics["limits_after"][:2],1e-12)
    means=data["spectrum"][:,P]
    peaks=F[data["spectrum"].argmax(axis=1)]
    mean_error=abs(means-objective.a[P].mean())
    possible=mean_error<=old[1]+.01
    if metrics["peak_ok"]:
        possible&=(peaks==objective.peak)
    idx=np.flatnonzero(possible)
    if not len(idx):return
    means=mean_error[idx]
    plane=data["blocks"][idx]
    region=data["center"][idx]
    # Cell averages and both full-field energies estimate the 23-frequency
    # squared error. Retain a small margin before the full-grid reranking.
    target=objective.a.reshape(23,10,5,10,5).mean(axis=(2,4))
    energy=data["energy"][idx].sum(1)+objective.norm**2
    candidates=[]
    for orientation in range(8):
        dot=25*np.einsum("bfij,fij->b",pose(plane,orientation),target)
        whole=100*np.sqrt(np.maximum(energy-2*dot,0))/objective.norm
        dif=pose(region,orientation)-objective.b[[P,H]]
        regions=np.sqrt(np.mean(dif*dif,axis=(2,3)))
        local=np.column_stack((abs(dif[:,:,2,2]),regions))
        keep=np.all((local<=objective.bounds[2:]+.1)|~np.asarray(metrics["limits_ok"][2:]),axis=1)
        for index in np.flatnonzero(keep):
            scale=np.array([whole[index],means[index]])/old
            candidates.append((max(scale)+.4*sum(scale),idx[index],orientation))
    if not candidates:return
    reranked=[]
    cache={}
    for _,idx,orientation in sorted(candidates)[:18]:
        if expired():break
        if idx not in cache:
            m=np.zeros((50,50))
            m.flat[data["ids"][idx]]=1
            cache[idx]=m,field(m)
        m,p=cache[idx]
        candidate=objective.evaluate(pose(p,orientation))[1]
        if preserves(candidate,metrics):
            pair=np.asarray(candidate["limits_after"][:2])/old
            reranked.append((not candidate["protected_ok"],max(pair)+.4*sum(pair),idx,orientation))
    for _,_,idx,orientation in sorted(reranked)[:2]:
        yield pose(cache[idx][0],orientation).copy()
