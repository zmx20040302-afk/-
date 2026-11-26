"""Choose sparse cluster starting points from a target-independent certified pose atlas."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from cluster_search import Screen, pose


@lru_cache(maxsize=1)
def atlas():
    with np.load(Path(__file__).with_name("sparse_cluster_positions.npz")) as data:
        return {key:data[key] for key in data.files}


def select_seeds(objective, initial, count):
    data=atlas()
    indices=np.flatnonzero(data["counts"]==count)
    if not indices.size:
        return [],False
    screen=Screen(objective,initial)
    values=data["sampled"][indices]
    plane=values[:,:,:64].reshape(-1,3,8,8)
    region=values[:,:,64:].reshape(-1,3,5,5)
    ratio=[]
    for orientation in range(8):
        a=pose(plane,orientation).reshape(-1,3,64)
        b=pose(region,orientation).reshape(-1,3,25)
        ratio.append(screen.ratio(np.concatenate((a,b),axis=2),data["means"][indices]))
    ratio=np.concatenate(ratio)
    excess=np.maximum(ratio-1.,0.)
    joint=ratio[:,:2].max(axis=1)
    all_errors=ratio.max(axis=1)+.18*excess.sum(axis=1)
    orders=[np.argsort(all_errors,kind="stable"),
            np.lexsort((all_errors,joint)),
            np.argsort(all_errors+.4*np.maximum(joint-1,0),kind="stable")]
    seeds,used=[],set()
    for order in orders:
        for index in order:
            orientation,pos=divmod(index,indices.size)
            idx=indices[pos]
            shape=int(data["shape"][idx])
            if shape in used:
                continue
            used.add(shape)
            mask=np.unpackbits(data["packed"][idx],count=2500).reshape(50,50)
            seeds.append(pose(mask,orientation).astype(float))
            break
    near=bool(all_errors.min() < 2.5 or (joint.min() < 1.5 and all_errors.min() < 4.))
    return seeds,near
