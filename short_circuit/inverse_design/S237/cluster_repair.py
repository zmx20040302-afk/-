"""Explore sparse layouts near a jointly feasible field/mean incumbent."""
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import parent_process

import numpy as np

from cluster_search import pose
from cluster_transfer import field, table
from common import H, P
from geometry_screen import receiver_rows


@lru_cache(maxsize=1)
def _band_workers():
    # Five-family requests already run one process per family.
    return None if parent_process() is not None else ThreadPoolExecutor(max_workers=4)


@lru_cache(maxsize=1)
def dictionaries():
    # The same frozen transfer serves every objective. The coarse quadrature
    # ranks exploratory mutations. All returned proposals receive a native solve.
    basis, canonical, orientations, bands = table()
    grid = np.array([2, 8, 15, 21, 28, 34, 41, 47])
    quad = np.empty((2500, 23, 64), np.complex64)
    planes = np.empty((2500, 2, 2500), np.complex64)
    for k, band in enumerate(bands):
        q = basis[:, :band.shape[-1]]
        native = q @ band @ q.T if k in (P, H) else None
        coarse = q[grid] @ band @ q[grid].T
        for orientation in range(8):
            rows = np.flatnonzero(orientations == orientation)
            quad[rows,k] = pose(coarse, orientation)[canonical[rows]].reshape(-1,64)
            if native is not None:
                planes[rows, int(k == H)] = pose(native, orientation)[canonical[rows]].reshape(-1,2500)
    real, imag = np.split(receiver_rows()[0], 2, axis=1)
    local = (real+1j*imag).reshape(2500,3,89)[:,[0,2],-25:]
    return grid, quad, planes, np.ascontiguousarray(local, dtype=np.complex64)


def combine(atoms, indices):
    result = atoms[indices[:,0]].copy()
    for column in indices.T[1:]:
        result += atoms[column]
    return result


def cost(ratios):
    excess = np.maximum(ratios-1, 0)
    return excess.max(axis=-1) + .21*excess.sum(axis=-1) + .004*np.minimum(ratios, 2).mean(axis=-1)


def protected_rank(metrics):
    ratios = np.r_[metrics["limits_after"],metrics["after"]] / np.maximum(
        np.r_[metrics["limits_before"],metrics["before"]], 1e-12)
    faults = np.count_nonzero(~np.array(metrics["limits_ok"])) + int(not metrics["peak_ok"])
    faults += int(metrics["after"] >= metrics["before"] and not metrics["exact_field"])
    return faults, float(cost(ratios))


def rough_geometry(ids):
    n = ids.shape[1]
    y,x = np.divmod(ids,50)
    radius = np.sqrt(y.var(axis=1)+x.var(axis=1))
    dx = abs(x[:,:,None]-x[:,None,:])
    dy = abs(y[:,:,None]-y[:,None,:])
    near = ((dx<=4)&(dy<=4)).sum(axis=2).max(axis=1)
    windows = ((dx<=8)&(dy<=8)).sum(axis=2).max(axis=1)
    adjacent = ((dx+dy)==1).sum(axis=(1,2))*.5
    return (np.maximum(radius/5.8-1,0)+.65*np.maximum(.80-windows/n,0)
            +.3*np.maximum(.53-near/n,0)+.04*np.maximum(3-adjacent,0))


def geometry_margin(mask, checker, calibration):
    from classifier import FAMILIES, ALPHA
    result = checker.predict(mask)
    if result["status"] == "single" and result["family"] == "cluster":
        return 0., True
    cluster = FAMILIES.index("cluster")
    distances = np.array([result["distances"][name] for name in FAMILIES])
    others = np.broadcast_to(distances,(len(distances),len(distances))).copy()
    np.fill_diagonal(others,np.inf)
    relative = distances/np.maximum(others.min(axis=1),1e-12)
    # Both original conformal p-values must exceed ALPHA/2. This continuous
    # distance to the original decision supplies an exploratory ranking only.
    rank = max(1, int(np.floor((calibration.shape[-1]+1)*ALPHA/2)))
    boundary = np.sort(calibration,axis=-1)[:,:,-rank]
    margin = np.maximum(distances/boundary[:,0]-1, relative/boundary[:,1]-1)
    return max(0.,margin[cluster])+max(0.,-np.delete(margin,cluster).min())+.025, False


def mutate(rng, initial, inferred, archive, generation, size):
    from cluster_mutations import planned_mutate
    planned = planned_mutate(rng, initial, inferred, archive, generation, size)
    if planned is not None:
        return planned
    n = len(initial)
    proposals = []
    for j in range(size):
        if generation == 0 or j < 320:
            if j % 3:
                cells = inferred.copy()
                amount = rng.integers(min(5,n),min(11,n+1))
                center = inferred[rng.integers(n)] if j % 7 else rng.integers(2500)
                y,x = divmod(center,50)
                distance = (cells//50-y)**2 + (cells%50-x)**2
                loc = np.argsort(distance*rng.uniform(.5,2,n))[-amount:]
                cells[loc] = (50*np.clip(y+np.rint(rng.normal(0,2.,amount)).astype(int),0,49)
                              +np.clip(x+np.rint(rng.normal(0,2.,amount)).astype(int),0,49))
            else:
                cells = initial.copy()
                amount = rng.integers(2,min(8,n+1))
                shift = rng.integers(-8,9,(2,amount))
                loc = rng.choice(n,amount,replace=False)
                y,x = np.divmod(cells[loc],50)
                cells[loc] = 50*np.clip(y+shift[0],0,49)+np.clip(x+shift[1],0,49)
        else:
            cells = archive[rng.integers(len(archive))][1].copy()
            amount = min(n,rng.choice([1,1,2,2,3,4,5]))
            loc = rng.choice(n,amount,replace=False)
            anchors = cells[loc]
            if j%9 == 0:
                anchors = inferred[rng.integers(n,size=amount)]
            if j%9 == 1:
                anchors = cells[rng.integers(n,size=amount)]
            y,x = np.divmod(anchors,50)
            radius = rng.choice([1,2,4,8])
            cells[loc] = (50*np.clip(y+rng.integers(-radius,radius+1,amount),0,49)
                          +np.clip(x+rng.integers(-radius,radius+1,amount),0,49))
        cells.sort()
        if np.all(np.diff(cells)):
            proposals.append(cells)
    return np.unique(proposals,axis=0)


class Ranker:
    def __init__(self, obj, incumbent):
        self.grid, self.quad, self.planes, self.local = dictionaries()
        self.obj = obj
        self.goal = obj.a[:,self.grid[:,None],self.grid].reshape(23,-1)
        self.scale = np.maximum(np.r_[obj.bounds,obj.before],1e-10)
        self.local_goal = obj.b[[P,H]].reshape(1,2,25)
        self.plane_goal = obj.a[[P,H]].reshape(1,2,2500)
        self.mean_goal = obj.a[P].mean()
        self.mean_delta = abs(incumbent[P]).mean()-abs(incumbent[P][self.grid[:,None],self.grid]).mean()
        self.parents = None

    def subset(self, ids):
        if self.parents is None:
            return []
        each = np.zeros((len(self.parents),2500),dtype=bool)
        np.put_along_axis(each, self.parents, True, axis=1)
        yes = each[:,ids]
        most = yes.sum(axis=2,dtype=np.int16).argmax(axis=0)
        matches = yes[most,np.arange(len(ids))]
        distance = ids.shape[1]-matches.sum(axis=1)
        blocks=[]
        for size in np.unique(distance):
            i=np.flatnonzero(distance==size)
            if size==0:
                blocks.append((i,most[i],None,None))
            elif size*2 < ids.shape[1]:
                parents=self.parents[most[i]]
                kept=(parents[:,:,None]==ids[i,None,:]).any(axis=2)
                down=parents[~kept].reshape(len(i),size)
                up=ids[i][~matches[i]].reshape(len(i),size)
                blocks.append((i,most[i],down,up))
            else:
                blocks.append((i,None,None,None))
        return blocks

    def sum_fields(self,atoms,ids,blocks,parent_fields):
        if not blocks:
            return combine(atoms,ids)
        result=np.empty((len(ids),*atoms.shape[1:]),atoms.dtype)
        for indices,anchors,down,up in blocks:
            if anchors is None:
                result[indices]=combine(atoms,ids[indices])
                continue
            values=parent_fields[anchors]
            if down is not None:
                for a,b in zip(down.T,up.T):
                    values += atoms[b]-atoms[a]
            result[indices]=values
        return result

    def whole(self,ids):
        # Evaluate many layouts per frequency. Summing source coefficients before
        # reconstruction avoids solving the same 23 separate systems per proposal.
        from cluster_transfer import source_bands
        def error_sum(item):
            k,(q,weights) = item
            values=np.empty(len(ids))
            for lo in range(0,len(ids),32):
                ci=ids[lo:lo+32]
                coefficient=weights[ci].sum(axis=1)
                error=abs(q @ coefficient @ q.T)-self.obj.a[k]
                values[lo:lo+len(ci)] = np.einsum("bxy,bxy->b",error,error)
            return values
        bands=enumerate(source_bands())
        executor=_band_workers() if len(ids)>=64 else None
        terms=executor.map(error_sum,bands) if executor is not None else map(error_sum,bands)
        norm2=sum(terms)
        return 100*np.sqrt(norm2)/self.obj.norm/self.scale[0]

    def levels(self, indices, full_planes, *, return_ancestors=False):
        result = []
        ancestors = np.zeros(len(indices),dtype=int) if return_ancestors else None
        parents=[combine(a,self.parents) if self.parents is not None else None for a in
                 (self.quad,self.local)]
        parent_planes=combine(self.planes,self.parents) if self.parents is not None and full_planes else None
        for start in range(0,len(indices),128):
            ids = indices[start:start+128]
            blocks=self.subset(ids)
            if return_ancestors:
                for index,nearest,_,_ in blocks:
                    if nearest is not None:
                        ancestors[start+index] = nearest
            q = abs(self.sum_fields(self.quad,ids,blocks,parents[0]))
            error = q-self.goal
            energy = np.einsum("bfq,bfq->bf",error,error)*(2500/64)
            if full_planes:
                plane = abs(self.sum_fields(self.planes,ids,blocks,parent_planes))-self.plane_goal
                energy[:,[P,H]] = np.einsum("bfq,bfq->bf",plane,plane)
                mean = abs(plane[:,0].mean(axis=1))
            else:
                mean = abs(q[:,P].mean(axis=1)+self.mean_delta-self.mean_goal)
            loc = abs(self.sum_fields(self.local,ids,blocks,parents[1]))-self.local_goal
            result.append(np.column_stack([100*np.sqrt(energy.sum(axis=1))/self.obj.norm,mean,
                abs(loc[:,:,12]),np.sqrt((loc*loc).mean(axis=2)),
                100*np.sqrt(energy[:,H])/self.obj.high_norm])/self.scale)
        ratios=np.concatenate(result)
        return (ratios,ancestors) if return_ancestors else ratios


def proposals(obj, mask, metrics, inferred, checker, expired):
    from cluster_geometry import signed_sparse
    start = np.flatnonzero(mask)
    n = len(start)
    if inferred is None or not 6 <= n <= 24 or not all(metrics["limits_ok"][:2]):
        return
    inferred = np.flatnonzero(inferred)
    if len(inferred) != n:
        return
    ranker = Ranker(obj,field(mask))
    ratios = np.r_[metrics["limits_after"],metrics["after"]]/ranker.scale
    correction = ratios-ranker.levels(start[None],False)[0]
    best = protected_rank(metrics)
    last_improvement = 0
    archive = [(0.,start,0.)]
    seen = set()
    rng = np.random.default_rng(3161)
    for gen in range(35):
        # Once just one criterion remains, spend the rest of this request on
        # the dedicated pair moves after two fruitless exploratory generations.
        if expired() or (best[0] == 1 and gen >= last_improvement+3):
            break
        ids = mutate(rng,start,inferred,archive,gen,4500)
        ranker.parents = np.array([start,inferred,*[node[1] for node in archive]])
        ratio = ranker.levels(ids,False) + correction
        geometry = rough_geometry(ids)
        value = cost(ratio)+.9*np.maximum(ratio[:,:2]-1,0).max(axis=1)
        chosen = np.unique(np.concatenate([np.argsort(value+w*geometry)[:145] for w in (.18,.65,2.,5.)]))
        ids,geometry = ids[chosen],geometry[chosen]
        ratio = ranker.levels(ids,True)
        value = cost(ratio)+.8*np.maximum(ratio[:,:2]-1,0).max(axis=1)
        chosen = np.unique(np.concatenate([np.argsort(value+w*geometry)[:65] for w in (.15,.65,2.)]))
        nodes = []
        chosen=np.array([i for i in chosen if tuple(ids[i]) not in seen],dtype=int)
        flags,margins=(),()
        if len(chosen):
            ratio[chosen,0]=ranker.whole(ids[chosen])
            _,flags,margins=signed_sparse(ids[chosen],checker)
        for i,certified,margin in zip(chosen,flags,margins):
            key = tuple(ids[i])
            if key in seen:
                continue
            seen.add(key)
            candidate = np.zeros((50,50))
            candidate.flat[ids[i]] = 1
            r = ratio[i]
            sc = cost(r)+np.maximum(r[:2]-1,0).max()
            if certified and np.all(r[:2]<1.00001):
                met = obj.evaluate(field(candidate))[1]
                rank = protected_rank(met)
                if rank < best or met["protected_ok"]:
                    yield candidate
                    best = rank
                    last_improvement = gen
            nodes.append((sc,ids[i],margin))
        if gen:
            nodes += archive
        archive, used = [], set()
        for j in range(12):
            weight = (.25,.6,1.,1.8,4.,9.)[j%6]
            for node in sorted(nodes,key=lambda p:p[0]+weight*p[2]):
                key = tuple(node[1])
                if key in used or (j<6 and any(len(set(key)&set(prev))>n-2 for prev in used)):
                    continue
                archive.append(node)
                used.add(key)
                break
        if not archive:
            break
