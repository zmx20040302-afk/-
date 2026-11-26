"""Bounded rigid subset search for a sparse six-constraint incumbent."""
import itertools
import numpy as np
from cluster_repair import Ranker
from cluster_geometry import calibration_for, margins, signed_sparse, subset_spatial, SPATIAL, describe_batch
from common import P,H

def excess_score(ratio):
    excess=np.maximum(ratio-1,0)
    return excess.max(axis=1)+.31*excess.sum(axis=1)+.02*ratio[:,6]

def coarse_values(obj,pressure,local,goal,scale):
    amp=abs(pressure)
    e=amp-goal
    energy=np.einsum("bfq,bfq->bf",e,e)*(2500/amp.shape[2])
    r=abs(local)-obj.b[[P,H]].reshape(1,2,25)
    return np.column_stack((100*np.sqrt(energy.sum(1))/obj.norm,
            abs(amp[:,P].mean(1)-obj.a[P].mean()),abs(r[:,:,12]),np.sqrt((r*r).mean(2)),
            100*np.sqrt(energy[:,H])/obj.high_norm))/scale

def neighborhood(obj,center,ratios,rank,checker,expired,cache):
    n=len(center);y,x=np.divmod(center,50)
    subsets=(np.arange(1,2**n)[:,None]>>np.arange(n))%2!=0
    grid=np.array([0,2,5,7])
    cols=(8*grid[:,None]+grid).ravel()
    quad=rank.quad[:,:,cols]
    local=rank.local;goal=rank.goal[:,cols]
    baseq=quad[center].sum(0);baser=local[center].sum(0)
    offset=ratios-coarse_values(obj,baseq[None],baser[None],goal,rank.scale)[0]
    base_features=describe_batch(center[None])[0]
    refs,scales,cal=calibration_for(n,checker)
    refs=np.ascontiguousarray((refs/scales).reshape(-1,len(scales)).T)
    norms=(refs**2).sum(0)
    bags=[]
    for dy,dx in itertools.product((-1,0,1),repeat=2):
        if expired():return None
        if not dy and not dx:continue
        yy,xx=y+dy,x+dx
        outside=(yy<0)|(yy>=50)|(xx<0)|(xx>=50)
        pos=50*yy.clip(0,49)+xx.clip(0,49)
        choose=subsets[~(subsets&outside).any(axis=1)]
        proposed=np.sort(np.where(choose,pos,center),axis=1)
        valid=(np.diff(proposed,axis=1)>0).all(axis=1)
        choose=choose[valid]
        if not len(choose):
            continue
        geometry=subset_spatial(center,pos,choose)
        choose=choose.astype(np.complex64)
        proposed=proposed[valid]
        dq=(quad[pos]-quad[center]).reshape(n,-1)
        dr=(local[pos]-local[center]).reshape(n,-1)
        for start in range(0,len(choose),256):
            bits=choose[start:start+256];ids=proposed[start:start+len(bits)]
            f=(bits@dq).reshape(-1,23,len(cols))+baseq
            r=(bits@dr).reshape(-1,2,25)+baser
            ratio=coarse_values(obj,f,r,goal,rank.scale)+offset
            feature=np.tile(base_features,(len(bits),1))
            feature[:,SPATIAL]=geometry[start:start+len(bits)]
            feat=feature/scales
            squared=(feat*feat).sum(1)[:,None]+norms-2*(feat@refs)
            dist=np.sqrt(squared.reshape(len(ids),5,-1).clip(min=0))
            geom=margins(np.partition(dist,2,axis=2)[:,:,:3].mean(2),cal)[0]
            bags.append((ids,ratio,np.maximum(geom,0)))
    if not bags:return None
    ids,ratio,g=(np.concatenate(a) for a in zip(*bags))
    c=excess_score(ratio)
    ix=np.unique(np.concatenate([np.argsort(c+w*g)[:80] for w in (.2,1,4,16)]+
           [np.argsort(ratio[:,6]+w*np.maximum(ratio[:,:6]-1,0).sum(axis=1)+3*g)[:120] for w in (1,6,20)]))
    ids=ids[ix]
    keys=[row.tobytes() for row in ids]
    missing=np.array([i for i,key in enumerate(keys) if key not in cache],dtype=int)
    if len(missing):
        rank.parents=center[None]
        pending=ids[missing]
        r=rank.levels(pending,True);r[:,0]=rank.whole(pending)
        signed,legal,_=signed_sparse(pending,checker)
        cache.update((keys[i],(rr,g,ok)) for i,rr,g,ok in zip(missing,r,signed,legal))
    ratio,signed,legal=zip(*(cache[key] for key in keys))
    return ids,np.array(ratio),np.array(signed),np.array(legal)

def proposals(obj,mask,pressure,metrics,checker,expired,steps=None):
    n=int(mask.sum())
    if not 6<=n<=12 or not all(metrics["limits_ok"]) or not metrics["peak_ok"]:return
    rank=Ranker(obj,pressure)
    base=np.flatnonzero(mask)
    observed=np.r_[metrics["limits_after"],metrics["after"]]/rank.scale
    queue=[(0.,base,observed)];seen=set();best=observed[-1]
    cache={}
    winner=None
    for _ in range(2*n if steps is None else steps):
        if expired():break
        queue.sort(key=lambda item:item[0])
        next_node=next((node for node in queue if tuple(node[1]) not in seen),None)
        if next_node is None:break
        _,center,ratios=next_node;seen.add(tuple(center))
        found=neighborhood(obj,center,ratios,rank,checker,expired,cache)
        if found is None:break
        ids,r,signed,legal=found
        qualified=np.flatnonzero(legal&(r[:,:6].max(axis=1)<=1.000001)&(r[:,6]<best))
        if len(qualified):
            i=qualified[np.argmin(r[qualified,6])]
            best=r[i,6];winner=ids[i]
            if best<1:break
        c=excess_score(r)+3*np.maximum(signed,0)
        ix=np.unique(np.r_[np.argsort(c)[:30],np.argsort(r[:,6]+3*np.maximum(signed,0)+2*np.maximum(r[:,:6]-1,0).sum(axis=1))[:30]])
        queue.extend((c[i],ids[i],r[i]) for i in ix)
    if winner is not None:
        m=np.zeros_like(mask);m.flat[winner]=1
        yield m
