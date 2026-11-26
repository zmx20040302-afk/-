"""Continue from a target-inferred sparse layout toward the unique cluster class."""
from browser_executor import SerialExecutor as ThreadPoolExecutor
import numpy as np

from classifier import ALPHA, band, comparable, direction, high_band, original
from cluster_geometry import calibration_for, signed_sparse
from cluster_repair import Ranker
from cluster_transfer import field


class Margins:
    def __init__(self,k,checker):
        self.k=k
        self.refs,self.scale,cal=calibration_for(k,checker)
        self.ref_matrix=np.ascontiguousarray((self.refs/self.scale).reshape(-1,len(self.scale)).T)
        self.ref_norms=(self.ref_matrix**2).sum(0)
        self.bound=np.sort(cal,axis=-1)[:,:,-max(1,int(np.floor((cal.shape[-1]+1)*ALPHA/2)))]
        c=np.arange(50)
        self.alphabet={n:np.exp((-2j*np.pi/n)*np.arange(2*49*n)).astype(np.complex64)
                       for n in (50,150)}
        for n in [5,9]:
            top=np.minimum(np.minimum(c[:,None],c[None,:]),50-n)
            bot=np.maximum(np.maximum(c[:,None]-n+1,c[None,:]-n+1),0)
            setattr(self,"win"+str(n),np.maximum(top-bot+1,0))
        self.rect=np.fft.fft2(np.ones((50,50)),s=(150,150))
        self.simple=np.fft.ifft2(original.LOW.astype(float)).real*2500
        self.quads=[]
        weights=np.zeros((150,150),complex)
        for w in [np.ones_like(direction),direction.real,direction.imag]:
            weights[band]=w
            pair=np.fft.ifft2(weights).real*150**2
            cross=np.fft.ifft2(weights*self.rect).real*150**2
            const=np.sum(weights[band].real*abs(self.rect[band])**2)
            self.quads.append((pair,cross,const))

    def adapt(self,parents):
        m=np.zeros((len(parents),2500))
        np.put_along_axis(m,parents,1,axis=1)
        masks=(m-self.k/2500).reshape(-1,50,50)
        coarse=abs(np.fft.fft2(masks,axes=(-2,-1)))
        fine=abs(np.fft.fft2(masks,s=(150,150),axes=(-2,-1)))
        self.atoms=[]
        for amps,allowed,n in [(coarse,original.PERIODIC,50),(fine,band,150),(fine,high_band,150)]:
            grid=np.flatnonzero(allowed)
            half=grid[(grid%n)<=n//2]
            ranks=np.argpartition(amps.reshape(len(m),-1)[:,half],-12,axis=1)[:,-12:]
            use=np.unique(half[ranks])
            y,x=np.divmod(np.arange(2500),50)
            phase=self.alphabet[n][y[:,None]*(use//n)+x[:,None]*(use%n)]
            offset=self.rect.flat[use]*self.k/2500 if n==150 else 0.
            self.atoms.append((phase,offset))

    def raw(self,ids):
        k=self.k
        y,x=np.divmod(ids,50)
        dy=y[:,:,None]-y[:,None,:];dx=x[:,:,None]-x[:,None,:]
        edge=np.minimum(np.minimum(x,49-x),np.minimum(y,49-y))
        c=y-y.mean(axis=1,keepdims=True);d=x-x.mean(axis=1,keepdims=True)
        yy=(c*c).mean(axis=1);xx=(d*d).mean(axis=1);xy=(c*d).mean(axis=1)
        gyr=np.sqrt(yy+xx)
        rad=np.hypot(yy-xx,2*xy);axis=(yy+xx-rad)/(yy+xx+rad)
        edges=4*k-(y==0).sum(1)-(y==49).sum(1)-(x==0).sum(1)-(x==49).sum(1)-((abs(dy)+abs(dx))==1).sum(axis=(1,2))
        win2=(self.win5[y[:,:,None],y[:,None,:]]*self.win5[x[:,:,None],x[:,None,:]]).sum(axis=(1,2))
        win=(self.win5[y,y]*self.win5[x,x]).sum(1)
        variance=(win2/46**2-(win/46**2)**2)/625
        hits=(0<=dy)&(dy<=8)
        hitsx=(0<=dx)&(dx<=8)
        dense=(hits.swapaxes(1,2).astype(np.float32)@hitsx.astype(np.float32)).max(axis=(1,2))/k
        low=self.simple[dy,dx].sum(axis=(1,2))
        energy=[]
        for mat,linear,offset in self.quads:
            energy.append(mat[dy,dx].sum(axis=(1,2))-2*k/2500*linear[y,x].sum(axis=1)+(k/2500)**2*offset)
        aniso=np.hypot(*energy[1:])/np.maximum(energy[0],1e-20)
        peaks=[abs(atom[ids].sum(axis=1)-bias).max(axis=1)/k for atom,bias in self.atoms]
        return np.column_stack([original.EDGE.mean()-edge.mean(axis=1),gyr,edges/(4900*2*k*(2500-k)/(2500*2499)),variance,
                    2*(k*peaks[0])**2/(k*(2500-k)),low/(k*(2500-k)),gyr,axis,dense,peaks[1],peaks[2],aniso])

    def margin_features(self,raw):
        n=len(raw)
        feat=comparable(raw,self.k)/self.scale
        square=(feat*feat).sum(1)[:,None]+self.ref_norms-2*(feat@self.ref_matrix)
        d=np.sqrt(np.maximum(square.reshape(n,*self.refs.shape[:2]),0))
        d=np.partition(d,2,axis=2)[:,:,:3].mean(2)
        x=np.broadcast_to(d[:,None,:],(n,5,5)).copy()
        x[:,np.arange(5),np.arange(5)]=np.inf
        relative=d/np.maximum(x.min(2),1e-12)
        margins=np.maximum(d/self.bound[:,0],relative/self.bound[:,1])-1
        return np.maximum(margins[:,1],-margins[:,[0,2,3,4]].min(1))

    def margins(self,ids):
        return np.concatenate([self.margin_features(self.raw(a)) for a in np.array_split(ids,(len(ids)+119)//120)])

def mutate(parents,rng,n):
    take=rng.integers(len(parents),size=n)
    cells=parents[take].copy()
    k=cells.shape[1]
    y,x=np.divmod(cells,50)
    center=rng.integers(k,size=(n,1))
    dy=y-np.take_along_axis(y,center,axis=1);dx=x-np.take_along_axis(x,center,axis=1)
    rv=rng.random(cells.shape)
    mode=np.arange(n)%6
    power=dy*dy+dx*dx+.9*rv
    select=np.where(mode[:,None]<2,power,rv)
    count=rng.choice([1,1,1,2,2,2,3,3,4,5,6,8],size=(n,1))
    active=np.argsort(np.argsort(select,axis=1),axis=1)<count
    scale=rng.choice([1,1,1,1,2,2,3,4,6,8,13],size=(n,1))
    jy=rng.integers(-scale,scale+1,size=cells.shape);jx=rng.integers(-scale,scale+1,size=cells.shape)
    rigid=(mode%3==0)[:,None];jy=np.where(rigid,jy[:,0,None],jy);jx=np.where(rigid,jx[:,0,None],jx)
    # Half turns and small shears preserve intra-block spacing while rearranging its placement.
    flip=(mode==4)[:,None]& (rng.random((n,1))<.6)
    yy=np.clip(y+np.where(active,jy+np.where(flip,dx-dy,0),0),0,49)
    xx=np.clip(x+np.where(active,jx+np.where(flip,-dx-dy,0),0),0,49)
    out=np.sort(50*yy+xx,axis=1)
    if len(parents)>1:
        extra=[]
        for i,a in enumerate(parents):
            for b in parents[i+1:]:
                common=np.intersect1d(a,b,assume_unique=True)
                pool=np.setxor1d(a,b,assume_unique=True)
                size=k-len(common)
                if size<2:continue
                # Fixed-count crossover keeps common driven cells and mixes the remaining locations.
                rand=rng.random((18,len(pool)))
                selected=pool[np.argpartition(rand,size-1,axis=1)[:,:size]]
                extra.append(np.column_stack([np.tile(common,(len(rand),1)),selected]))
        if extra:out=np.concatenate([out,np.sort(np.concatenate(extra),axis=1)])
    return np.unique(out[(np.diff(out,axis=1)>0).all(axis=1)],axis=0)


def reshuffle(parents,rng,n):
    k=parents.shape[1]; a=parents[rng.integers(len(parents),size=n)]
    y,x=np.divmod(a,50)
    anchor=rng.integers(k,size=(n,1));ay=np.take_along_axis(y,anchor,axis=1);ax=np.take_along_axis(x,anchor,axis=1)
    frac=rng.uniform(.2,1,(n,1));pivot=rng.random((n,1))
    for yy,ref in [(ay,y),(ax,x)]:
        yy[:,0]=np.where(pivot[:,0]<.3,np.median(ref,axis=1),yy[:,0])
    q=rng.random((n,k));dist=(ay-y)**2+(ax-x)**2
    wanted=rng.integers(1,6,(n,1))
    order=np.argsort(np.argsort(np.where(np.arange(n)[:,None]%3==0,-dist*rng.uniform(.5,1.5,(n,k)),q),axis=1),axis=1)
    selected=order<wanted
    cy=np.where(selected,y+(ay-y)*frac+rng.normal(0,2,(n,k)),y)
    cx=np.where(selected,x+(ax-x)*frac+rng.normal(0,2,(n,k)),x)
    out=np.sort(50*np.rint(cy).astype(int).clip(0,49)+np.rint(cx).astype(int).clip(0,49),axis=1)
    return out[(np.diff(out,axis=1)>0).all(axis=1)]


def acoustic_cost(ratios):
    excess = np.maximum(ratios-1,0)
    return excess.max(axis=1) + .28*excess.sum(axis=1)


def checked_geometry(ids,checker,cache):
    keys=[row.tobytes() for row in ids]
    missing=np.array([i for i,key in enumerate(keys) if key not in cache],dtype=int)
    if len(missing):
        signed,legal,_=signed_sparse(ids[missing],checker)
        cache.update((keys[i],(g,ok)) for i,g,ok in zip(missing,signed,legal))
    signed,legal=zip(*(cache[key] for key in keys))
    return np.asarray(signed),np.asarray(legal)


def whole_errors(ids,rank,cache):
    keys=[row.tobytes() for row in ids]
    missing=np.array([i for i,key in enumerate(keys) if key not in cache],dtype=int)
    if len(missing):
        cache.update((keys[i],value) for i,value in zip(missing,rank.whole(ids[missing])))
    return np.array([cache[key] for key in keys])


def select_parents(archive, count, capacity):
    keys = list(archive)
    ratios = np.stack([archive[key][0] for key in keys])
    geometry = np.array([archive[key][1] for key in keys])
    cost = acoustic_cost(ratios)
    orders = [np.argsort(geometry+w*cost) for w in (.15,.3,.7,1,2,5,10,20,45,100)]
    orders.extend(np.argsort(geometry+30*np.maximum(cost-slack,0))
                  for slack in (.01,.04,.07,.11,.16))
    orders.append(np.argsort(cost+4*np.maximum(geometry,0)))
    chosen, positions = [], []
    for shared in (count-1,count):
        for order in orders:
            for i in order[:90]:
                points=set(keys[i])
                if all(len(points & other)<shared for other in positions):
                    chosen.append(i)
                    positions.append(points)
                    break
    retained = set(np.concatenate([order[:15] for order in orders]))
    retained.update(chosen)
    return (np.array([keys[i] for i in chosen[:capacity]]),
            {keys[i]:archive[keys[i]] for i in sorted(retained)})


def proposals(obj, inferred, checker, expired, *, rounds=60, size=3000,
              seed=8473, capacity=20, trace=None):
    """Propose cluster masks, beginning with positions inferred from the target field.

    The caller scores each proposal with the original forward model and classifier.
    Both parents and rejected intermediates may be outside the requested family.
    """
    if inferred is None or expired():
        return
    cells=np.flatnonzero(inferred)
    if not 6<=len(cells)<=24:
        return
    rank=Ranker(obj,field(inferred))
    margin=Margins(len(cells),checker)
    rng=np.random.default_rng(seed)
    parents=cells[None]
    archive={}
    geometry_cache={}
    error_cache={}
    best=np.inf
    fallback=None
    # Geometry and acoustic screening use independent read-only dictionaries.
    with ThreadPoolExecutor(max_workers=2) as worker:
        for generation in range(rounds):
            if expired():
                break
            rank.parents=parents
            margin.adapt(parents)
            ids=np.unique(np.concatenate([reshuffle(parents,rng,size),mutate(parents,rng,size)]),axis=0)
            approximate=[worker.submit(margin.margins,block) for block in np.array_split(ids,2)]
            if generation:
                parent_ratios=np.stack([archive[tuple(parent)][0] for parent in parents])
            else:
                parent_ratios=rank.levels(parents,True)
                parent_ratios[:,0]=rank.whole(parents)
            offset=parent_ratios-rank.levels(parents,False)
            coarse,ancestor=rank.levels(ids,False,return_ancestors=True)
            cost=acoustic_cost(coarse+offset[ancestor])
            geom=np.concatenate([job.result() for job in approximate])
            short=[np.argsort(geom+w*cost)[:60] for w in (.2,.8,2,8,20,80)]
            short.extend(np.argsort(geom+20*np.maximum(cost-slack,0))[:35]
                         for slack in (.04,.10,.18))
            ids=ids[np.unique(np.concatenate(short))]
            exact=worker.submit(checked_geometry,ids,checker,geometry_cache)
            ratios=rank.levels(ids,True)
            ratios[:,0]=whole_errors(ids,rank,error_cache)
            signed,legal=exact.result()
            for points,values,g in zip(ids,ratios,signed):
                archive[tuple(points)]=(values,g)
            qualifying=np.flatnonzero(legal & (ratios[:,:6].max(axis=1)<1.00001) & (ratios[:,6]<best))
            if trace is not None:
                passing=np.flatnonzero(ratios.max(axis=1)<1)
                trace.append((generation,len(ids),float(signed[passing].min()) if len(passing) else None))
            if len(qualifying):
                i=qualifying[np.argmin(ratios[qualifying,6])]
                best=ratios[i,6]
                result=np.zeros_like(inferred)
                result.flat[ids[i]]=1
                if best<1.000001:
                    yield result
                else:
                    fallback=result
            parents,archive=select_parents(archive,len(cells),capacity)
            # After building a diverse archive, try joint two-source displacements
            # before spending the rest of the budget on further generations.
            if generation+1==24 and not expired():
                from cluster_projection import proposals as paired_proposals
                yield from paired_proposals(parents,archive,rank,margin,checker,expired)
    if fallback is not None and best>=1.000001:
        yield fallback

