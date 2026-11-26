"""Sparse-array geometry for screening many proposals before formal classification."""
import numpy as np

from classifier import ALPHA, FAMILIES, original, band, high_band, kx, ky

CLUSTER=FAMILIES.index("cluster")
SPATIAL=np.array([0,1,2,3,6,7,8])


def calibration_for(n,checker):
    if n in checker.levels:
        i=np.flatnonzero(checker.levels==n)[0]
        return checker.ref[i],checker.scales[i],checker.cal[i]
    if n not in checker.extra:
        m=np.zeros((50,50),np.uint8)
        m.flat[:n]=1
        checker.predict(m)
    return checker.extra[n]


def footprint(x):
    lower=np.maximum(0,x-4)
    upper=np.minimum(45,x)
    overlap=np.maximum(0,np.minimum(upper[:,:,None],upper[:,None,:]) -
                       np.maximum(lower[:,:,None],lower[:,None,:])+1)
    return overlap, upper-lower+1


def spatial(ids):
    y,x=np.divmod(ids,50)
    n=x.shape[1]
    dx=x[:,:,None]-x[:,None,:]
    dy=y[:,:,None]-y[:,None,:]
    adj=((abs(dx)+abs(dy))==1).sum(axis=(1,2))
    edges=(x==0).sum(axis=1)+(x==49).sum(axis=1)+(y==0).sum(axis=1)+(y==49).sum(axis=1)
    interfaces=(4*n-edges-adj)/(4900*2*n*(2500-n)/(2500*2499))
    gx=x-x.mean(axis=1,keepdims=True)
    gy=y-y.mean(axis=1,keepdims=True)
    xx=(gx**2).mean(axis=1);yy=(gy**2).mean(axis=1);xy=(gx*gy).mean(axis=1)
    trace=xx+yy
    gap=np.sqrt((xx-yy)**2+4*xy**2)
    ratio=np.maximum(trace-gap,0)/np.maximum(trace+gap,2e-12)
    xo,xl=footprint(x);yo,yl=footprint(y)
    var=((xo*yo).sum(axis=(1,2))/46**2-((xl*yl).sum(axis=1)/46**2)**2)/625
    var/=n/2500*(1-n/2500)
    # The leftmost and uppermost points of an optimal window can be chosen
    # from the driven cells. Sliding it inside the edge preserves the count.
    xm=((dx>=0)&(dx<=8)).astype(np.int8)
    ym=((dy>=0)&(dy<=8)).astype(np.int8)
    dense=(xm.transpose(0,2,1)@ym).max(axis=(1,2))/n
    rad=np.sqrt(trace)
    return np.column_stack([original.EDGE.mean()-original.EDGE.ravel()[ids].mean(axis=1),
                            rad,interfaces,var,rad,ratio,dense])


def subset_spatial(center, dest, choose):
    """Exact seven spatial coordinates for mixtures of two positions per source."""
    n=len(center)
    old=np.column_stack(np.divmod(center,50))
    new=np.column_stack(np.divmod(dest,50))
    def additive(points):
        y,x=points.T
        edge=original.EDGE[y,x]
        borders=(points==0).sum(axis=1)+(points==49).sum(axis=1)
        covers=(np.minimum(points,45)-np.maximum(points-4,0)+1).prod(axis=1)
        return np.column_stack([x,y,x*x,y*y,x*y,edge,borders,covers])
    def pair_terms(left,right):
        lo=np.maximum(np.maximum(left[:,None,:],right[None,:,:])-4,0)
        hi=np.minimum(np.minimum(left[:,None,:],right[None,:,:]),45)
        overlap=np.maximum(hi-lo+1,0).prod(axis=2)
        adjacent=(abs(left[:,None]-right).sum(axis=2)==1).astype(int)
        return np.stack([overlap,adjacent],axis=2)
    # Total window count squared and number of adjacent pairs are quadratic
    # functions of the chosen positions. All other moments below are linear.
    old1,new1=additive(old),additive(new)
    bits=choose.astype(np.int32)
    linear=old1.sum(axis=0)+bits@(new1-old1)
    x,y,xx,yy,xy,edge,borders,covers=linear.T
    xx=xx/n-(x/n)**2;yy=yy/n-(y/n)**2;xy=xy/n-x*y/n**2
    trace=xx+yy
    gap=np.hypot(xx-yy,2*xy)
    axis=np.maximum(trace-gap,0)/np.maximum(trace+gap,2e-12)
    rad=np.sqrt(np.maximum(trace,0))
    a=pair_terms(old,old);b=pair_terms(new,old);c=pair_terms(new,new)
    lo,hi=np.triu_indices(n,1)
    delta=b-a
    order=np.arange(n)
    first=2*(delta.sum(axis=1)-delta[order,order])+c[order,order]-a[order,order]
    second=2*(c-b-b.transpose(1,0,2)+a)[lo,hi]
    joint=a.sum(axis=(0,1))+bits@first+(bits[:,lo]*bits[:,hi])@second
    variance=(joint[:,0]/46**2-(covers/46**2)**2)/(625*(n/2500)*(1-n/2500))
    interfaces=(4*n-borders-joint[:,1])/(4900*2*n*(2500-n)/(2500*2499))
    # A 9x9 window contains a union of old unmoved and new moved sources.
    starts=np.indices((42,42)).reshape(2,-1).T
    powers=1<<np.arange(n)
    def window_codes(points):
        offset=points[None]-starts[:,None]
        return ((offset>=0)&(offset<=8)).all(axis=2)@powers
    pairs=np.unique(np.column_stack([window_codes(old),window_codes(new)]),axis=0).astype(np.uint32)
    oldcodes,newcodes=pairs.T
    dominated=(((oldcodes[:,None]&oldcodes)==oldcodes[:,None])&
               ((newcodes[:,None]&newcodes)==newcodes[:,None]))
    np.fill_diagonal(dominated,False)
    oldcodes,newcodes=pairs[~dominated.any(axis=1)].T
    code=(bits@powers).astype(np.uint32)[:,None]
    occupied=(oldcodes&~code)|(newcodes&code)
    dense=np.bitwise_count(occupied).max(axis=1)/n
    return np.column_stack([original.EDGE.mean()-edge/n,rad,interfaces,variance,rad,axis,dense])


def margins(distance,cal):
    sorted_d=np.sort(distance,axis=1)
    other=np.where(distance==sorted_d[:,:1],sorted_d[:,1:2],sorted_d[:,:1])
    relative=distance/np.maximum(other,1e-12)
    rank=max(1,int(np.floor((cal.shape[-1]+1)*ALPHA/2)))
    bound=np.sort(cal,axis=-1)[:,:,-rank]
    per=np.maximum(distance/bound[:,0],relative/bound[:,1])-1
    rivals=np.delete(per,CLUSTER,axis=1).min(axis=1)
    signed=np.maximum(per[:,CLUSTER],-rivals)
    ok=(per[:,CLUSTER]<=0)&(rivals>0)
    penalty=np.where(ok,0,np.maximum(per[:,CLUSTER],0)+np.maximum(-rivals,0)+.025)
    return signed,ok,penalty


def distances(features,ref,scale):
    delta=(features[:,None,None,:]-ref[None])/scale
    d=np.sqrt(np.einsum("bcni,bcni->bcn",delta,delta))
    return np.partition(d,2,axis=2)[:,:,:3].mean(axis=2)


def describe_batch(ids):
    n=ids.shape[1]
    b=len(ids)
    v=np.empty((b,12))
    v[:,SPATIAL]=spatial(ids)
    arrays=np.zeros((b,2500))
    arrays[np.arange(b)[:,None],ids]=1.
    arrays=arrays.reshape(-1,50,50)-n/2500
    # NumPy's real transform keeps one copy of each conjugate receiver. At
    # kx=0/Nyquist both ky signs remain in this half-plane.
    native=abs(np.fft.rfft2(arrays,axes=(-2,-1)))**2
    mult=np.ones(26)*2
    mult[[0,-1]]=1
    per=original.PERIODIC[:,:26]; low=original.LOW[:,:26]
    v[:,4]=2*native[:,per].max(axis=1)/n**2
    v[:,5]=(native*mult)[:,low].sum(axis=1)/n**2
    fine=abs(np.fft.rfft2(arrays,s=(150,150),axes=(-2,-1)))/n
    take=band[:,:76]; high=high_band[:,:76]
    v[:,9]=fine[:,take].max(axis=1)
    v[:,10]=fine[:,high].max(axis=1)
    fx,fy=kx[:,:76][take],ky[:,:76][take]
    axis=(fx+1j*fy)**2/(fx**2+fy**2)
    weights=np.where(fx==0,1.,2.)
    power=fine[:,take]**2*weights
    v[:,11]=abs(power@axis)/np.maximum(power.sum(axis=1),1e-16)
    return v


def signed_sparse(ids,checker):
    ref,scale,cal=calibration_for(ids.shape[1],checker)
    pieces=[margins(distances(describe_batch(ids[j:j+24]),ref,scale),cal)
            for j in range(0,len(ids),24)]
    return tuple(np.concatenate(x) for x in zip(*pieces))
