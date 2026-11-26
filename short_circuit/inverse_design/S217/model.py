from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
with np.load(ROOT/'kernels.npz') as data:
    C,G,DIAG,FB,N,FREQ,AXIS=[data[k] for k in ('C','G','diag','feedback','N','MHz','axis')]
SHAPE=(50,50)

def mask_array(values):
    a=np.asarray(values)
    if a.shape!=SHAPE or not np.isin(a,[0,1]).all():
        raise ValueError('开关图须为 50 行 × 50 列，取值为 0 或 1')
    return a.astype(float)

def conv(kernel,v):
    padded=np.fft.fft2(v,s=(100,100),axes=(-2,-1))
    return np.fft.ifft2(kernel*padded)[:50,:50]

def solve(mask,k,rhs,adjoint=False):
    """Diagonal preconditioning and BiCGSTAB, with explicit equation residual."""
    diag=DIAG[k]+(1-mask)*FB[k];kernel=C[k]
    if adjoint:diag=diag.conj();kernel=kernel.conj()
    def op(v):return v+conv(kernel,v)/diag
    b=rhs/diag
    norm=np.linalg.norm(b)
    if norm==0:return np.zeros(SHAPE,complex)
    x=np.zeros_like(b,dtype=complex);r=b.copy();shadow=r.copy()
    rho=alpha=omega=1.;p=np.zeros_like(x);v=p.copy()
    for it in range(600):
        new=np.vdot(shadow,r)
        if new==0 or omega==0:break
        p=r+(new/rho)*(alpha/omega)*(p-omega*v)
        v=op(p);den=np.vdot(shadow,v)
        if den==0:break
        alpha=new/den;s=r-alpha*v
        x+=alpha*p
        if np.linalg.norm(s)<=norm*1e-11:break
        t=op(s);den=np.vdot(t,t)
        if den==0:break
        omega=np.vdot(t,s)/den;x+=omega*s;r=s-omega*t;rho=new
        if np.linalg.norm(r)<=norm*1e-11:break
    rel=np.linalg.norm((op(x)-b)*diag)/np.linalg.norm(rhs)
    if not np.isfinite(rel) or rel>1e-9:raise ValueError(f'{FREQ[k]:g} MHz 方程未收敛，相对残差 {rel:.2g}')
    return x

def forward(mask,indices):
    u=[];p=[]
    for k in indices:
        v=solve(mask,k,N[k]*mask);u.append(v);p.append(conv(G[k],v))
    return np.array(p),np.array(u)

def parse_goals(rows):
    # CSV is frequency MHz, centered x and y mm, amplitude Pa.
    if not isinstance(rows,list) or not 1<=len(rows)<=57500:raise ValueError('目标需有 1 至 57500 行')
    goals={}
    for i,row in enumerate(rows,1):
        if len(row)!=4:raise ValueError(f'第 {i} 行需要四列')
        f,x,y,a=map(float,row)
        if not np.isfinite([f,x,y,a]).all() or a<0:raise ValueError(f'第 {i} 行幅值或坐标不合法')
        k=int(np.argmin(abs(FREQ-f)));cx=int(np.argmin(abs(AXIS-x)));cy=int(np.argmin(abs(AXIS-y)))
        if max(abs(FREQ[k]-f),abs(AXIS[cx]-x),abs(AXIS[cy]-y))>1e-6:raise ValueError(f'第 {i} 行的频率或坐标不在原生网格')
        if k not in goals:goals[k]=np.full(SHAPE,np.nan)
        if np.isfinite(goals[k][cy,cx]):raise ValueError(f'第 {i} 行重复指定同一目标')
        goals[k][cy,cx]=a
    return np.array(sorted(goals)),np.stack([goals[k] for k in sorted(goals)])

def inverse(mask,rows,fixed_count=False,steps=6):
    ks,goal=parse_goals(rows); used=np.isfinite(goal); initial=mask.copy()
    def evaluate(m):
        p,u=forward(m,ks); diff=np.where(used,abs(p)-goal,0.)
        return float(np.sum(diff**2)),p,u,diff
    cost,p,u,err=evaluate(mask);initial_rmse=float(np.sqrt(cost/used.sum()))
    accepted=0
    for _ in range(steps):
        if cost<1e-15:break
        grad=np.zeros(SHAPE)
        for j,k in enumerate(ks):
            # At zero field choose a phase for the first descent step.
            phase=np.divide(p[j],abs(p[j]),out=np.ones_like(p[j]),where=abs(p[j])>0)
            adj=solve(mask,k,conv(G[k].conj(),err[j]*phase),True)
            grad+=np.real(adj.conj()*(N[k]+FB[k]*u[j]))
        g=grad.ravel();active=np.flatnonzero(mask);idle=np.flatnonzero(1-mask)
        changes=np.argsort(g*(1-2*mask.ravel()))
        if fixed_count:
            down=active[np.argsort(-g[active])];up=idle[np.argsort(g[idle])]
            n=min(len(up),len(down))
        else:n=int(np.count_nonzero(g*(1-2*mask.ravel())<0))
        # Test binary steps against the actual field; accept only exact improvements.
        sizes=sorted({min(n,s) for s in [1,4,16,64,256,625]},reverse=True)
        improved=False
        for size in sizes:
            if not size:continue
            trial=mask.copy()
            if fixed_count:trial.flat[down[:size]]=0;trial.flat[up[:size]]=1
            else:trial.flat[changes[:size]]=1-trial.flat[changes[:size]]
            result=evaluate(trial)
            if result[0]<cost:
                mask=trial;cost,p,u,err=result;accepted+=1;improved=True;break
        if not improved:break
    vals=abs(err[used]);target_norm=np.linalg.norm(goal[used])
    return mask,dict(items=int(used.sum()),passed=int((vals<=10).sum()),rmse=float(np.sqrt(cost/len(vals))),initial_rmse=initial_rmse,maximum=float(vals.max()),relative=float(100*np.sqrt(cost)/target_norm) if target_norm else None,accepted=accepted,changed=int((mask!=initial).sum()))

