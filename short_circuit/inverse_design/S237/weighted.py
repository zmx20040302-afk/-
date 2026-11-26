from model import np, parse_goals, forward, SHAPE, solve, conv, G, N, FB

def inverse(mask,rows,fixed_count=False,steps=6):
    ks,goal=parse_goals(rows); used=np.isfinite(goal); initial=mask.copy()
    energy=np.nansum(goal**2,axis=(1,2))
    if np.any(energy<=0):raise ValueError('逐频归一化需要各目标频率至少一个非零声压，可改用等权 Pa。')
    weight=1/energy
    def evaluate(m):
        p,u=forward(m,ks); diff=np.where(used,abs(p)-goal,0.)
        return float(np.sum(weight[:,None,None]*diff**2)),p,u,diff
    cost,p,u,err=evaluate(mask);initial_rmse=float(np.sqrt(np.sum(err**2)/used.sum()))
    accepted=0
    for _ in range(steps):
        if cost<1e-15:break
        grad=np.zeros(SHAPE)
        for j,k in enumerate(ks):
            # At zero field choose a phase for the first descent step.
            phase=np.divide(p[j],abs(p[j]),out=np.ones_like(p[j]),where=abs(p[j])>0)
            adj=solve(mask,k,conv(G[k].conj(),weight[j]*err[j]*phase),True)
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
    return mask,dict(items=int(used.sum()),passed=int((vals<=10).sum()),rmse=float(np.sqrt(np.mean(vals**2))),initial_rmse=initial_rmse,maximum=float(vals.max()),relative=float(100*np.linalg.norm(vals)/target_norm) if target_norm else None,accepted=accepted,changed=int((mask!=initial).sum()))

