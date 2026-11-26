import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np

D=Path(__file__).resolve().parent
SRC=D.parent/'S231'
sys.path.insert(0,str(D.parent/'S217'))
import model
F=model.FREQ
H=int(np.flatnonzero(F==12.5)[0])
P=int(np.flatnonzero(F==5.6)[0])
K=(P,H)
CASES=json.loads((SRC/'targets.json').read_text())
SEEDS=(20261123,20261124,20261125)
coords=np.linspace(-.5,.5,5)
ix=np.searchsorted(model.AXIS,coords)-1
w=(coords-model.AXIS[ix])/(model.AXIS[ix+1]-model.AXIS[ix])
L=np.zeros((5,50))
L[np.arange(5),ix]=1-w
L[np.arange(5),ix+1]=w
def roi(p):return L@p@L.T
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        out=csv.DictWriter(f,rows[0].keys());out.writeheader();out.writerows(rows)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stem(case,seed):return f"{case['name']}_{seed}"
def initial(case,seed):
    m=np.zeros(2500,dtype='u1')
    m[np.random.default_rng(seed).permutation(2500)[:case['estimate']]]=1
    assert hashlib.sha256(m.tobytes()).hexdigest()==case['initial_hash'][str(seed)]
    return m.reshape(50,50).astype(float)
def paired(case,seed):
    name=stem(case,seed)
    s=next(r for r in read(SRC/'summary.csv') if int(r['sample'])==case['sample'] and int(r['seed'])==seed and r['method']=='norm')
    rr=[r for r in read(SRC/'center_and_roi.csv') if int(r['sample'])==case['sample'] and int(r['seed'])==seed and r['method']=='norm']
    rr=[next(r for r in rr if float(r['MHz'])==F[k]) for k in K]
    # Bounds in native reported units: percent, then Pa.
    bounds=np.array([float(s['whole_pct']),float(s['peak_abs_error_Pa']),*[abs(float(r['center_bias_Pa'])) for r in rr],*[float(r['roi_rmse_Pa']) for r in rr]])
    return bounds,float(s['high_pct']),float(s['peak_MHz'])
def goals(case):
    file=SRC/(case['name']+'.npz');assert digest(file)==case['goal_sha']
    with np.load(file) as z:a=z['amp'];np.testing.assert_equal(z['MHz'],F)
    rows=[r for r in read(SRC/'roi25_points.csv') if int(r['sample'])==case['sample'] and r['seed']==str(SEEDS[0]) and r['method']=='norm']
    b=np.zeros((23,5,5))
    for r in rows:b[list(F).index(float(r['MHz'])),list(coords).index(float(r['y_mm'])),list(coords).index(float(r['x_mm']))]=float(r['target_amp_Pa'])
    assert len(rows)==575
    return a,b
def score(p,a,b):
    e=abs(p)-a;diff=abs(roi(p[list(K)]))-b[list(K)]
    bias=e[P].mean()
    limits=np.array([100*np.linalg.norm(e)/np.linalg.norm(a),abs(bias),*abs(diff[:,2,2]),*np.sqrt((diff**2).mean(axis=(1,2)))])
    return limits,100*np.linalg.norm(e[H])/np.linalg.norm(a[H]),F[abs(p).mean(axis=(1,2)).argmax()]
