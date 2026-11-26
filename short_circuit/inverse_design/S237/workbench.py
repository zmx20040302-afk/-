"""50×50 open-circuit PMUT local workbench."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys, json, base64, binascii, time, webbrowser
from pathlib import Path
from urllib.request import urlopen
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'S217'))
import server as legacy
from model import np,FREQ,mask_array,forward,solve,conv,N,G,FB
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(HERE.parent/'S232'))
from common import score,roi,H

APP = 'pmut-program-v3'

def read_target(encoded):
    raw = base64.b64decode(encoded, validate=True)
    target = np.frombuffer(raw, dtype='<c16').reshape(23, 50, 50)
    if not np.all(np.isfinite(target)):
        raise ValueError('目标包含非有限值')
    return target
def encode(p): return base64.b64encode(p.astype('<c16',copy=False).tobytes()).decode()
def protected(m,target):
    a=abs(target);b=abs(roi(target))
    if not np.any(a[H]) or not np.isfinite(np.linalg.norm(a)):raise ValueError('目标的12.50 MHz整面声压须有有限的非零幅值')
    start=time.perf_counter();end=start+30
    p,u=forward(m,range(23));bounds,base_high,peak=score(p,a,b)
    original=m.copy();best=base_high;trials=accepted=0
    while best>1e-10 and time.perf_counter()<end:
        phase=np.divide(p[H],abs(p[H]),out=np.ones_like(p[H]),where=abs(p[H])>0)
        adj=solve(m,H,conv(G[H].conj(),(abs(p[H])-a[H])*phase),True)
        grad=(adj.conj()*(N[H]+FB[H]*u[H])).real.ravel()
        on=np.flatnonzero(m);off=np.flatnonzero(1-m)
        down=on[np.argsort(-grad[on])];up=off[np.argsort(grad[off])]
        moved=False
        for size,i,j in [(32,0,0),(12,0,0),(4,0,0),(2,0,0),(1,0,0),(1,1,0),(1,0,1),(1,1,1),(1,2,0),(1,0,2)]:
            if size+i>len(on) or size+j>len(off):continue
            trial=m.copy();trial.flat[down[i:i+size]]=0;trial.flat[up[j:j+size]]=1
            waves=[];flows=[]
            for k in range(23):
                if time.perf_counter()>=end:break
                v=solve(trial,k,N[k]*trial);flows.append(v);waves.append(conv(G[k],v))
            if len(waves)!=23:break
            pp=np.asarray(waves);lim,high,f=score(pp,a,b)
            if time.perf_counter()>end:break
            trials+=1
            if high<best and f==peak and np.all(lim<=bounds):
                m,p,u,best=trial,pp,np.asarray(flows),high
                accepted+=1;moved=True;break
        if not moved:break
    return dict(mask=m.astype(int).tolist(),field=encode(p),velocity=encode(u),trials=trials,changed=int((original!=m).sum()),accepted=accepted,elapsed=time.perf_counter()-start,before=float(base_high),after=float(best))
class Handler(legacy.Handler):
    def do_GET(self):
        if self.path=='/health':return self.reply(dict(app=APP, protected_batch=True))
        if self.path=='/docs/S321/report.html':
            return self.reply((HERE.parent/'S321'/'report.html').read_bytes(),
                              mime='text/html; charset=utf-8')
        if self.path=='/docs/S308/classification.html':
            return self.reply((HERE.parent/'S308'/'classification.html').read_bytes(),
                              mime='text/html; charset=utf-8')
        if self.path in ('/','/studio.js','/studio.css'):
            filename={'/':'index.html','/studio.js':'studio.js','/studio.css':'studio.css'}[self.path]
            mime={'/':'text/html','/studio.js':'application/javascript','/studio.css':'text/css'}[self.path]
            return self.reply((HERE/filename).read_bytes(),mime=mime+'; charset=utf-8')
        if self.path=='/example':
            name='S005525_20261123'
            src=HERE.parent/'S233'
            return self.reply(dict(target=encode(np.load(src/'S005525_ref.npy')),mask=np.loadtxt(src/(name+'_base.csv'),delimiter=',',dtype=int).tolist(),candidate=np.loadtxt(HERE.parent/'S234'/(name+'_visit38.csv'),delimiter=',',dtype=int).tolist()))
        super().do_GET()
    def do_POST(self):
        if self.path not in ('/protected','/weighted','/family-design','/protected-families'):return super().do_POST()
        if self.headers.get('Origin') not in (None,legacy.URL) or self.headers.get_content_type()!='application/json':return self.send_error(403)
        if not legacy.slots.acquire(False):return self.reply(dict(error='求解器正在计算，请稍后再试。'),409)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=8000000:raise ValueError('无效的请求长度')
            body=json.loads(self.rfile.read(n))
            if not isinstance(body, dict):
                raise ValueError('请求需要包含开关矩阵和目标')
            m=mask_array(body['mask'])
            if self.path=='/protected-families':
                target=read_target(body['target'])
                if not np.any(target[H]) or not np.isfinite(np.linalg.norm(target)):
                    raise ValueError('整面目标及12.50 MHz声场须有有限的非零幅值。')
                return self.stream_families(m,target)
            if self.path=='/family-design':
                from family_design import design_family
                family = body.get('family')
                mode = body.get('mode')
                if family not in ('random', 'cluster', 'edge', 'stripe', 'patch'):
                    raise ValueError('请选择支持的排布类别')
                if mode not in ('absolute', 'weighted', 'protected'):
                    raise ValueError('未知的逆向设计模式')
                fixed = body.get('fixed', False)
                if type(fixed) is not bool:
                    raise ValueError('无效的通电数约束')
                if mode == 'protected':
                    result = design_family(m, family, mode, fixed=True, target=read_target(body['target']))
                else:
                    result = design_family(m, family, mode, goals=body['goals'], fixed=fixed)
                return self.reply(result)
            if self.path=='/weighted':
                from weighted import inverse
                fixed=body.get('fixed',False)
                if type(fixed)!=bool:raise ValueError('无效的通电数约束')
                m,metrics=inverse(m,body['goals'],fixed)
                p,u=forward(m,range(23))
                return self.reply(dict(mask=m.astype(int).tolist(),metrics=metrics,field=encode(p),velocity=encode(u)))
            target=read_target(body['target'])
            self.reply(protected(m,target))
        except (ValueError,KeyError,TypeError,OverflowError,binascii.Error) as exc:self.reply(dict(error=str(exc)),400)
        except Exception as exc:self.reply(dict(error=f'计算失败：{exc}'),500)
        finally:legacy.slots.release()

    def stream_families(self, mask, target):
        from plane_batch import design_all
        self.send_response(200)
        self.send_header('Content-Type','application/x-ndjson; charset=utf-8')
        self.send_header('Cache-Control','no-store')
        self.send_header('Connection','close')
        self.end_headers()
        self.close_connection=True
        batch=design_all(mask,target)
        try:
            for result in batch:
                line=json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':'))
                self.wfile.write((line+'\n').encode('utf-8'))
                self.wfile.flush()
        except (ConnectionError, OSError):
            pass
        except Exception as exc:
            # Headers have already been sent. The reader marks only missing families failed.
            try:
                self.wfile.write((json.dumps({'error':str(exc)},ensure_ascii=False)+'\n').encode('utf-8'))
            except OSError:
                pass
        finally:
            batch.close()
if __name__=='__main__':
    try:
        with urlopen(legacy.URL+'/health',timeout=.8) as response:active=json.load(response).get('app')
    except OSError:active=None
    if active and active!=APP:sys.exit('端口8797已有其他服务或旧版工作台')
    if not active:
        with ThreadingHTTPServer(('127.0.0.1',8797),Handler) as srv:
            if '--open' in sys.argv:webbrowser.open(legacy.URL)
            srv.serve_forever()
    elif '--open' in sys.argv:webbrowser.open(legacy.URL)
