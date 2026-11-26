"""Local full-field programmable PMUT workbench."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import base64,json,sys,threading,webbrowser
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.request import urlopen
from model import ROOT,FREQ,AXIS,mask_array,forward,inverse,np

URL='http://127.0.0.1:8797'
slots=threading.BoundedSemaphore(1)
example=np.loadtxt(ROOT/'start.csv',delimiter=',',dtype=np.uint8)
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,data,code=200,mime='application/json; charset=utf-8'):
        raw=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers()
        try:self.wfile.write(raw)
        except (ConnectionAbortedError,BrokenPipeError,ConnectionResetError):pass
    def do_GET(self):
        if self.path=='/':return self.reply((ROOT/'index.html').read_bytes(),mime='text/html; charset=utf-8')
        if self.path=='/health':return self.reply(dict(app='pmut-fullfield-v1'))
        if self.path=='/config':return self.reply(dict(frequency=FREQ.tolist(),axis=np.round(AXIS,5).tolist(),mask=example.tolist()))
        self.send_error(404)
    def do_POST(self):
        if self.path not in ('/predict','/design'):return self.send_error(404)
        if self.headers.get('Origin') not in (None,URL) or self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send_error(403)
        if not slots.acquire(False):return self.reply(dict(error='求解器正在计算，请稍后再试。'),409)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=8000000:raise ValueError('请求大小超出限制')
            body=json.loads(self.rfile.read(size));mask=mask_array(body['mask']);metrics=None
            if self.path=='/design':
                fixed=body.get('fixed',False)
                if type(fixed)!=bool:raise ValueError('通电数约束格式有误')
                mask,metrics=inverse(mask,body['goals'],fixed)
            p,u=forward(mask,range(len(FREQ)))
            self.reply(dict(mask=mask.astype(int).tolist(),metrics=metrics,velocity=base64.b64encode(u.astype('<c16').tobytes()).decode(),field=base64.b64encode(p.astype('<c16').tobytes()).decode()))
        except (ValueError,KeyError,TypeError,OverflowError) as e:self.reply(dict(error=str(e)),400)
        except Exception as e:self.reply(dict(error=f'计算失败：{e}'),500)
        finally:slots.release()
if __name__=='__main__':
    try:
        with urlopen(URL+'/health',timeout=.7) as res:exists=json.load(res).get('app')=='pmut-fullfield-v1'
    except (ValueError,OSError):exists=False
    if exists:
        if '--open' in sys.argv:webbrowser.open(URL)
    else:
        with ThreadingHTTPServer(('127.0.0.1',8797),Handler) as server:
            if '--open' in sys.argv:webbrowser.open(URL)
            server.serve_forever()

