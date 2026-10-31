"""Loopback-only UI server with bounded uploads and one serial inference worker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit, parse_qs
import uuid

from .engine import Engine, NAMES, clean, ROOT
from .extraction import extract_video, readiness

MAX_UPLOAD=100*1024*1024
STATIC=ROOT/'web'
JOBS=ROOT/'sessions'
METHODS=set(NAMES)

class Application:
    def __init__(self):
        self.engine=Engine();self.jobs={};self.lock=threading.Lock()
        self.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='m5-inference')
        JOBS.mkdir(parents=True,exist_ok=True)

    def validate(self,audio,visual,method,source):
        if audio not in ('learned','handcrafted','combined') or visual not in ('hc','deep','combined') or method not in METHODS:
            raise ValueError('Unknown feature or fusion option.')
        if method not in self.engine.methods(audio,visual):raise ValueError('No trained fusion model supports these features.')
        if source=='video' and visual!='hc':raise ValueError('Raw video supports HC320 visual features. Deep and combined visual features are available in validation replay.')

    def create(self,options):
        with self.lock:
            if sum(j['status'] in ('queued','running') for j in self.jobs.values())>=3:
                raise ValueError('Three analysis jobs are active. Wait for one to finish.')
            token=uuid.uuid4().hex;folder=JOBS/token;folder.mkdir()
            self.jobs[token]={'id':token,'status':'queued','stage':'Queued for analysis','created':time.time(),'options':options}
        return token,folder

    def progress(self,token,stage):
        with self.lock:self.jobs[token].update(status='running',stage=stage)

    def run(self,token,folder,options,path=None):
        try:
            audio,visual,method=(options[k] for k in ('audio','visual','method'))
            if options['source']=='video':
                d,meta=extract_video(path,folder,lambda s:self.progress(token,s),options.get('face_mode','mediapipe'),audio)
                meta['original_filename']=options.get('filename','video')
            else:
                self.progress(token,'Reading validation features and verifying the sample ID')
                d,meta=self.engine.cached(options['sample_id'],audio,visual)
            self.progress(token,'Loading the locked checkpoint for fusion and explanation')
            result=self.engine.predict(d,meta,method,audio,visual)
            result['job_id']=token
            target=folder/'result.json';partial=folder/'result.json.partial'
            partial.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False));partial.replace(target)
            with self.lock:self.jobs[token].update(status='completed',stage='Analysis complete',elapsed_sec=round(time.time()-self.jobs[token]['created'],2),result=result)
        except Exception as exc:
            with self.lock:self.jobs[token].update(status='failed',stage='Analysis failed',error=str(exc))

    def catalog(self):
        return {'class_names':list(__import__('fusion').CLASSES),'raw_video':readiness(),
                'samples':self.engine.samples(),'methods':NAMES,
                'compatibility':{a:{v:self.engine.methods(a,v) for v in ('hc','deep','combined')} for a in ('learned','handcrafted','combined')},
                'raw_audio':['learned','handcrafted','combined'],'raw_visual':['hc'],'max_upload_mb':100,'max_duration_sec':30,
                'seed':42,'notice':'Videos are processed locally using WavLM audio and handcrafted visual features. Deep and combined visual representations are available in validation replay.'}

def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args):pass

        def check_origin(self):
            expected=f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') not in (expected,f'localhost:{self.server.server_port}'):
                raise ValueError('Only the local demo origin is accepted.')
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://{expected}',f'http://localhost:{self.server.server_port}'):
                raise ValueError('Cross-origin requests are not accepted.')

        def send_json(self,value,status=200):
            data=json.dumps(clean(value),ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.headers_common('application/json; charset=utf-8',len(data));self.end_headers();self.wfile.write(data)

        def headers_common(self,mime,length):
            self.send_header('Content-Type',mime);self.send_header('Content-Length',str(length))
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")

        def send_file(self,path,download=False):
            if not path.is_file():return self.send_json({'error':'File not found.'},404)
            size=path.stat().st_size;start,end=0,size-1;status=200
            value=self.headers.get('Range')
            if value:
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',value)
                if not match:return self.send_json({'error':'Unsupported Range'},416)
                start=int(match[1]);end=min(int(match[2]) if match[2] else size-1,size-1)
                if start>end:return self.send_json({'error':'Invalid Range'},416)
                status=206
            self.send_response(status)
            self.headers_common(mimetypes.guess_type(path.name)[0] or 'application/octet-stream',end-start+1)
            self.send_header('Accept-Ranges','bytes')
            if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            if download:self.send_header('Content-Disposition',f'attachment; filename="{path.name}"')
            self.end_headers()
            with path.open('rb') as f:
                f.seek(start);remaining=end-start+1
                while remaining:
                    block=f.read(min(1024*1024,remaining))
                    if not block:break
                    self.wfile.write(block);remaining-=len(block)

        def do_GET(self):
            try:
                self.check_origin();path=urlsplit(self.path).path
                static={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
                if path in static:return self.send_file(STATIC/static[path])
                if path=='/api/catalog':return self.send_json(app.catalog())
                match=re.fullmatch(r'/api/jobs/([a-f0-9]{32})(?:/files/([a-zA-Z0-9_.-]+))?',path)
                if not match:return self.send_json({'error':'Not found'},404)
                token,name=match.groups()
                with app.lock:job=app.jobs.get(token)
                if job is None:return self.send_json({'error':'Unknown session, or the server has restarted.'},404)
                if name:
                    if name not in ('result.json','extracted_features.npz','audio.wav','input.mp4','input.mov','input.avi','input.webm','input.mkv') and not re.fullmatch(r'face_\d{2}\.jpg',name):
                        return self.send_json({'error':'Not found'},404)
                    return self.send_file(JOBS/token/name,download=name in ('result.json','extracted_features.npz'))
                return self.send_json(dict(job))
            except (BrokenPipeError,ConnectionResetError):pass
            except ValueError as exc:self.send_json({'error':str(exc)},400)

        def do_POST(self):
            token=None
            try:
                self.check_origin();url=urlsplit(self.path)
                if url.path not in ('/api/upload','/api/replay'):return self.send_json({'error':'Not found'},404)
                length=int(self.headers.get('Content-Length','0'))
                if length<=0 or length>MAX_UPLOAD:raise ValueError('File is empty or exceeds 100 MB.')
                if url.path=='/api/replay':
                    if length>16384:raise ValueError('Request too large')
                    options=json.loads(self.rfile.read(length));options['source']='replay'
                    app.validate(options['audio'],options['visual'],options['method'],'replay')
                    if options['sample_id'] not in app.engine.samples():raise ValueError('Unknown validation sample ID.')
                    token,folder=app.create(options);path=None
                else:
                    query=parse_qs(url.query);options={k:query.get(k,[v])[0] for k,v in {'audio':'learned','visual':'hc','method':'concat','face_mode':'mediapipe'}.items()}
                    options['source']='video'
                    app.validate(options['audio'],options['visual'],options['method'],'video')
                    if options['face_mode'] not in ('mediapipe','center'):raise ValueError('Unknown face mode')
                    if not readiness()['ready']:raise ValueError('Video extraction dependencies are not ready. Run setup first.')
                    filename=self.headers.get('X-File-Name','video.mp4')
                    from urllib.parse import unquote
                    filename=Path(unquote(filename)).name
                    suffix=Path(filename).suffix.lower()
                    if suffix not in ('.mp4','.mov','.avi','.webm','.mkv'):raise ValueError('Upload an MP4/MOV/AVI/WebM/MKV video.')
                    options['filename']=filename
                    token,folder=app.create(options);path=folder/('input'+suffix)
                    self.connection.settimeout(120)
                    with path.open('xb') as f:
                        left=length
                        while left:
                            block=self.rfile.read(min(left,1024*1024))
                            if not block:raise ValueError('Upload interrupted.')
                            f.write(block);left-=len(block)
                app.worker.submit(app.run,token,folder,options,path)
                self.send_json({'job_id':token},202)
            except (ValueError,KeyError,json.JSONDecodeError) as exc:
                if token:
                    with app.lock:app.jobs[token].update(status='failed',stage='Upload failed',error=str(exc))
                self.send_json({'error':str(exc)},400)
            except (BrokenPipeError,ConnectionResetError):pass
            except (OSError,TimeoutError) as exc:
                if token:
                    with app.lock:app.jobs[token].update(status='failed',stage='Upload failed',error=str(exc))
                self.send_json({'error':'Upload interrupted or timed out. Select the video again.'},400)
    return Handler

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8765);args=p.parse_args()
    app=Application();server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(app))
    print(f'M5 demo ready: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();app.worker.shutdown(wait=False,cancel_futures=True)

if __name__=='__main__':main()
