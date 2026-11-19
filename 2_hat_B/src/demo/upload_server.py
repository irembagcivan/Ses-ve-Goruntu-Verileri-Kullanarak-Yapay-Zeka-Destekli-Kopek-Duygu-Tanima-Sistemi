"""Loopback-only asynchronous video upload API with isolated job directories."""
import argparse,json,mimetypes,re,secrets,shutil,subprocess,sys,threading,uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,parse_qs,quote
from src.demo.release import active_release

MAX_BYTES=250*1024*1024
ALLOWED={'.mov','.mp4','.mkv','.webm','.avi'}

def upload_name(value):
    name=value.replace('\\','/').split('/')[-1].strip()
    if not name or len(name)>180 or Path(name).suffix.lower() not in ALLOWED:raise ValueError('Desteklenen video dosyası seç: MOV, MP4, MKV, WEBM, AVI')
    return name

class App:
    def __init__(self,args):
        self.args=args;self.token=secrets.token_urlsafe(32);self.jobs={};self.lock=threading.Lock();self.pool=ThreadPoolExecutor(max_workers=1);args.jobs.mkdir(parents=True,exist_ok=True)
        from src.demo.documents import Documents
        self.documents=Documents(args.jobs/'_documents',Path(__file__).resolve().parents[2])
        for directory in args.jobs.iterdir():
            if not directory.is_dir() or not re.fullmatch('[0-9a-f]{32}',directory.name):continue
            try:
                metadata=json.loads((directory/'job.json').read_text()) if (directory/'job.json').exists() else {'id':directory.name,'name':'Önceki analiz '+directory.name[:6]}
                if not isinstance(metadata,dict):continue
                if (directory/'result.json').exists():json.loads((directory/'result.json').read_text());metadata['status']='done'
                elif metadata.get('status')=='failed':pass
                else:metadata.update(status='failed',error='Önceki oturumda analiz tamamlanmadı; yeniden yükleyebilirsin.')
                metadata['id']=directory.name;self.jobs[directory.name]=metadata
            except (ValueError,OSError):pass
    def reserve(self,name):
        with self.lock:
            if sum(j['status'] in {'uploading','queued','running'} for j in self.jobs.values())>=3:raise OverflowError('En çok üç analiz sırada olabilir')
            key=uuid.uuid4().hex;self.jobs[key]={'id':key,'name':name,'status':'uploading'};return key
    def work(self,key):
        with self.lock:self.jobs[key]['status']='running'
        job=self.args.jobs/key
        command=[sys.executable,'-m','src.demo.analyze_upload','--video',str(next(job.glob('source.*'))),'--output',str(job),
            '--segmenter',str(self.args.segmenter),'--model',str(self.args.model),'--weights',str(self.args.weights),'--detector',str(self.args.detector),'--pose',str(self.args.pose)]
        try:
            result=subprocess.run(command,cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,timeout=900)
            (job/'analysis.log').write_text(result.stdout+'\n'+result.stderr)
            if result.returncode or not (job/'result.json').exists():raise RuntimeError('Video çözümlenemedi veya gerekli model kullanılamadı. Başka bir video deneyebilirsin.')
            with self.lock:self.jobs[key]['status']='done'
        except Exception as error:
            with self.lock:self.jobs[key].update(status='failed',error=str(error)[:300])
    def snapshot(self,key):
        with self.lock:result=dict(self.jobs[key])
        progress=self.args.jobs/key/'progress.json'
        if progress.exists():
            try:result.update(json.loads(progress.read_text()))
            except (ValueError,OSError):pass
        if result['status']=='done':result['result']=json.loads((self.args.jobs/key/'result.json').read_text())
        return result

def handler(app):
    class Handler(BaseHTTPRequestHandler):
        def valid_host(self):return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        def send_json(self,value,status=200):
            raw=json.dumps(value,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if not self.valid_host():return self.send_json({'error':'Invalid host'},403)
            path=urlsplit(self.path).path
            if path in {'/','/analyze'}:
                template=Path(__file__).with_name('upload.html').read_text().replace('"__CANINE_APP_TOKEN__"',json.dumps(app.token));template=template.replace('<a href="/gallery">127 videolu galeri ve model karşılaştırmaları</a>','<p>Yeni video analizi · galeri bu pakette bulunmuyor.</p>') if not app.args.gallery.exists() else template;raw=template.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();return self.wfile.write(raw)
            if path in {'/evaluation','/presentation'}:
                category='report' if path=='/evaluation' else 'slides'
                raw=Path(__file__).with_name('documents.html').read_text().replace('__CATEGORY__',category).encode()
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();return self.wfile.write(raw)
            doc=re.fullmatch(r'/api/documents/(report|slides)(?:/([a-f0-9]{32}|builtin-report|builtin-slides)/file)?',path)
            if doc:
                if not doc[2]:return self.send_json(app.documents.list(doc[1]))
                try:record,file=app.documents.file(doc[1],doc[2])
                except KeyError:return self.send_json({'error':'Belge bulunamadı'},404)
                raw=file.read_bytes();extension='.html' if record.get('builtin') else Path(record['name']).suffix.lower()
                if record.get('builtin'):
                    base='/' if doc[2]=='builtin-report' else '/docs/'
                    raw=raw.replace(b'<meta charset=',('<base href="'+base+'"><meta charset=').encode(),1)
                self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type('file'+extension)[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(raw)));self.send_header('X-Content-Type-Options','nosniff')
                if not record.get('builtin'):
                    disposition='inline' if extension=='.pdf' else 'attachment'
                    self.send_header('Content-Disposition',disposition+"; filename*=UTF-8''"+quote(record['name']))
                self.end_headers();return self.wfile.write(raw)
            if path in {'/start','/BASLA.html','/evaluation','/DEGERLENDIRME.html','/setup','/KURULUM.html','/presentation','/docs/hoca_sunumu.html'}:
                file=Path(__file__).resolve().parents[2]/{'/start':'BASLA.html','/BASLA.html':'BASLA.html','/evaluation':'DEGERLENDIRME.html','/DEGERLENDIRME.html':'DEGERLENDIRME.html','/setup':'KURULUM.html','/KURULUM.html':'KURULUM.html','/presentation':'docs/hoca_sunumu.html','/docs/hoca_sunumu.html':'docs/hoca_sunumu.html'}[path]
                if not file.is_file():return self.send_json({'error':'Page not exported'},404)
                raw=file.read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.end_headers();return self.wfile.write(raw)
            if path=='/api/health':
                metadata=json.loads(app.args.model.with_suffix('.json').read_text()) if app.args.model.with_suffix('.json').exists() else {}
                return self.send_json({'application':'canine-local-analyzer','version':6,'annotation_version':metadata.get('annotation_version','legacy-v12'),'capabilities':['video_upload','instance_segmentation','temporal_models','gallery_upload_panel','single_modality_inference','session_refresh','persistent_launcher']})
            if path=='/api/session':
                return self.send_json({'token':app.token})
            if path=='/gallery':
                file=app.args.gallery
                if not file.is_file():return self.send_json({'error':'Gallery not exported'},404)
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(file.stat().st_size));self.end_headers()
                with file.open('rb') as stream:shutil.copyfileobj(stream,self.wfile)
                return
            if path=='/api/jobs':
                with app.lock:rows=[{k:v for k,v in job.items()} for job in app.jobs.values()]
                return self.send_json(rows)
            match=re.fullmatch(r'/api/jobs/([0-9a-f]{32})(?:/(video.mp4|evidence.jpg|segmented.mp4|mask.png|mask_overlay.jpg|instance_masks.zip|segmentation.json))?',path)
            if not match or match[1] not in app.jobs:return self.send_json({'error':'Not found'},404)
            if not match[2]:return self.send_json(app.snapshot(match[1]))
            asset={'segmented.mp4':'segmented.mp4','mask.png':'mask.png','mask_overlay.jpg':'overlay.jpg','instance_masks.zip':'instance_masks.zip','segmentation.json':'segmentation.json'}.get(match[2]);file=(app.args.jobs/match[1]/'segmentation'/asset) if asset else app.args.jobs/match[1]/match[2]
            if app.jobs[match[1]]['status']!='done' or not file.is_file():return self.send_json({'error':'Not found'},404)
            size=file.stat().st_size;start,end=0,size-1;status=200;range_header=self.headers.get('Range')
            if range_header:
                selected=re.fullmatch(r'bytes=(\d+)-(\d*)',range_header)
                if not selected:return self.send_json({'error':'Invalid range'},416)
                start=int(selected[1]);end=min(size-1,int(selected[2])) if selected[2] else size-1
                if start>end:return self.send_json({'error':'Invalid range'},416)
                status=206
            self.send_response(status);self.send_header('Content-Type',mimetypes.guess_type(file.name)[0] or 'application/octet-stream');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
            if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            with file.open('rb') as stream:
                stream.seek(start);remaining=end-start+1
                while remaining:
                    chunk=stream.read(min(1024*1024,remaining))
                    if not chunk:break
                    self.wfile.write(chunk);remaining-=len(chunk)
        def do_POST(self):
            if not self.valid_host() or self.headers.get('X-Canine-Token')!=app.token:return self.send_json({'error':'Oturum değişti veya istek geçersiz. Sayfayı yenile.'},403)
            origin=self.headers.get('Origin')
            if origin and origin not in {f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}:return self.send_json({'error':'Invalid origin'},403)
            parsed=urlsplit(self.path)
            document=re.fullmatch(r'/api/documents/(report|slides)',parsed.path)
            if document:
                try:
                    from src.demo.documents import MAX_DOCUMENT_BYTES
                    length=int(self.headers.get('Content-Length','0'))
                    if length<=0 or length>MAX_DOCUMENT_BYTES:return self.send_json({'error':'Dosya boş veya 50 MB sınırını aşıyor.'},413)
                    self.connection.settimeout(60);raw=self.rfile.read(length)
                    if len(raw)!=length:raise ValueError('Dosya aktarımı tamamlanmadı.')
                    record=app.documents.add(document[1],parse_qs(parsed.query).get('name',[''])[0],raw)
                    return self.send_json(record,201)
                except ValueError as error:return self.send_json({'error':str(error)},400)
            if parsed.path!='/api/jobs':return self.send_json({'error':'Not found'},404)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length<=0 or length>MAX_BYTES:return self.send_json({'error':'Dosya 250 MB sınırını aşıyor veya boş'},413)
                name=upload_name(parse_qs(parsed.query).get('name',[''])[0]);key=app.reserve(name)
            except ValueError as error:return self.send_json({'error':str(error)},400)
            except OverflowError as error:return self.send_json({'error':str(error)},429)
            directory=app.args.jobs/key;directory.mkdir();(directory/'job.json').write_text(json.dumps(app.jobs[key],ensure_ascii=False));file=directory/('source'+Path(name).suffix.lower());temporary=file.with_suffix('.part');self.connection.settimeout(60)
            try:
                with temporary.open('wb') as output:
                    remaining=length
                    while remaining:
                        chunk=self.rfile.read(min(1024*1024,remaining))
                        if not chunk:raise ValueError('Dosya aktarımı yarım kaldı')
                        output.write(chunk);remaining-=len(chunk)
                temporary.replace(file)
                with app.lock:app.jobs[key]['status']='queued'
                app.pool.submit(app.work,key);self.send_json({'id':key,'status':'queued'},202)
            except Exception:
                temporary.unlink(missing_ok=True)
                with app.lock:app.jobs[key].update(status='failed',error='Dosya aktarımı başarısız')
                self.send_json({'error':'Dosya aktarımı başarısız'},400)
        def do_DELETE(self):
            if not self.valid_host() or self.headers.get('X-Canine-Token')!=app.token:return self.send_json({'error':'Oturumu yenileyip tekrar dene.'},403)
            origin=self.headers.get('Origin')
            if origin and origin not in {f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}:return self.send_json({'error':'Invalid origin'},403)
            match=re.fullmatch(r'/api/documents/(report|slides)/([a-f0-9]{32}|builtin-report|builtin-slides)',urlsplit(self.path).path)
            if not match:return self.send_json({'error':'Not found'},404)
            try:app.documents.delete(match[1],match[2])
            except KeyError:return self.send_json({'error':'Belge bulunamadı'},404)
            return self.send_json({'deleted':True})
        def log_message(self,*args):pass
    return Handler

if __name__=='__main__':
    release=active_release()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=release.get('port',8766));parser.add_argument('--jobs',type=Path,default=Path('results/live_analysis/jobs'));parser.add_argument('--gallery',type=Path,default=Path(release['gallery']))
    for name,default in [('model',release['model']),('weights','models/yamnet.h5'),('segmenter','models/yolov8s-seg.pt'),('detector','models/yolov8n.pt'),('pose','models/pose/superanimal_quadruped_rtmpose_s.pt')]:parser.add_argument('--'+name,type=Path,default=Path(default))
    args=parser.parse_args()
    for field in ['jobs','model','weights','detector','pose','segmenter','gallery']:setattr(args,field,getattr(args,field).resolve())
    for field in ['model','weights','detector','pose','segmenter']:
        if not getattr(args,field).is_file():parser.error('Missing checkpoint: '+field)
    app=App(args);server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(app));print(f'Yeni video analizi: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();app.pool.shutdown(wait=False,cancel_futures=True)
