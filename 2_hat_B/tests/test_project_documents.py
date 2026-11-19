import argparse
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from src.demo.upload_server import App,handler


def test_documents_add_download_delete_and_restart(tmp_path):
    app=App(argparse.Namespace(jobs=tmp_path/'jobs',model=tmp_path/'model',gallery=tmp_path/'gallery'))
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(app))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    def request(method,path,raw=None,token=None):
        c=HTTPConnection('127.0.0.1',server.server_port)
        c.request(method,path,body=raw,headers={'X-Canine-Token':token or ''})
        r=c.getresponse();data=r.read();headers=dict(r.getheaders());c.close();return r.status,data,headers
    try:
        for page in ['/evaluation','/presentation']:
            status,raw,_=request('GET',page)
            assert status==200 and b'__CATEGORY__' not in raw and b'Dosya ekle' in raw
        assert request('POST','/api/documents/report?name=test.pdf',b'%PDF-1.4 test')[0]==403
        assert request('POST','/api/documents/report?name=bad.html',b'<script>',app.token)[0]==400
        assert request('POST','/api/documents/report?name=bad.pdf',b'not PDF',app.token)[0]==400
        status,raw,_=request('POST','/api/documents/report?name=../raporum.pdf',b'%PDF-1.4 sample',app.token)
        assert status==201
        record=json.loads(raw);key=record['id'];assert record['name']=='raporum.pdf'
        path='/api/documents/report/'+key
        status,raw,headers=request('GET',path+'/file')
        assert status==200 and raw==b'%PDF-1.4 sample' and headers['Content-Type']=='application/pdf'
        assert request('GET','/api/documents/slides/'+key+'/file')[0]==404
        assert request('DELETE',path)[0]==403
        fresh=App(app.args)
        try:assert any(r['id']==key for r in fresh.documents.list('report'))
        finally:fresh.pool.shutdown(wait=True)
        assert request('DELETE',path,token=app.token)[0]==200
        assert request('GET',path+'/file')[0]==404
        assert not (app.documents.root/key).exists()
        assert request('DELETE','/api/documents/report/builtin-report',token=app.token)[0]==200
        assert (app.documents.project/'DEGERLENDIRME.html').exists()
        assert not any(r['id']=='builtin-report' for r in app.documents.list('report'))
    finally:
        server.shutdown();server.server_close();app.pool.shutdown(wait=True)
