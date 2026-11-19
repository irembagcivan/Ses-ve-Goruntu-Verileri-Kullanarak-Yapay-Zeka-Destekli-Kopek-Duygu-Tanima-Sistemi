import argparse
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from src.demo.upload_server import App,handler


def test_entry_pages_and_analyze_alias_have_valid_fresh_token(tmp_path):
    app=App(argparse.Namespace(jobs=tmp_path/'jobs',model=tmp_path/'model.pt',gallery=tmp_path/'gallery.html'))
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(app))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def get(path):
        c=HTTPConnection('127.0.0.1',server.server_port);c.request('GET',path);r=c.getresponse()
        value=(r.status,r.read().decode(),dict(r.getheaders()));c.close();return value
    try:
        for path in ['/','/analyze']:
            status,body,headers=get(path)
            assert status==200 and json.dumps(app.token) in body
            assert '__CANINE_APP_TOKEN__' not in body
            assert headers['Cache-Control']=='no-store'
        for path in ['/start','/evaluation','/setup','/presentation']:
            status,body,_=get(path);assert status==200 and '<html' in body
        status,body,_=get('/api/health');health=json.loads(body)
        assert status==200 and health['version']==6
        assert 'single_modality_inference' in health['capabilities']
        status,body,headers=get('/api/session');assert status==200 and json.loads(body)['token']==app.token
        assert headers['Cache-Control']=='no-store'
        assert get('/src/demo/upload_server.py')[0]==404
        assert get('/../README.md')[0]==404
    finally:
        server.shutdown();server.server_close();app.pool.shutdown(wait=True)
