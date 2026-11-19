import argparse,json,threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import numpy as np
import pytest
import torch
from src.deep_learning.sequence import SequenceNet,scale_fit
from src.demo.upload_server import App,handler,upload_name

@pytest.mark.parametrize('kind',['cnn','gru','bigru'])
def test_temporal_network_ignores_padding(kind):
    torch.manual_seed(42);model=SequenceNet([4,3,4,4,3],kind).eval();x=torch.randn(2,6,1141);lengths=torch.tensor([3,5])
    y=x.clone();y[0,3:]=500;y[1,5:]=-500
    with torch.inference_mode():a=model(x,lengths);b=model(y,lengths)
    assert all(torch.allclose(left,right,atol=1e-6) for left,right in zip(a,b))


def test_gru_uses_time_order():
    torch.manual_seed(42);net=SequenceNet([4,3,4,4,3],'gru').eval();x=torch.randn(1,5,1141)
    with torch.inference_mode():a=net(x,torch.tensor([5]));b=net(x.flip(1),torch.tensor([5]))
    assert any(not torch.allclose(left,right) for left,right in zip(a,b))


def test_sequence_scaler_excludes_test_and_padding():
    x=np.zeros((2,4,3));x[0,:2]=2;x[0,2:]=999;x[1]=10000
    assert scale_fit(x,np.array([2,4]),np.array([0]))['mean']==[2.,2.,2.]

@pytest.mark.parametrize('name',['bad.exe','','file.txt'])
def test_reject_non_video_upload_names(name):
    with pytest.raises(ValueError):upload_name(name)


def test_upload_filename_never_controls_storage_path():
    assert upload_name('../../escape.MOV')=='escape.MOV'
    assert upload_name('C:\\private\\test.mp4')=='test.mp4'


def test_upload_api_token_origin_limits_and_queue(tmp_path):
    args=argparse.Namespace(jobs=tmp_path,model=tmp_path/'model',weights=tmp_path/'weights',detector=tmp_path/'detector',pose=tmp_path/'pose',gallery=tmp_path/'gallery')
    app=App(args);app.work=lambda key:None
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(app));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def request(method,path,body=None,headers=None):
        connection=HTTPConnection('127.0.0.1',server.server_port);connection.request(method,path,body=body,headers=headers or {});response=connection.getresponse();content=response.read();status=response.status;connection.close();return status,json.loads(content)
    try:
        assert request('GET','/api/health')[0]==200
        assert request('GET','/api/health',headers={'Host':'evil.example'})[0]==403
        assert request('POST','/api/jobs?name=test.mov',b'abc')[0]==403
        headers={'X-Canine-Token':app.token}
        assert request('POST','/api/jobs?name=test.mov',b'abc',headers|{'Origin':'https://evil.example'})[0]==403
        assert request('POST','/api/jobs?name=test.exe',b'abc',headers)[0]==400
        status,job=request('POST','/api/jobs?name=../../test.mov',b'abc',headers)
        assert status==202 and (tmp_path/job['id']/'source.mov').read_bytes()==b'abc'
        assert request('GET','/api/jobs/'+job['id'])[1]['name']=='test.mov'
        assert request('GET','/api/jobs/'+job['id']+'/../../model')[0]==404
    finally:server.shutdown();server.server_close();app.pool.shutdown(wait=True)


def test_completed_upload_history_survives_restart(tmp_path):
    key='a'*32;directory=tmp_path/key;directory.mkdir()
    (directory/'job.json').write_text(json.dumps({'id':key,'name':'prior.mov','status':'queued'}))
    (directory/'result.json').write_text(json.dumps({'schema_version':1}))
    args=argparse.Namespace(jobs=tmp_path)
    app=App(args)
    try:
        assert app.jobs[key]['status']=='done'
        assert app.snapshot(key)['name']=='prior.mov'
    finally:app.pool.shutdown(wait=True)
