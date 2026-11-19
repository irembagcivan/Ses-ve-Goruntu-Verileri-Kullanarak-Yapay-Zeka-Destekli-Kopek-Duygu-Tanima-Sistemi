import json
import subprocess
from pathlib import Path

from src.emotion.body import MODEL, infer


def test_body_model_never_claims_visible_body_with_low_detection_coverage():
    assert infer({}, {'detected_frames':3,'sampled_frames':15})['status']=='unavailable'
    meta=json.loads(MODEL.read_text())
    report=json.loads(Path('results/demo_v13_release/report.json').read_text())
    test_ids={r['video_id'] for r in report['samples'] if r['split']=='test'}
    assert not test_ids.intersection(meta['train_videos'])
    assert set(meta['classes'])=={'gergin','gevsek','cekingen_kapali'}
    prediction=infer(dict(zip(meta['columns'],meta['scaler']['median'])),{'detected_frames':10,'sampled_frames':10})
    assert abs(sum(prediction['scores'].values())-1)<1e-8
    assert prediction['scores_calibrated'] is False


def test_launcher_starts_independent_server_and_returns_without_browser(monkeypatch):
    from src.demo import launch
    probes=[];calls=[]
    monkeypatch.setattr(launch,'check',lambda root:{'missing_modules':[],'missing_files':[]})
    monkeypatch.setattr(launch.sys,'argv',['launcher','--no-browser'])
    class Response:
        def read(self):
            return json.dumps({'application':'canine-local-analyzer','version':6,'annotation_version':launch.active_release()['version']}).encode()
    def probe(*args,**kwargs):
        probes.append(args)
        if len(probes)==1:raise OSError('Server stopped')
        return Response()
    class Child:
        def poll(self):return None
    def spawn(command,**kwargs):calls.append((command,kwargs));return Child()
    monkeypatch.setattr(launch.urllib.request,'urlopen',probe)
    monkeypatch.setattr(launch.subprocess,'Popen',spawn)
    monkeypatch.setattr(launch.webbrowser,'open',lambda url: (_ for _ in ()).throw(AssertionError('No browser expected')))
    assert launch.main()==0
    assert calls[0][0][1:]==['-m','src.demo.upload_server']
    assert calls[0][1]['stdin']==subprocess.DEVNULL
    if launch.sys.platform!='win32':assert calls[0][1]['start_new_session'] is True
