"""Collector tests never require model weights or ML package installation."""
import hashlib
import json
import subprocess
from types import SimpleNamespace as NS

import pytest

from scripts import collect_environment as collector


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr(collector.platform, 'python_version', lambda: '3.11.17')
    monkeypatch.setattr(collector.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(collector.platform, 'mac_ver', lambda: ('14.6', (), ''))
    monkeypatch.setattr(collector.platform, 'machine', lambda: 'arm64')
    versions = {'torch':'2.6.0','torchvision':'0.21.0','deeplabcut':'3.0.2',
                'ultralytics':'8.4.189','numpy':'2.4.6','opencv-python':'5.0.0'}
    def version(name):
        if name not in versions: raise collector.metadata.PackageNotFoundError(name)
        return versions[name]
    monkeypatch.setattr(collector.metadata, 'version', version)
    imports=[]
    def import_module(name):
        imports.append(name)
        if name == 'torch': return NS(backends=NS(mps=NS(is_available=lambda:True)))
        if name == 'cv2': return NS(__version__='5.0.0')
        raise AssertionError('Unexpected package import')
    monkeypatch.setattr(collector.importlib, 'import_module', import_module)
    monkeypatch.setattr(collector, 'git_info', lambda repo: {'commit':'a'*40,'dirty':False})
    return imports


def test_reports_observed_values_not_hardcoded_versions(isolated):
    report=collector.collect_environment()
    assert report['python_version']=='3.11.17'
    assert report['os']=={'system':'Darwin','version':'14.6'}
    assert report['architecture']=='arm64'
    assert report['packages']['numpy']['numpy']=='2.4.6'
    assert report['packages']['torch']['torch']=='2.6.0'
    assert report['runtime']['opencv_version']=='5.0.0'
    assert report['runtime']['mps_available'] is True
    assert report['git']['commit']=='a'*40
    assert isolated==['torch','cv2']


def test_missing_or_broken_packages_are_unknown(monkeypatch):
    def missing(*args): raise ImportError('/Users/secret-user/token-secret')
    monkeypatch.setattr(collector.metadata,'version',missing)
    monkeypatch.setattr(collector.importlib,'import_module',missing)
    assert collector.distribution_version('torch') is None
    runtime=collector.runtime_probe()
    assert runtime['mps_available'] is None
    assert runtime['opencv_version'] is None
    assert 'secret' not in json.dumps(runtime)


def test_mps_unavailable_is_false_not_unknown(monkeypatch):
    monkeypatch.setattr(collector.importlib,'import_module',lambda name:
        NS(backends=NS(mps=NS(is_available=lambda:False))) if name=='torch' else NS(__version__='4.10.0'))
    assert collector.runtime_probe()['mps_available'] is False


@pytest.mark.parametrize('value,expected', [('2.6.0+alice-private-host','2.6.0'),
    ('3.0.0rc1','3.0.0rc1'),('1.2.dev3','1.2.dev3'),('secret-token',None),
    ('/Users/alice/2.4.6',None),(None,None)])
def test_version_privacy(value,expected):
    assert collector.safe_version(value)==expected


def test_checkpoint_streamed_digest_without_filename(tmp_path):
    checkpoint=tmp_path/'alice-token-model.pt'
    data=b'fixture'*400000
    checkpoint.write_bytes(data)
    result=collector.checkpoint_info(checkpoint,1)
    assert result=={'id':'checkpoint_1','sha256':hashlib.sha256(data).hexdigest(),'status':'hashed'}
    assert 'alice' not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(result)


def test_checkpoint_errors_do_not_leak_paths(tmp_path):
    assert collector.checkpoint_info(tmp_path/'alice-missing.pt',1)['status']=='missing'
    assert collector.checkpoint_info(tmp_path,2)['status']=='not_a_file'


def test_checkpoint_permission_error_is_redacted(tmp_path,monkeypatch):
    checkpoint=tmp_path/'private.pt';checkpoint.write_bytes(b'fixture')
    def fail(*args,**kwargs): raise PermissionError('/Users/alice/private.pt')
    monkeypatch.setattr(collector.Path,'open',fail)
    assert collector.checkpoint_info(checkpoint,1)=={'id':'checkpoint_1','sha256':None,'status':'unreadable'}


def test_git_only_exports_commit_and_dirty(monkeypatch,tmp_path):
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        return NS(returncode=0,stdout='a'*40+'\n' if 'rev-parse' in command else '?? /Users/alice/private.txt\n')
    monkeypatch.setattr(collector.subprocess,'run',run)
    assert collector.git_info(tmp_path)=={'commit':'a'*40,'dirty':True}
    assert len(calls)==2
    assert all('remote' not in command for command in calls)


def test_missing_git_and_invalid_commit_are_unknown(monkeypatch,tmp_path):
    def fail(*args,**kwargs): raise FileNotFoundError('/Users/alice/git')
    monkeypatch.setattr(collector.subprocess,'run',fail)
    assert collector.git_info(tmp_path)=={'commit':None,'dirty':None}
    monkeypatch.setattr(collector.subprocess,'run',lambda *args,**kwargs:NS(returncode=1,stdout='username-secret'))
    assert collector.git_info(tmp_path)=={'commit':None,'dirty':None}


def test_json_cli_and_privacy_allowlist(tmp_path,monkeypatch,isolated):
    monkeypatch.setenv('API_KEY','never-export-api-key')
    monkeypatch.setenv('USER','never-export-username')
    output=tmp_path/'report.json'
    checkpoint=tmp_path/'private-user-model.pt';checkpoint.write_bytes(b'fixture')
    assert collector.main(['--output',str(output),'--checkpoint',str(checkpoint)])==0
    text=output.read_text();report=json.loads(text)
    assert set(report)=={'schema_version','python_version','os','architecture','packages','runtime','git','checkpoints'}
    for secret in ['never-export-api-key','never-export-username','private-user-model',str(tmp_path)]:
        assert secret not in text
    assert report['checkpoints'][0]['status']=='hashed'


def test_output_does_not_overwrite(tmp_path,isolated):
    output=tmp_path/'report.json';output.write_text('preserve')
    with pytest.raises(SystemExit) as exc: collector.main(['--output',str(output)])
    assert exc.value.code==2 and output.read_text()=='preserve'


def test_same_checkpoint_and_output_rejected(tmp_path,isolated):
    output=tmp_path/'report.json'
    with pytest.raises(SystemExit): collector.main(['--output',str(output),'--checkpoint',str(output)])
    assert not output.exists()


def test_output_failure_message_does_not_leak_path(tmp_path,capsys,isolated):
    with pytest.raises(SystemExit):collector.main(['--output',str(tmp_path/'secret-user'/'out.json')])
    assert 'secret-user' not in capsys.readouterr().err


def test_unknown_platform_values_are_not_exported(monkeypatch,isolated):
    monkeypatch.setattr(collector.platform,'system',lambda:'user-secret-os')
    monkeypatch.setattr(collector.platform,'machine',lambda:'alice-host')
    monkeypatch.setattr(collector.platform,'release',lambda:'6.8.0-alice-private-host')
    report=collector.collect_environment()
    assert report['os']=={'system':None,'version':'6.8.0'}
    assert report['architecture'] is None
    assert 'alice' not in json.dumps(report)
