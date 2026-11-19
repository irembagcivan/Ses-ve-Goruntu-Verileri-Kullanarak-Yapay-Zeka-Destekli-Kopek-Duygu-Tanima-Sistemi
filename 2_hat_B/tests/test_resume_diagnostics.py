"""Resume diagnoses preserve strict validation and never invoke inference."""
import json
import pytest
from src.batch.runner import completion_diagnostic, run_batch, digest
from src.batch.diagnose_resume import diagnose


def make_attempt(root, identity):
    from src.batch.runner import ARTIFACTS
    root.mkdir()
    for name in ARTIFACTS: (root/name).write_text('content')
    (root/'pose.metadata.json').write_text(json.dumps({'run_status':'complete','frames_written':1}))
    (root/'motion_summary.json').write_text(json.dumps({'status':'complete','total_frames':1,
                                                       'input_sha256':digest(root/'pose.jsonl')}))
    receipt={'status':'complete','identity':identity,'result':{'frames':1},
             'artifacts':{n:digest(root/n) for n in ARTIFACTS}}
    (root/'complete.json').write_text(json.dumps(receipt))
    (root/'started.json').write_text(json.dumps({'identity':identity}))


@pytest.mark.parametrize('key',['input_sha256','git','code_sha256','runtime','pose_sha256',
                                'detector_sha256','pose_config_sha256','confidence'])
def test_changed_groups(tmp_path,key):
    identity={'input_sha256':'a','settings':{}}
    attempt=tmp_path/'attempt';make_attempt(attempt,identity)
    changed=json.loads(json.dumps(identity))
    if key=='input_sha256': changed[key]='b'
    else: changed['settings'][key]='private-value'
    diagnostic,result=completion_diagnostic(attempt,changed)
    assert result is None and diagnostic['changed_groups']==[key]
    assert 'private-value' not in json.dumps(diagnostic)


@pytest.mark.parametrize('change,reason',[
    ('marker','completion_marker_missing'),('hash','artifact_hash_mismatch'),
    ('metadata','metadata_or_summary_mismatch'),('invalid','unreadable_or_invalid_completion_data')])
def test_artifact_checks(tmp_path,change,reason):
    identity={'input_sha256':'a','settings':{}}
    attempt=tmp_path/'attempt';make_attempt(attempt,identity)
    if change=='marker': (attempt/'complete.json').unlink()
    elif change=='hash': (attempt/'pose.jsonl').write_text('changed')
    elif change=='invalid': (attempt/'complete.json').write_text('{')
    else:
        path=attempt/'pose.metadata.json';path.write_text(json.dumps({'run_status':'failed','frames_written':1}))
        receipt=json.loads((attempt/'complete.json').read_text())
        receipt['artifacts']['pose.metadata.json']=digest(path)
        (attempt/'complete.json').write_text(json.dumps(receipt))
    assert completion_diagnostic(attempt,identity)[0]['reason']==reason


def test_readonly_comparison(tmp_path):
    a=tmp_path/'first';b=tmp_path/'second'
    make_attempt(a,{'input_sha256':'a','settings':{'git':{'commit':'old'}}})
    make_attempt(b,{'input_sha256':'a','settings':{'git':{'commit':'new'}}})
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=diagnose(a,b)
    assert result['comparison']['changed_groups']==['git']
    assert all(p.read_bytes()==v for p,v in before.items())


def test_report_decisions(tmp_path):
    source=tmp_path/'input';source.mkdir();(source/'dog.mov').touch()
    output=tmp_path/'output'
    def fake(source, target, settings):
        # Same minimal artifact contract used by batch tests.
        from src.batch.runner import ARTIFACTS
        for n in ARTIFACTS: (target/n).write_text('content')
        (target/'pose.metadata.json').write_text(json.dumps({'run_status':'complete','frames_written':1}))
        (target/'motion_summary.json').write_text(json.dumps({'status':'no_valid_motion','total_frames':1,
                                                            'input_sha256':digest(target/'pose.jsonl')}))
        return {'frames':1}
    a=run_batch(source,output,{},fake)
    b=run_batch(source,output,{},fake)
    c=run_batch(source,output,{},fake,resume=False)
    assert a['videos'][0]['resume_decision']['reason']=='no_prior_attempts'
    assert b['videos'][0]['resume_decision']['reason']=='verified_complete'
    assert c['videos'][0]['resume_decision']['reason']=='disabled'
