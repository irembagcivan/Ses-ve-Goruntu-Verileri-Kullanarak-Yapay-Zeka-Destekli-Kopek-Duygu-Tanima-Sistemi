"""Orchestration tests use fake artifacts, never model inference."""
import json
from pathlib import Path
import pytest
from src.batch.runner import discover, video_id, run_batch, digest


def dump(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / 'inputs'
    source.mkdir()
    output = tmp_path / 'outputs'
    return source, output


def fake(source, target, settings):
    (target / 'skeleton.mp4').write_bytes(b'fake video')
    (target / 'pose.jsonl').write_text('{}\n')
    (target / 'motion.csv').write_text('frame_index\n0\n')
    dump(target / 'pose.metadata.json', {'run_status': 'complete', 'frames_written': 1})
    dump(target / 'motion_summary.json', {'status': 'no_valid_motion', 'total_frames': 1,
                                        'input_sha256': digest(target / 'pose.jsonl')})
    return {'frames': 1, 'duration_seconds': .04, 'pose_export_status': 'complete',
            'motion_analysis_status': 'no_valid_motion', 'device': 'cpu'}


def test_discovery(setup):
    source, _ = setup
    for name in ['b.MOV', 'A.mp4', 'a.MOV', 'c.txt', 'd.avi']:
        (source / name).touch()
    (source / 'nested').mkdir()
    (source / 'nested' / 'x.mov').touch()
    assert [p.name for p in discover(source)] == ['a.MOV', 'A.mp4', 'b.MOV']


def test_empty_report(setup):
    source, output = setup
    report = run_batch(source, output, {}, fake)
    assert report['total'] == 0
    assert len(list(output.glob('batch-run-*/report.json'))) == 1
    assert not (output / '.batch.lock').exists()


def test_resume_and_preservation(setup):
    source, output = setup
    (source / 'dog.mov').write_bytes(b'input')
    first = run_batch(source, output, {}, fake)
    snapshot = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    def forbidden(*args):
        raise AssertionError('Must not run')
    second = run_batch(source, output, {}, forbidden)
    assert first['videos'][0]['status'] == 'complete'
    assert second['videos'][0]['status'] == 'skipped_verified'
    assert all(p.read_bytes() == data for p, data in snapshot.items())


def test_failure_continues_partial_preserved(setup):
    source, output = setup
    for name in ['a.mov', 'b.mp4']:
        (source / name).touch()
    def fail_one(path, target, settings):
        if path.name == 'a.mov':
            (target / 'pose.jsonl').write_text('partial')
            raise ValueError('/private/user/secret-token')
        return fake(path, target, settings)
    result = run_batch(source, output, {}, fail_one)
    assert [r['status'] for r in result['videos']] == ['failed', 'complete']
    assert result['videos'][0]['error_category'] == 'ValueError'
    assert 'secret-token' not in json.dumps(result)
    assert next(output.glob('video-*/attempt-*/failure.json')).is_file()
    run_batch(source, output, {}, fake)
    assert any(p.read_text() == 'partial' for p in output.glob('video-*/attempt-*/pose.jsonl'))


@pytest.mark.parametrize('change', ['artifact', 'metadata', 'settings', 'source', 'receipt'])
def test_resume_invalidated(setup, change):
    source, output = setup
    video = source / 'dog.mov'
    video.write_bytes(b'input')
    run_batch(source, output, {}, fake)
    attempt = next(output.glob('video-*/attempt-*'))
    settings = {}
    if change == 'artifact':
        (attempt / 'skeleton.mp4').write_bytes(b'changed')
    elif change == 'metadata':
        dump(attempt / 'pose.metadata.json', {'run_status': 'failed', 'frames_written': 1})
    elif change == 'settings':
        settings = {'threshold': .9}
    elif change == 'source':
        video.write_bytes(b'changed')
    else:
        (attempt / 'complete.json').write_text('broken')
    result = run_batch(source, output, settings, fake)
    assert result['videos'][0]['status'] == 'complete'
    assert result['videos'][0]['attempt'] == 'attempt-000002'


def test_collisions_limit_no_resume(setup):
    source, output = setup
    # Case-insensitive filesystems cannot store dog.mov and dog.MOV separately.
    # Keep a real same-stem/different-extension collision and an uppercase suffix.
    for name in ['other.MOV', 'dog.mov', 'dog.mp4']:
        (source / name).touch()
    videos = discover(source)
    assert [p.name for p in videos] == ['dog.mov', 'dog.mp4', 'other.MOV']
    assert len({video_id(p) for p in videos}) == 3
    # Hashing names is independent of whether both names can coexist on disk.
    assert video_id(Path('dog.mov')) != video_id(Path('dog.MOV'))
    expected_ids = [video_id(p) for p in videos[:2]]
    first = run_batch(source, output, {}, fake, limit=2)
    assert first['total'] == 2
    assert [r['video_id'] for r in first['videos']] == expected_ids
    assert all(r['status'] == 'complete' for r in first['videos'])
    original = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    result = run_batch(source, output, {}, fake, limit=2, resume=False)
    assert [r['video_id'] for r in result['videos']] == expected_ids
    assert all(r['status'] == 'complete' and r['attempt'] == 'attempt-000002'
               for r in result['videos'])
    assert all(p.read_bytes() == content for p, content in original.items())
    resumed = run_batch(source, output, {}, fake, limit=2)
    assert all(r['status'] == 'skipped_verified' and r['attempt'] == 'attempt-000002'
               for r in resumed['videos'])


def test_existing_partial_and_lock(setup):
    source, output = setup
    video = source / 'dog.mov'
    video.touch()
    partial = output / video_id(video) / 'attempt-000001'
    partial.mkdir(parents=True)
    (partial / 'skeleton.mp4').write_bytes(b'keep')
    result = run_batch(source, output, {}, fake)
    assert result['videos'][0]['attempt'] == 'attempt-000002'
    assert (partial / 'skeleton.mp4').read_bytes() == b'keep'
    (output / '.batch.lock').mkdir()
    with pytest.raises(FileExistsError):
        run_batch(source, output, {}, fake)


def test_invalid_limit_and_output(setup):
    source, output = setup
    with pytest.raises(ValueError):
        run_batch(source, output, {}, fake, limit=0)
    with pytest.raises(ValueError):
        run_batch(source, source / 'outputs', {}, fake)


def test_cli_missing_weights(setup):
    from src.batch.run_batch import main
    source, output = setup
    with pytest.raises(SystemExit) as exc:
        main(['--input-dir', str(source), '--output-dir', str(output),
              '--detector-weights', 'missing.pt', '--pose-weights', 'missing.pt'])
    assert exc.value.code == 2
    assert not output.exists()


def test_existing_pipeline_reused_and_models_loaded_once(tmp_path, monkeypatch):
    from src.batch.runner import LocalPipeline
    import src.pose_estimation.backend as backend
    import src.pose_estimation.adapter as adapter
    import src.pose_estimation.device as device
    import src.pose_estimation.export as export
    import src.preprocessing.detection as detection
    import src.preprocessing.video as video
    import src.analysis.report as report
    calls = []
    sentinel = object()
    def load(*args):
        calls.append('load_pose')
        return sentinel
    def detect(*args):
        calls.append('load_detector')
        return sentinel
    monkeypatch.setattr(backend, 'DeepLabCutRTMPose', load)
    monkeypatch.setattr(detection, 'YoloDogDetector', detect)
    monkeypatch.setattr(adapter, 'DogPoseAdapter', lambda *a: sentinel)
    monkeypatch.setattr(device, 'select_device', lambda requested: requested)
    monkeypatch.setattr(export, 'run_metadata', lambda *a: {'device': 'cpu'})
    def process(source, output, detector, estimator, threshold, chosen, **kwargs):
        assert detector is sentinel and estimator is sentinel
        assert chosen == 'cpu' and kwargs['dog_skeleton'] == 'conservative'
        assert kwargs['pose_output'].name == 'pose.jsonl'
        calls.append('video')
        return {'frames': 10, 'fps': 25}
    monkeypatch.setattr(video, 'process_video', process)
    def analyze(*args):
        calls.append('motion')
        return {'status': 'complete'}
    monkeypatch.setattr(report, 'analyze_file', analyze)
    pipeline = LocalPipeline('det.pt', 'pose.pt', None, 'cpu', .25, .15)
    for _ in range(2):
        result = pipeline(Path('sample.mov'), tmp_path,
                          {'keypoint_confidence': .5, 'motion_confidence': .5})
        assert result['duration_seconds'] == .4
    assert calls == ['load_pose', 'load_detector', 'video', 'motion', 'video', 'motion']
