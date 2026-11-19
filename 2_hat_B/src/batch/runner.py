"""Sequential orchestration with content-verified, non-destructive resume."""
import hashlib
import json
import time
from pathlib import Path

ARTIFACTS = ('skeleton.mp4', 'pose.jsonl', 'pose.metadata.json',
             'motion.csv', 'motion_summary.json')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def discover(directory):
    root = Path(directory)
    if not root.is_dir():
        raise ValueError('Input directory is missing')
    return sorted((p for p in root.iterdir() if p.is_file() and not p.is_symlink()
                   and p.suffix.lower() in {'.mov', '.mp4'}),
                  key=lambda p: (p.name.casefold(), p.name))


def video_id(path):
    # Full filename (including case/extension) avoids stem collisions; no user names in reports.
    return 'video-' + hashlib.sha256(path.name.encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def completion_diagnostic(attempt, identity):
    """Same strict resume checks, with path-free reasons and settings group names."""
    def fail(reason, **extra):
        return {'eligible': False, 'reason': reason, **extra}, None
    try:
        if attempt.is_symlink():
            return fail('unsafe_attempt')
        marker = attempt / 'complete.json'
        if not marker.is_file():
            return fail('completion_marker_missing')
        receipt = read_json(marker)
        if receipt['status'] != 'complete':
            return fail('completion_status_not_complete')
        saved = receipt['identity']
        if saved != identity:
            groups = []
            if saved.get('input_sha256') != identity.get('input_sha256'):
                groups.append('input_sha256')
            old, new = saved.get('settings', {}), identity.get('settings', {})
            # Only fixed schema labels; do not echo arbitrary keys or values.
            for key in ('runtime', 'schema_version', 'device_requested', 'detector_sha256',
                        'pose_sha256', 'pose_config_sha256', 'confidence', 'keypoint_confidence',
                        'motion_confidence', 'margin', 'visualization', 'code_sha256', 'git'):
                if old.get(key) != new.get(key): groups.append(key)
            return fail('identity_mismatch', changed_groups=groups or ['other_settings'])
        hashes = receipt['artifacts']
        if set(hashes) != set(ARTIFACTS):
            return fail('artifact_manifest_mismatch')
        for name in ARTIFACTS:
            path = attempt / name
            if path.is_symlink() or not path.is_file() or not path.stat().st_size:
                return fail('artifact_missing_empty_or_unsafe', artifact=name)
            if digest(path) != hashes[name]:
                return fail('artifact_hash_mismatch', artifact=name)
        meta = read_json(attempt / 'pose.metadata.json')
        summary = read_json(attempt / 'motion_summary.json')
        frames = meta['frames_written']
        if (meta['run_status'] != 'complete' or type(frames) is not int or frames <= 0
                or summary['total_frames'] != frames
                or summary['status'] not in {'complete', 'no_valid_motion'}
                or summary['input_sha256'] != hashes['pose.jsonl']):
            return fail('metadata_or_summary_mismatch')
        return {'eligible': True, 'reason': 'verified_complete'}, receipt['result']
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return fail('unreadable_or_invalid_completion_data')


def completed(attempt, identity):
    """Compatibility wrapper: preserve existing fail-closed resume decisions."""
    return completion_diagnostic(attempt, identity)[1]


def new_directory(parent, prefix):
    for index in range(1, 1000000):
        path = parent / f'{prefix}-{index:06d}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError('No free attempt identifier')


class LocalPipeline:
    """Load each model once, then reuse existing frame-streaming and analysis functions."""
    def __init__(self, detector_weights, pose_weights, pose_config, device, confidence, margin):
        self.paths = detector_weights, pose_weights, pose_config
        self.requested_device = device
        self.confidence, self.margin = confidence, margin
        self.models = None

    def __call__(self, source, target, settings):
        from src.pose_estimation.export import run_metadata
        from src.preprocessing.video import process_video
        from src.analysis.features import MotionSettings
        from src.analysis.report import analyze_file
        if self.models is None:
            from src.pose_estimation.device import select_device
            from src.pose_estimation.backend import DeepLabCutRTMPose
            from src.preprocessing.detection import YoloDogDetector
            from src.pose_estimation.adapter import DogPoseAdapter
            device = select_device(self.requested_device)
            detector_path, pose_path, config = self.paths
            backend = DeepLabCutRTMPose(pose_path, config, device)
            detector = YoloDogDetector(detector_path, device, self.confidence)
            self.models = detector, DogPoseAdapter(backend, self.margin), device
            self.metadata = run_metadata(detector_path, pose_path, device, self.confidence,
                                         settings['keypoint_confidence'], self.margin, 'conservative')
        detector, estimator, device = self.models
        info = process_video(source, target / 'skeleton.mp4', detector, estimator,
                             settings['keypoint_confidence'], device, dog_skeleton='conservative',
                             pose_output=target / 'pose.jsonl', pose_metadata=self.metadata)
        summary = analyze_file(target / 'pose.jsonl', target / 'motion.csv',
                               target / 'motion_summary.json',
                               MotionSettings(confidence=settings['motion_confidence']))
        return {'frames': info['frames'], 'duration_seconds': info['frames'] / info['fps'],
                'pose_export_status': 'complete', 'motion_analysis_status': summary['status'],
                'device': device}


def run_batch(input_dir, output_dir, settings, pipeline, limit=None, resume=True):
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('Limit must be a positive integer')
    sources = discover(input_dir)
    if limit is not None:
        sources = sources[:limit]
    root = Path(output_dir)
    # Avoid placing derived outputs inside the input dataset.
    if root.resolve().is_relative_to(Path(input_dir).resolve()):
        raise ValueError('Output directory must be outside input directory')
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError('Output directory must not be a symlink')
    # Exclusive mkdir also prevents concurrent batch inference into this output root.
    lock = root / '.batch.lock'
    lock.mkdir()
    records = []
    try:
        report_dir = new_directory(root, 'batch-run')
        for source in sources:
            started = time.perf_counter()
            row = {'video_id': video_id(source), 'status': 'failed', 'frames': None,
                   'duration_seconds': None, 'processing_seconds': None,
                   'pose_export_status': 'not_completed', 'motion_analysis_status': 'not_completed',
                   'error_category': None, 'attempt': None,
                   'resume_decision': {'enabled': resume, 'candidates': [],
                                       'reason': 'no_prior_attempts' if resume else 'disabled'}}
            attempt = None
            try:
                identity = {'input_sha256': digest(source), 'settings': settings}
                home = root / row['video_id']
                if home.is_symlink():
                    raise ValueError('Unsafe output directory')
                home.mkdir(exist_ok=True)
                previous = None
                if resume:
                    for candidate in sorted(home.glob('attempt-*'), reverse=True):
                        diagnostic, previous = completion_diagnostic(candidate, identity)
                        row['resume_decision']['candidates'].append(diagnostic)
                        row['resume_decision']['reason'] = ('verified_complete' if previous is not None
                                                            else 'no_eligible_attempt')
                        if previous is not None:
                            decision = row['resume_decision']
                            row.update(previous, status='skipped_verified', attempt=candidate.name)
                            row['resume_decision'] = decision
                            break
                if previous is None:
                    attempt = new_directory(home, 'attempt')
                    row['attempt'] = attempt.name
                    write_json(attempt / 'started.json', {'status': 'in_progress', 'identity': identity})
                    result = pipeline(source, attempt, settings)
                    if digest(source) != identity['input_sha256']:
                        raise ValueError('Input changed during processing')
                    row.update(result, status='complete')
                    row['processing_seconds'] = time.perf_counter() - started
                    receipt = {'status': 'complete', 'identity': identity, 'result': row.copy(),
                               'artifacts': {name: digest(attempt / name) for name in ARTIFACTS}}
                    write_json(attempt / 'complete.json', receipt)
                    if completed(attempt, identity) is None:
                        # A receipt which fails validation is never resumed as successful.
                        raise ValueError('Output completion validation failed')
            except Exception as exc:
                row.update(status='failed', error_category=type(exc).__name__)
                # Never serialize exception text: libraries can expose absolute paths/tokens.
                if attempt is not None:
                    try:
                        write_json(attempt / 'failure.json', {'status': 'failed',
                                   'error_category': type(exc).__name__})
                    except OSError:
                        pass
            row['processing_seconds'] = time.perf_counter() - started
            records.append(row)
            write_json(report_dir / f'{len(records):06d}.json', row)
        report = {'schema_version': 1, 'videos': records, 'total': len(records),
                  'failed': sum(r['status'] == 'failed' for r in records),
                  'limitations': ['MOV/MP4 only; decoder support depends on local OpenCV.',
                                  'Silent constant-FPS output; timestamps are frame_index/fps.',
                                  'No tracking or emotion classification.',
                                  'Motion analysis uses only single-dog frames.',
                                  'No valid motion is a completed analysis, not a successful movement measurement.']}
        write_json(report_dir / 'report.json', report)
        return report
    finally:
        lock.rmdir()
