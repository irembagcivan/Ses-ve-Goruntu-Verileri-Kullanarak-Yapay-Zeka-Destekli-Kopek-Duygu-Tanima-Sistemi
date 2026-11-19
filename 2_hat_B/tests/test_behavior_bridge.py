import json
from pathlib import Path

from src.emotion.behavior_bridge import MODEL, infer


def test_missing_or_invalid_behavior_never_forces_emotion():
    assert infer(None)['status'] == 'unavailable'
    assert infer({'scores': {}, 'predictions': {}})['status'] == 'unavailable'


def test_four_class_bridge_uses_training_only_and_returns_four_scores():
    meta = json.loads(MODEL.read_text())
    report = json.loads(Path('results/demo_v13_release/report.json').read_text())
    test_ids = {r['video_id'] for r in report['samples'] if r['split'] == 'test'}
    assert not test_ids.intersection(meta['train_videos'])
    prediction = report['samples'][0]['predictions']['neural_visual']
    scores = {t: {c: float(c == prediction[t]) for c in cs}
              for t, cs in report['classes'].items()}
    result = infer({'predictions': prediction, 'scores': scores})
    assert set(result['scores']) == {'Happy', 'Sad', 'Angry', 'Relaxed'}
    assert abs(sum(result['scores'].values()) - 1) < 1e-8
    assert result['independent_emotion_accuracy_measured'] is False
    scores['kulak'] = {'notr': float('nan')}
    assert infer({'predictions': prediction, 'scores': scores})['status'] == 'unavailable'


def test_low_detection_coverage_excludes_visual_claims_and_can_use_dog_audio():
    report = json.loads(Path('results/demo_v13_release/report.json').read_text())
    prediction = report['samples'][0]['predictions']['neural_visual']
    scores = {t: {c: float(c == prediction[t]) for c in cs}
              for t, cs in report['classes'].items()}
    comparison = {'predictions': prediction, 'scores': scores}
    quality = {'detected_frames': 3, 'sampled_frames': 15}
    assert infer(comparison, quality=quality)['status'] == 'unavailable'
    audio = {'status': 'ok', 'windows': [
        {'scores': {'Dog': .9, 'Growling': .65, 'Bark': .4, 'Speech': .01}}]}
    result = infer(comparison, quality=quality, audio_analysis=audio)
    assert result['modality'] == 'audio'
    assert not result['behavior_evidence']
    assert result['audio_evidence'][0]['event'] == 'Growling'
    audio['windows'][0]['scores']['Speech'] = .95
    assert infer(comparison, quality=quality, audio_analysis=audio)['status'] == 'unavailable'


def test_speech_or_silence_without_visible_dog_never_forces_four_emotions():
    audio = {'status': 'ok', 'windows': [{'scores': {'Speech': .9, 'Bark': .1}}]}
    assert infer(None, audio_analysis=audio)['status'] == 'unavailable'
