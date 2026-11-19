"""Synthetic JSONL quality tests; never claim model accuracy or real inference."""
import json
from copy import deepcopy
import pytest
from src.analysis.regions import model_names
from src.quality.analyze import analyze, iou


def dog(x=10, confidence=.9):
    return {'bbox_xyxy': [0,0,100,100], 'detection_confidence': .8,
            'keypoints': [{'index': i, 'name': n, 'x': x, 'y': 10, 'confidence': confidence,
                          'in_frame': True, 'confidence_valid': True}
                         for i,n in enumerate(model_names())]}


def frame(index, dogs):
    return {'frame_index': index, 'timestamp_seconds': index/25,
            'frame_width': 200, 'frame_height': 200, 'dogs': dogs}


def measure(tmp_path, frames, **kwargs):
    source = tmp_path/'pose.jsonl'; output = tmp_path/'quality.json'
    source.write_text(''.join(json.dumps(f)+'\n' for f in frames))
    result = analyze(source,output,**kwargs)
    rows = [json.loads(s) for s in output.with_suffix('.frames.jsonl').read_text().splitlines()]
    return result,rows


def test_zero_single_multiple(tmp_path):
    result, rows = measure(tmp_path,[frame(0,[]),frame(1,[dog()]),frame(2,[dog(),dog()])])
    assert result['total_frames'] == 3
    assert result['rates']['no_dog'] == result['rates']['single_dog'] == result['rates']['multi_dog'] == 1/3
    assert result['mean_detection_confidence'] == pytest.approx(.8)
    assert result['mean_keypoint_confidence'] == pytest.approx(.9)
    assert rows[2]['overlapping_pairs'][0]['iou'] == 1
    assert rows[2]['frame_index'] == 2
    assert rows[2]['temporal'] is None
    assert result['per_keypoint'][0]['reliability_rate'] == 1


def test_missing_low_and_invalid(tmp_path):
    d = dog(); d['keypoints'].pop()
    d['keypoints'][0]['confidence'] = .4
    d['keypoints'][1]['x'] = None
    result, rows = measure(tmp_path,[frame(0,[d])])
    r = rows[0]['dogs'][0]
    assert r['visible_counts'] == {'0.3': 37, '0.5': 36, '0.7': 36}
    assert r['missing_keypoints'] == 1 and r['invalid_keypoints'] == 1
    assert result['missing_keypoint_rate'] == 1/39
    assert result['per_keypoint'][0]['reliability_rate'] == 0


def test_transitions(tmp_path):
    d = dog(); d['keypoints'][0]['confidence'] = .2
    d['keypoints'].pop(1)
    d['keypoints'][1]['x'] = None  # original index 2
    _,rows = measure(tmp_path,[frame(0,[dog()]),frame(1,[d]),frame(2,[dog()])])
    t = rows[1]['temporal']
    assert t['below_threshold'] == [0] and t['missing'] == [1] and t['invalid'] == [2]
    assert t['visible_lost'] == [0,1,2]
    assert rows[2]['temporal']['reappeared'] == [0,1,2]


def test_jump_and_no_identity_bridging(tmp_path):
    result, rows = measure(tmp_path,[frame(0,[dog()]),frame(1,[dog(x=150)]),
                                    frame(2,[dog(),dog()]),frame(3,[dog()]),
                                    frame(4,[]),frame(5,[dog()])])
    assert rows[1]['temporal']['jump_indices'] == list(range(39))
    assert all(rows[i]['temporal'] is None for i in (2,3,4,5))
    assert result['counts']['jump_pairs'] == 39


@pytest.mark.parametrize('value', [None,float('nan'),float('inf'),-1,201])
def test_invalid_coordinates(tmp_path,value):
    d=dog(); d['keypoints'][0]['x']=value
    _, rows=measure(tmp_path,[frame(0,[d])])
    assert rows[0]['dogs'][0]['reliable_keypoints'] == 38


def test_corrupt_input_breaks_chain(tmp_path):
    source=tmp_path/'pose.jsonl'; target=tmp_path/'quality.json'
    source.write_text(json.dumps(frame(0,[dog()]))+'\n{broken\n'+json.dumps(frame(2,[dog()]))+'\n')
    result=analyze(source,target)
    assert result['counts']['invalid_records']==1
    assert result['counts']['temporal_pairs']==0
    assert result['total_frames']==2
    assert 'NaN' not in target.read_text()


def test_duplicate_and_wrong_names(tmp_path):
    d=dog(); d['keypoints'].append(deepcopy(d['keypoints'][0])); d['keypoints'][1]['name']='wrong'
    _,rows=measure(tmp_path,[frame(0,[d])])
    assert rows[0]['dogs'][0]['missing_keypoints']==2


def test_no_overwrite(tmp_path):
    source=tmp_path/'pose.jsonl'; source.write_text('')
    target=tmp_path/'report.json'; target.write_text('keep')
    with pytest.raises(FileExistsError): analyze(source,target)
    assert target.read_text()=='keep'


def test_empty_unknown_averages(tmp_path):
    result,_=measure(tmp_path,[])
    assert result['mean_keypoint_confidence'] is None
    assert result['rates']['no_dog'] is None
    assert result['counts']['records']==0


def test_iou():
    assert iou([0,0,10,10],[20,20,30,30])==0
    assert iou([0,0,0,10],[0,0,10,10]) is None


def test_png_export(tmp_path):
    import cv2
    import numpy as np
    from src.quality.visualize import export_samples
    measure(tmp_path,[frame(0,[dog(confidence=.2)])])
    # Fake decoder uses one synthetic frame; no real video/model is needed.
    class Capture:
        closed=False
        def isOpened(self): return True
        def read(self): return True,np.zeros((200,200,3),dtype=np.uint8)
        def release(self): self.closed=True
    capture=Capture()
    from unittest.mock import patch
    with patch.object(cv2,'VideoCapture',return_value=capture):
        result=export_samples(tmp_path/'pose.jsonl',tmp_path/'quality.frames.jsonl',
                              'unused.mov',tmp_path/'png')
    assert result['png_count']==3
    assert capture.closed
    assert len(list((tmp_path/'png').glob('*.png')))==3


def test_optional_png_failure_preserves_quality(tmp_path,monkeypatch):
    from src.quality.run_quality import main
    import src.quality.visualize as visualization
    source=tmp_path/'pose.jsonl'; source.write_text(json.dumps(frame(0,[]))+'\n')
    def fail(*a): raise OSError('private path')
    monkeypatch.setattr(visualization,'export_samples',fail)
    target=tmp_path/'quality.json'
    assert main(['--input',str(source),'--output',str(target),
                 '--video','absent.mov','--frames-dir',str(tmp_path/'png')])==2
    assert json.loads(target.read_text())['total_frames']==1


def test_context_metadata_and_summary(tmp_path):
    from src.quality.analyze import context_checks
    meta=tmp_path/'meta.json';motion=tmp_path/'motion.json'
    meta.write_text(json.dumps({'run_status':'complete','frames_written':834,
        'visualization_profile':'conservative','thresholds':{'visualization_keypoint_confidence':.5},
        'private':'/Users/private'}))
    motion.write_text(json.dumps({'total_frames':834,'input_sha256':'abc'}))
    result=context_checks('abc',834,meta,motion)
    assert result['pose_metadata']['visualization_profile']=='conservative'
    assert result['pose_metadata']['frame_count_matches']
    assert result['motion_summary']['input_hash_matches']
    assert '/Users' not in json.dumps(result)


def test_nonoverlap_single_detections_not_compared(tmp_path):
    other=dog();other['bbox_xyxy']=[110,110,190,190]
    result,rows=measure(tmp_path,[frame(0,[dog()]),frame(1,[other])])
    assert rows[1]['temporal'] is None
    assert result['counts']['temporal_pairs']==0
