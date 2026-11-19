"""Controlled review tests with synthetic JSONL and fake decoded frames."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from src.analysis.regions import model_names
from src.quality.analyze import analyze, inspect_points
from src.quality.review import panel_points, run_review, select_video


def dog(confidence=.8,x=10):
    return {'bbox_xyxy':[0,0,100,100],'detection_confidence':.7,
        'keypoints':[{'index':i,'name':n,'x':x,'y':20,'confidence':confidence,
                      'in_frame':True,'confidence_valid':True} for i,n in enumerate(model_names())]}


def frame(i,dogs):
    return {'frame_index':i,'timestamp_seconds':i/25,'frame_width':200,'frame_height':200,'dogs':dogs}


@pytest.fixture
def evidence(tmp_path):
    frames=[frame(0,[]),frame(1,[dog(.4)]),frame(2,[dog(.2)]),
            frame(3,[dog(.8)]),frame(4,[dog(.8,x=150)]),frame(5,[])]
    pose=tmp_path/'pose.jsonl';quality=tmp_path/'quality.json'
    pose.write_text(''.join(json.dumps(f)+'\n' for f in frames))
    analyze(pose,quality)
    return pose,quality,quality.with_suffix('.frames.jsonl')


def manifest(tmp_path,evidence,count=1):
    pose,quality,frames=evidence
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps({'videos':[{'pose':pose.name,'quality':quality.name,'frames':frames.name}
                                          for _ in range(count)]}))
    return path


def test_four_masks_and_threshold_boundaries():
    names=model_names();points=inspect_points(dog(.3),names,200,200)
    assert len(panel_points(points,names,'raw39',.3))==39
    assert len(panel_points(points,names,'raw39',.5))==0
    assert len(panel_points(points,names,'conservative',.3))==9
    assert len(panel_points(points,names,'conservative',.5))==0
    points=inspect_points(dog(.5),names,200,200)
    assert len(panel_points(points,names,'raw39',.5))==39
    assert len(panel_points(points,names,'conservative',.5))==9


def test_deterministic_selection(evidence):
    first=select_video(*evidence,per_category=1)
    second=select_video(*evidence,per_category=1)
    assert first==second
    chosen={r['frame_index']:r['categories'] for r in first['selected_frames']}
    assert chosen=={0:['no_dog'],2:['low_confidence'],3:['visibility_change'],4:['coordinate_jump']}
    assert first['detection_coverage']==4/6
    assert first['confidence_distributions']['keypoint']['sample_count']==4*39


def test_manual_table_blank_and_three_video_report(tmp_path,evidence):
    path=manifest(tmp_path,evidence,count=3)
    result=run_review(path,tmp_path/'out',per_category=1)
    assert len(result['videos'])==3
    with (tmp_path/'out/manual_review.csv').open(newline='') as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==12
    assert {r['video_id'] for r in rows}=={'video1','video2','video3'}
    assert all(r[k]=='' for r in rows for k in ('manual_dog_visible','manual_anatomy_correct','notes'))
    assert '/workspace' not in json.dumps(result)
    assert result['videos'][0]['selected_frames'][0]['detection_count']==0


def test_inputs_and_existing_outputs_preserved(tmp_path,evidence):
    path=manifest(tmp_path,evidence)
    original={p:p.read_bytes() for p in (*evidence,path)}
    run_review(path,tmp_path/'out')
    outputs={p:p.read_bytes() for p in (tmp_path/'out').rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):run_review(path,tmp_path/'out')
    assert all(p.read_bytes()==v for p,v in {**original,**outputs}.items())


def test_missing_data_has_no_fake_scores(tmp_path):
    pose=tmp_path/'pose.jsonl';quality=tmp_path/'quality.json'
    pose.write_text(json.dumps(frame(0,[{'keypoints':[]}]))+'\n')
    analyze(pose,quality)
    result=select_video(pose,quality,quality.with_suffix('.frames.jsonl'))
    assert result['confidence_distributions']['keypoint']['sample_count']==0
    assert result['mean_visible_keypoints_per_dog']['raw39_0.3']==0
    assert not result['selected_frames']  # Unknown mean is not "low confidence".


def test_empty_and_corrupt_records(tmp_path):
    pose=tmp_path/'pose.jsonl';quality=tmp_path/'quality.json'
    pose.write_text('{broken\n')
    analyze(pose,quality)
    result=select_video(pose,quality,quality.with_suffix('.frames.jsonl'))
    assert result['detection_coverage'] is None and result['invalid_records']==1


def test_stale_report_rejected(evidence):
    pose,_,_=evidence
    pose.write_text(pose.read_text()+'\n')
    with pytest.raises(ValueError,match='does not match'):select_video(*evidence)


def test_stale_detail_rejected(evidence):
    _,_,detail=evidence
    rows=[json.loads(l) for l in detail.read_text().splitlines()]
    rows[0]['dog_count']=2
    detail.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='frame mismatch'):select_video(*evidence)


def test_optional_render_failure_still_writes_table(tmp_path,evidence,monkeypatch):
    import src.quality.review_visualize as renderer
    path=manifest(tmp_path,evidence)
    data=json.loads(path.read_text());data['videos'][0]['video']='original.mov'
    path.write_text(json.dumps(data));(tmp_path/'original.mov').write_bytes(b'not a real video')
    def fail(*args):raise ValueError('private path')
    monkeypatch.setattr(renderer,'render_comparisons',fail)
    result=run_review(path,tmp_path/'out',render=True)
    assert result['videos'][0]['visualization']['status']=='failed_or_unavailable'
    assert (tmp_path/'out/manual_review.csv').exists()
    assert 'private path' not in json.dumps(result)


def test_render_same_frame_four_views(tmp_path,evidence):
    import cv2
    import numpy as np
    from src.quality.review_visualize import render_comparisons
    selected=select_video(*evidence,per_category=1)['selected_frames']
    selected=[row for row in selected if row['frame_index']==2]
    class Capture:
        closed=False
        def isOpened(self):return True
        def read(self):return True,np.zeros((200,200,3),dtype=np.uint8)
        def release(self):self.closed=True
    capture=Capture();labels=[]
    original=cv2.putText
    def put(image,text,*args,**kwargs):
        labels.append(text)
        return original(image,text,*args,**kwargs)
    with patch.object(cv2,'VideoCapture',return_value=capture),patch.object(cv2,'putText',side_effect=put):
        result=render_comparisons(evidence[0],'unused.mov',selected,tmp_path/'png')
    assert result['png_count']==5 and capture.closed
    assert len(list((tmp_path/'png').glob('*.png')))==5
    assert not result['anatomical_edges_drawn']
    assert cv2.imread(str(tmp_path/'png/frame-00000002-comparison.png')).shape[:2]==(540,400)
    assert any('raw39' in label for label in labels)
    with (tmp_path/'png/frame-00000002-points.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==39
    assert rows[0]['keypoint_name']=='nose' and float(rows[0]['confidence'])==.2


def test_point_names_and_confidence_labels(tmp_path,evidence):
    import cv2
    import numpy as np
    from src.quality.review_visualize import render_comparisons
    row={'frame_index':1,'line_number':2}
    class Capture:
        def isOpened(self):return True
        def read(self):return True,np.zeros((200,200,3),dtype=np.uint8)
        def release(self):pass
    labels=[]
    with patch.object(cv2,'VideoCapture',return_value=Capture()),patch.object(cv2,'putText',
            side_effect=lambda image,text,*a:labels.append(text) or image):
        render_comparisons(evidence[0],'unused.mov',[row],tmp_path/'png')
    assert any('nose 0.400' in s for s in labels)
    assert any('tail_end 0.400' in s for s in labels)


@pytest.mark.parametrize('option,value',[('per_category',0),('low_threshold',float('nan')),('visibility_min',0)])
def test_invalid_selection_settings(evidence,option,value):
    with pytest.raises(ValueError):select_video(*evidence,**{option:value})
