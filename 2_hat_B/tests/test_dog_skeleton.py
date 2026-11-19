"""Project-authored anatomical guide tests; no real inference or video needed."""
from types import SimpleNamespace as NS
import math

import numpy as np
import pytest

from src.pose_estimation.dog_skeleton import (
    DOG_VISUALIZATION_EDGES, DOG_VISUALIZATION_NAMES, dog_visualization_schema,
)
from src.pose_estimation.interfaces import DogPose
from src.pose_estimation.metadata import load_superanimal_metadata
from src.pose_estimation.visualization import draw_poses
from src.preprocessing.interfaces import DogDetection


class Recorder:
    FONT_HERSHEY_SIMPLEX=0
    def __init__(self):self.lines=[];self.points=[]
    def rectangle(self,*args):pass
    def putText(self,*args):pass
    def line(self,image,start,end,*args):self.lines.append((start,end))
    def circle(self,image,point,*args):self.points.append(point)


@pytest.fixture
def fixture():
    _,schema=load_superanimal_metadata()
    pose=DogPose(schema.names,tuple((10.+i*3,20.+i*2) for i in range(39)),(.9,)*39)
    image=np.zeros((150,200,3),dtype=np.uint8)
    dogs=[DogDetection((1,1,180,140),.95)]
    return schema,pose,image,dogs


def test_schema_is_separate_and_official_skeleton_stays_empty(fixture):
    schema,pose,_,_=fixture
    custom=dog_visualization_schema(schema.names)
    assert custom is not schema and custom.names==schema.names==pose.keypoint_names
    assert schema.edges==()
    assert len(custom.edges)==18 and len(DOG_VISUALIZATION_NAMES)==26
    assert load_superanimal_metadata()[0]['metadata']['skeleton']==[]


def test_anatomical_selection_excludes_antlers_and_ambiguous_bridges(fixture):
    schema,_,_,_=fixture
    assert DOG_VISUALIZATION_NAMES.issubset(schema.names)
    assert not any('antler' in name for name in DOG_VISUALIZATION_NAMES)
    assert {'nose','left_eye','right_eye','left_earbase','right_earbase',
            'left_earend','right_earend','tail_base','tail_end'}.issubset(DOG_VISUALIZATION_NAMES)
    for location in ('front_left','front_right','back_left','back_right'):
        assert (location+'_thai',location+'_knee') in DOG_VISUALIZATION_EDGES
        assert (location+'_knee',location+'_paw') in DOG_VISUALIZATION_EDGES
    assert ('back_base','back_middle') in DOG_VISUALIZATION_EDGES
    assert ('back_middle','back_end') in DOG_VISUALIZATION_EDGES
    # No invented attachment of thigh markers to unspecified shoulder/hip positions.
    assert not any(('thai' in a and ('back_' in b or 'neck_' in b)) or
                   ('thai' in b and ('back_' in a or 'neck_' in a))
                   for a,b in DOG_VISUALIZATION_EDGES if not ('knee' in a or 'knee' in b))
    assert len(set(frozenset(edge) for edge in DOG_VISUALIZATION_EDGES))==18
    assert all(a!=b for a,b in DOG_VISUALIZATION_EDGES)


def test_indices_are_resolved_by_names_even_if_order_changes(fixture):
    schema,_,_,_=fixture
    reordered=tuple(reversed(schema.names))
    custom=dog_visualization_schema(reordered)
    assert custom.names==reordered
    assert tuple((custom.names[a],custom.names[b]) for a,b in custom.edges)==DOG_VISUALIZATION_EDGES


def test_unknown_or_duplicate_names_fail(fixture):
    names=fixture[0].names
    with pytest.raises(ValueError):dog_visualization_schema(('nose',))
    with pytest.raises(ValueError):dog_visualization_schema(names+(names[0],))


def test_default_mode_is_unchanged(fixture):
    schema,pose,image,dogs=fixture
    cv=Recorder()
    draw_poses(image,dogs,[pose],schema,cv_module=cv)
    assert cv.lines==[] and len(cv.points)==39


def test_dog_mode_draws_selected_points_and_named_edges_without_mutation(fixture):
    schema,pose,image,dogs=fixture
    original=image.copy();cv=Recorder()
    result=draw_poses(image,dogs,[pose],schema,cv_module=cv,dog_skeleton=True)
    assert len(cv.lines)==18 and len(cv.points)==26
    assert result is not image and np.array_equal(image,original)
    assert pose.xy==tuple((10.+i*3,20.+i*2) for i in range(39))
    selected={tuple(map(round,pose.xy[i])) for i,name in enumerate(schema.names) if name in DOG_VISUALIZATION_NAMES}
    assert set(cv.points)==selected
    assert all(a in selected and b in selected for a,b in cv.lines)


@pytest.mark.parametrize('xy,confidence', [((10.,20.),.49),((math.nan,20.),.9),
    ((10.,math.inf),.9),((-1.,20.),.9),((200.,20.),.9),((10.,150.),.9),
    ((10.,20.),math.nan),((10.,20.),math.inf),((10.,20.),-1.),((10.,20.),1.1)])
def test_invalid_endpoint_prevents_point_and_all_incident_lines(fixture,xy,confidence):
    schema,pose,image,dogs=fixture
    index=schema.names.index('left_eye')
    points=list(pose.xy);points[index]=xy
    scores=list(pose.confidence);scores[index]=confidence
    modified=DogPose(schema.names,tuple(points),tuple(scores))
    cv=Recorder()
    draw_poses(image,dogs,[modified],schema,threshold=.5,cv_module=cv,dog_skeleton=True)
    incident=sum('left_eye' in edge for edge in DOG_VISUALIZATION_EDGES)
    assert len(cv.points)==25
    assert len(cv.lines)==18-incident


def test_exact_confidence_threshold_is_inclusive(fixture):
    schema,pose,image,dogs=fixture
    cv=Recorder();pose=DogPose(pose.keypoint_names,pose.xy,(.5,)*39)
    draw_poses(image,dogs,[pose],schema,threshold=.5,cv_module=cv,dog_skeleton=True)
    assert len(cv.lines)==18


def test_video_cli_passes_visualization_mode_only(tmp_path,monkeypatch):
    from src.pose_estimation import run_video
    source=tmp_path/'in.mp4';source.touch()
    calls=[]
    monkeypatch.setattr(run_video,'select_device',lambda device:'cpu')
    monkeypatch.setattr(run_video,'DeepLabCutRTMPose',lambda *args:NS(schema=NS(names=())))
    monkeypatch.setattr(run_video,'YoloDogDetector',lambda *args:NS())
    monkeypatch.setattr(run_video,'process_video',lambda *args,**kwargs:calls.append((args,kwargs)) or {})
    base=['--input',str(source),'--output',str(tmp_path/'out.mp4'),
          '--pose-weights','local.pt','--detector-weights','local-detection.pt']
    assert run_video.main(base+['--dog-skeleton'])==0
    assert calls[-1][1]=={'dog_skeleton':True}
    assert run_video.main(base)==0
    assert calls[-1][1]=={'dog_skeleton':False}


def test_video_pipeline_forwards_mode_without_changing_inference(tmp_path,monkeypatch,fixture):
    from src.preprocessing import video
    schema,pose,pixels,dogs=fixture
    source=tmp_path/'in.mp4';source.touch()
    reads=iter([(True,pixels),(False,None)])
    calls=[];written=[]
    cv=NS(CAP_PROP_FPS=1,CAP_PROP_FRAME_WIDTH=2,CAP_PROP_FRAME_HEIGHT=3,
          VideoCapture=lambda path:NS(isOpened=lambda:True,get=lambda k:{1:30,2:200,3:150}[k],
                                      read=lambda:next(reads),release=lambda:None),
          VideoWriter_fourcc=lambda *args:0,
          VideoWriter=lambda *args:NS(isOpened=lambda:True,write=written.append,release=lambda:None))
    def detect(frame):calls.append('detect');return dogs
    def estimate(frame,detected):
        assert detected is dogs
        calls.append('pose');return [pose]
    def draw(*args,**kwargs):
        assert args[2]==[pose] and kwargs=={'dog_skeleton':True}
        calls.append('draw');return pixels
    monkeypatch.setattr(video,'draw_poses',draw)
    result=video.process_video(source,tmp_path/'out.mp4',NS(detect=detect),
                               NS(estimate=estimate,schema=schema),cv_module=cv,dog_skeleton=True)
    assert calls==['detect','pose','draw'] and len(written)==1 and result['frames']==1


def test_in_frame_subpixel_point_rasterizes_inside_image(fixture):
    schema,pose,image,dogs=fixture
    points=list(pose.xy);index=schema.names.index('nose');points[index]=(199.9,149.9)
    pose=DogPose(schema.names,tuple(points),pose.confidence)
    cv=Recorder()
    draw_poses(image,dogs,[pose],schema,cv_module=cv,dog_skeleton=True)
    assert (199,149) in cv.points
    assert all(0<=x<200 and 0<=y<150 for edge in cv.lines for x,y in edge)
    assert pose.xy[index]==(199.9,149.9)
