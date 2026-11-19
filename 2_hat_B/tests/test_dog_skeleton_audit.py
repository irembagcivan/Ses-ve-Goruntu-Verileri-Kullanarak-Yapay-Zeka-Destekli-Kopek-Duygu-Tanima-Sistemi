"""QC proves graph integrity and rendering filters, not anatomical accuracy."""
from types import SimpleNamespace as NS
import math

import numpy as np
import pytest

from src.pose_estimation.dog_skeleton import (
    DOG_VISUALIZATION_EDGES, CONSERVATIVE_EDGES, CONSERVATIVE_NAMES,
    dog_visualization_schema, validate_visualization_edges,
)
from src.pose_estimation.metadata import load_superanimal_metadata
from src.pose_estimation.interfaces import DogPose
from src.preprocessing.interfaces import DogDetection
from src.pose_estimation.visualization import draw_poses


class Recorder:
    FONT_HERSHEY_SIMPLEX=0
    def __init__(self):self.lines=[];self.points=[]
    def rectangle(self,*args):pass
    def putText(self,*args):pass
    def line(self,image,a,b,*args):self.lines.append((a,b))
    def circle(self,image,xy,*args):self.points.append(xy)


@pytest.mark.parametrize('edges', [
    (('nose','not_a_model_name'),), (('nose','nose'),),
    (('nose','left_eye'),('nose','left_eye')),
    (('nose','left_eye'),('left_eye','nose')),
    (('right_antler_base','right_antler_end'),), (('nose',),),
])
def test_graph_validator_rejects_invalid_connections(edges):
    _,schema=load_superanimal_metadata()
    with pytest.raises(ValueError):validate_visualization_edges(schema.names,edges)


@pytest.mark.parametrize('profile,edge_count',[('legacy',18),('conservative',7)])
def test_profiles_use_valid_unique_labels_without_mutation(profile,edge_count):
    evidence,schema=load_superanimal_metadata()
    resolved=dog_visualization_schema(schema.names,profile)
    assert resolved.names==schema.names and len(resolved.edges)==edge_count
    assert schema.edges==() and evidence['metadata']['skeleton']==[]
    edges=[(schema.names[a],schema.names[b]) for a,b in resolved.edges]
    assert len({frozenset(edge) for edge in edges})==edge_count
    assert all(a!=b and 'antler' not in a and 'antler' not in b for a,b in edges)


def test_conservative_profile_removes_uncertain_regions_without_new_edges():
    assert len(CONSERVATIVE_NAMES)==9
    assert set(CONSERVATIVE_EDGES).issubset(DOG_VISUALIZATION_EDGES)
    assert not any(any(region in name for region in ('thai','knee','paw','neck','back'))
                   for edge in CONSERVATIVE_EDGES for name in edge)


def test_conservative_indices_follow_names_not_numeric_order():
    _,schema=load_superanimal_metadata()
    names=tuple(reversed(schema.names))
    resolved=dog_visualization_schema(names,'conservative')
    assert tuple((names[a],names[b]) for a,b in resolved.edges)==CONSERVATIVE_EDGES


@pytest.mark.parametrize('mode,expected_lines',[(True,18),('conservative',7)])
@pytest.mark.parametrize('xy,confidence', [((20.,20.),.49),((math.nan,20.),.9),
    ((math.inf,20.),.9),((-1.,20.),.9),((200.,20.),.9),((20.,150.),.9),
    ((20.,20.),math.nan),((20.,20.),math.inf)])
def test_both_profiles_suppress_invalid_point_and_incident_edges(mode,expected_lines,xy,confidence):
    _,schema=load_superanimal_metadata()
    points=[(10.+i*3,20.+i*2) for i in range(39)]
    scores=[.9]*39;index=schema.names.index('left_eye')
    points[index]=xy;scores[index]=confidence
    pose=DogPose(schema.names,tuple(points),tuple(scores));cv=Recorder()
    draw_poses(np.zeros((150,200,3),dtype=np.uint8),[DogDetection((0,0,180,140),.9)],
               [pose],schema,cv_module=cv,dog_skeleton=mode)
    assert len(cv.lines)==expected_lines-2
    assert len(cv.points)==(25 if mode is True else 8)
    assert all(0<=x<200 and 0<=y<150 for edge in cv.lines for x,y in edge)
    np.testing.assert_equal(pose.xy[index],xy)
    np.testing.assert_equal(pose.confidence[index],confidence)


def test_conservative_does_not_draw_high_confidence_uncertain_body_markers():
    _,schema=load_superanimal_metadata()
    points=tuple((10.+i*3,20.+i*2) for i in range(39))
    pose=DogPose(schema.names,points,(.99,)*39);cv=Recorder()
    draw_poses(np.zeros((150,200,3),dtype=np.uint8),[DogDetection((0,0,180,140),.9)],
               [pose],schema,cv_module=cv,dog_skeleton='conservative')
    expected={tuple(map(round,points[i])) for i,name in enumerate(schema.names) if name in CONSERVATIVE_NAMES}
    assert set(cv.points)==expected
    assert pose.xy==points and len(pose.xy)==39


def test_profile_requires_explicit_dog_mode():
    from src.pose_estimation.run_video import main
    with pytest.raises(SystemExit) as exc:
        main(['--input','unused.mov','--output','unused.mp4','--pose-weights','unused.pt',
              '--detector-weights','unused.pt','--dog-skeleton-profile','conservative'])
    assert exc.value.code==2


def test_cli_forwards_conservative_without_new_inference_arguments(tmp_path,monkeypatch):
    from src.pose_estimation import run_video
    source=tmp_path/'in.mov';source.touch();calls=[]
    monkeypatch.setattr(run_video,'select_device',lambda requested:'cpu')
    monkeypatch.setattr(run_video,'DeepLabCutRTMPose',lambda *args:NS(schema=NS(names=())))
    monkeypatch.setattr(run_video,'YoloDogDetector',lambda *args:NS())
    monkeypatch.setattr(run_video,'process_video',lambda *args,**kw:calls.append(kw) or {})
    run_video.main(['--input',str(source),'--output',str(tmp_path/'out.mp4'),
                    '--pose-weights','local.pt','--detector-weights','local.pt',
                    '--dog-skeleton','--dog-skeleton-profile','conservative'])
    assert calls==[{'dog_skeleton':'conservative'}]
