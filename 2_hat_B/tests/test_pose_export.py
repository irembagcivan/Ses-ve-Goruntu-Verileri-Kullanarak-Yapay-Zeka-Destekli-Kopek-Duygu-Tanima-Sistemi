"""Streaming serialization tests: synthetic frames/poses, no model inference."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.pose_estimation.export import (
    PoseJsonlWriter, frame_record, metadata_path, run_metadata, validate_export_paths,
)
from src.pose_estimation.interfaces import DogPose
from src.pose_estimation.metadata import load_superanimal_metadata
from src.preprocessing.interfaces import DogDetection
from src.preprocessing import video


@pytest.fixture
def schema():return load_superanimal_metadata()[1]


def pose(schema):
    return DogPose(schema.names,tuple((i+.123456789,i+.234567891) for i in range(39)),(.01,)*39)


def strict_loads(line):
    def reject(value):raise AssertionError('Non-standard JSON constant: '+value)
    return json.loads(line,parse_constant=reject)


def test_single_dog_precision_schema_and_no_tracking(schema):
    value=pose(schema)
    record=frame_record(2,60,1920,1080,[DogDetection((.1,.2,100.3,200.4),.9)],[value],schema.names)
    dog=record['dogs'][0]
    assert record['timestamp_seconds']==2/60 and record['timestamp_source']=='frame_index_div_fps'
    assert dog['bbox_xyxy']==[.1,.2,100.3,200.4]
    assert dog['detection_index']==0 and 'dog_id' not in dog
    assert len(dog['keypoints'])==39
    assert [p['index'] for p in dog['keypoints']]==list(range(39))
    assert [p['name'] for p in dog['keypoints']]==list(schema.names)
    assert dog['keypoints'][0]['x']==.123456789
    assert dog['keypoints'][0]['confidence']==.01  # Drawing threshold is not applied.
    assert any('antler' in p['name'] for p in dog['keypoints'])
    assert value.xy[0]==(.123456789,.234567891)


def test_many_and_no_dogs(schema):
    detection=DogDetection((0,0,10,10),.9)
    result=frame_record(0,30,100,100,[detection,detection],[pose(schema),pose(schema)],schema.names)
    assert [dog['detection_index'] for dog in result['dogs']]==[0,1]
    assert frame_record(1,30,100,100,[],[],schema.names)['dogs']==[]


def test_invalid_values_are_null_but_finite_raw_values_are_preserved(schema):
    points=list(pose(schema).xy);points[0]=(math.nan,math.inf);points[1]=(-2.,300.)
    confidence=[.9]*39;confidence[0]=math.nan
    value=DogPose(schema.names,tuple(points),tuple(confidence))
    record=frame_record(0,30,100,100,[DogDetection((0,0,math.inf,math.nan),math.nan)],[value],schema.names)
    decoded=strict_loads(json.dumps(record,allow_nan=False))['dogs'][0]
    assert decoded['bbox_xyxy']==[0.,0.,None,None]
    assert decoded['detection_confidence'] is None
    assert decoded['keypoints'][0]['x'] is decoded['keypoints'][0]['y'] is None
    assert decoded['keypoints'][0]['confidence'] is None
    assert decoded['keypoints'][1]['x']==-2. and decoded['keypoints'][1]['y']==300.
    assert decoded['keypoints'][1]['in_frame'] is False


def test_schema_mismatch_rejected(schema):
    p=pose(schema)
    with pytest.raises(ValueError):frame_record(0,30,100,100,[],[p],schema.names)
    bad=DogPose(tuple(reversed(schema.names)),p.xy,p.confidence)
    with pytest.raises(ValueError):frame_record(0,30,100,100,[DogDetection((0,0,1,1),1)],[bad],schema.names)


def test_writer_order_strict_json_and_failed_prefix(tmp_path,schema):
    path=tmp_path/'pose.jsonl'
    writer=PoseJsonlWriter(path,schema.names,30,100,100,run_metadata())
    try:
        with pytest.raises(ValueError):writer.write(1,[],[])
        writer.write(0,[],[])
        assert strict_loads(path.read_text())['frame_index']==0
        with pytest.raises(ValueError):writer.write(0,[],[])
        writer.write(1,[],[])
    finally:writer.close(complete=False)
    assert writer.stream.closed and writer.metadata_stream.closed
    assert [strict_loads(line)['frame_index'] for line in path.read_text().splitlines()]==[0,1]
    metadata=json.loads(metadata_path(path).read_text())
    assert metadata['run_status']=='failed' and metadata['frames_written']==2
    with pytest.raises(ValueError):writer.write(2,[],[])


def test_existing_files_and_collision_are_protected(tmp_path,schema):
    path=tmp_path/'pose.jsonl'
    metadata_path(path).write_text('keep')
    with pytest.raises(FileExistsError):PoseJsonlWriter(path,schema.names,30,100,100,{})
    assert metadata_path(path).read_text()=='keep' and not path.exists()
    with pytest.raises(ValueError):validate_export_paths(path,path)
    with pytest.raises(ValueError):validate_export_paths(tmp_path/'out.json')


def test_model_metadata_contains_hashes_not_paths(tmp_path,schema):
    private=tmp_path/'private-username.pt';private.write_bytes(b'synthetic hash fixture')
    data=run_metadata(private,private,'mps',.25,.5,.15,'conservative')
    assert data['models']['pose']['sha256']==hashlib.sha256(private.read_bytes()).hexdigest()
    assert data['device']=='mps'
    assert data['keypoint_schema']['names']==list(schema.names)
    assert data['source_pts_verified'] is False and data['tracking_enabled'] is False
    assert 'private-username' not in json.dumps(data) and str(tmp_path) not in json.dumps(data)


class FakeVideo:
    CAP_PROP_FPS=1;CAP_PROP_FRAME_WIDTH=2;CAP_PROP_FRAME_HEIGHT=3
    def __init__(self,pose_path):
        self.index=0;self.closed=[];self.written=[];self.pose_path=pose_path
    def VideoCapture(self,path):
        return NS(isOpened=lambda:True,get=lambda key:{1:60.,2:100,3:100}[key],
                  read=self.read,release=lambda:self.closed.append('capture'))
    def read(self):
        # Each prior frame must already be on disk before reading the next.
        if self.index>0:
            assert len(self.pose_path.read_text().splitlines())==min(self.index,3)
        if self.index==3:return False,None
        self.index+=1
        return True,np.zeros((100,100,3),dtype=np.uint8)
    def VideoWriter_fourcc(self,*args):return 0
    def VideoWriter(self,*args):
        return NS(isOpened=lambda:True,write=self.written.append,release=lambda:self.closed.append('video_writer'))


def test_video_streams_each_frame_and_closes_resources(tmp_path,monkeypatch,schema):
    source=tmp_path/'input.mov';source.touch();target=tmp_path/'video.mp4';path=tmp_path/'pose.jsonl'
    cv=FakeVideo(path);detection=DogDetection((0,0,80,80),.9)
    detector=NS(detect=lambda frame:([detection]*(frame.index+1) if frame.index<2 else []))
    estimator=NS(schema=schema,estimate=lambda frame,dogs:[pose(schema) for dog in dogs])
    monkeypatch.setattr(video,'draw_poses',lambda pixels,*args,**kwargs:pixels)
    result=video.process_video(source,target,detector,estimator,cv_module=cv,
                               dog_skeleton='conservative',pose_output=path)
    records=[strict_loads(line) for line in path.read_text().splitlines()]
    assert [r['frame_index'] for r in records]==[0,1,2]
    assert [len(r['dogs']) for r in records]==[1,2,0]
    assert len(records[0]['dogs'][0]['keypoints'])==39
    assert records[1]['timestamp_seconds']==1/60
    assert result['frames']==3 and len(cv.written)==3
    assert cv.closed==['capture','video_writer']
    assert json.loads(metadata_path(path).read_text())['run_status']=='complete'


def test_video_failure_retains_labelled_prefix_and_closes_resources(tmp_path,monkeypatch,schema):
    source=tmp_path/'input.mp4';source.touch();target=tmp_path/'video.mp4';path=tmp_path/'pose.jsonl'
    cv=FakeVideo(path)
    def detect(frame):
        if frame.index==1:raise RuntimeError('synthetic inference error')
        return []
    monkeypatch.setattr(video,'draw_poses',lambda pixels,*args,**kwargs:pixels)
    with pytest.raises(RuntimeError):
        video.process_video(source,target,NS(detect=detect),NS(schema=schema,estimate=lambda frame,dogs:[]),
                            cv_module=cv,pose_output=path)
    assert cv.closed==['capture','video_writer']
    assert len(path.read_text().splitlines())==1
    metadata=json.loads(metadata_path(path).read_text())
    assert metadata['run_status']=='failed' and metadata['frames_written']==1
    assert not target.exists()


def test_cli_optional_export_metadata_and_legacy_kwargs(tmp_path,monkeypatch,schema):
    from src.pose_estimation import run_video
    source=tmp_path/'source.mp4';source.touch();checkpoint=tmp_path/'fixture.pt';checkpoint.write_bytes(b'fixture')
    calls=[]
    monkeypatch.setattr(run_video,'select_device',lambda arg:'cpu')
    monkeypatch.setattr(run_video,'DeepLabCutRTMPose',lambda *args:NS(schema=schema))
    monkeypatch.setattr(run_video,'YoloDogDetector',lambda *args:NS())
    monkeypatch.setattr(run_video,'process_video',lambda *args,**kwargs:calls.append(kwargs) or {})
    args=['--input',str(source),'--output',str(tmp_path/'video.mp4'),
          '--detector-weights',str(checkpoint),'--pose-weights',str(checkpoint)]
    run_video.main(args)
    assert calls[-1]=={'dog_skeleton':False}
    run_video.main(args+['--pose-output',str(tmp_path/'pose.jsonl')])
    assert calls[-1]['pose_metadata']['models']['pose']['status']=='hashed'
    assert calls[-1]['pose_metadata']['thresholds']['detection_confidence']==.25
