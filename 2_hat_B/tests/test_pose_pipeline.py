from pathlib import Path
from types import SimpleNamespace as NS
import math
import pytest

from src.pose_estimation.geometry import xyxy_to_xywh, padded_crop
from src.pose_estimation.metadata import PoseSchema, MetadataError
from src.pose_estimation.interfaces import DogPose
from src.pose_estimation.visualization import visible_keypoints, draw_poses
from src.pose_estimation.adapter import DogPoseAdapter
from src.pose_estimation.device import select_device, inference_error
from src.pose_estimation.backend import DeepLabCutRTMPose
from src.preprocessing.interfaces import DogDetection, VideoFrame
from src.preprocessing.detection import YoloDogDetector
from src.preprocessing.video import validate_video_paths, process_video

# Synthetic fixtures, deliberately NOT a claimed canine skeleton.
SCHEMA = PoseSchema.from_metadata({'bodyparts': ['test_a', 'test_b'], 'skeleton': [['test_a', 'test_b']]})


class Image:
    def __init__(self, height=100, width=200):
        self.shape = (height, width, 3)
    def copy(self):
        return Image(*self.shape[:2])
    def __getitem__(self, key):
        y, x = key
        return Image(y.stop-y.start, x.stop-x.start)


def test_box_conversion():
    assert xyxy_to_xywh((10, 20, 40, 60)) == (10, 20, 30, 40)


@pytest.mark.parametrize('box', [(0,0,0,4), (5,0,2,4), (0,0,math.nan,5), (0,0,math.inf,5)])
def test_invalid_boxes(box):
    with pytest.raises(ValueError): xyxy_to_xywh(box)


def test_crop_margin_and_bounds():
    c = padded_crop((10, 20, 50, 60), 100, 80, .25)
    assert (c.x1,c.y1,c.x2,c.y2) == (0,10,60,70)
    assert c.extract(Image(80,100)).shape == (60,60,3)
    assert c.to_original([(0,0),(30,30)], (30,30)) == ((0,10),(60,70))
    assert c.to_original([(2,3)]) == ((2,13),)


@pytest.mark.parametrize('box,margin', [((200,0,220,20),.1), ((0,0,20,20),-.1), ((0,0,20,20),.6)])
def test_invalid_crop(box,margin):
    with pytest.raises(ValueError): padded_crop(box,100,100,margin)


def test_low_confidence_nan_and_outside_filtered():
    pose = DogPose(('a','b','c','d','e'), ((1,2),(3,4),(math.nan,2),(-1,0),(8,8)), (.5,.49,.9,.9,math.nan))
    assert visible_keypoints(pose,.5,(10,10)) == (True,False,False,False,False)


@pytest.mark.parametrize('metadata', [None, {}, {'bodyparts':['a'],'skeleton':[]},
    {'bodyparts':['a','a'],'skeleton':[['a','a']]}, {'bodyparts':['a'],'skeleton':[['a','b']]}])
def test_unverified_schema_rejected(metadata):
    with pytest.raises(MetadataError): PoseSchema.from_metadata(metadata)


def test_multiple_dogs_transform_independently():
    backend = NS(schema=SCHEMA, predict_crop=lambda crop: ([(1,2),(3,4)],[.9,.8]))
    estimator = DogPoseAdapter(backend,margin=0)
    frame = VideoFrame(0,0,Image())
    dogs = [DogDetection((10,20,40,60),.9),DogDetection((70,40,100,70),.8)]
    poses = estimator.estimate(frame,dogs)
    assert poses[0].xy == ((11,22),(13,24))
    assert poses[1].xy == ((71,42),(73,44))
    assert estimator.estimate(frame,[]) == []


class Tensor:
    def __init__(self, values): self.values=values
    def cpu(self): return self
    def tolist(self): return self.values


def test_detection_metadata_class_and_scores():
    calls=[]
    boxes=NS(xyxy=Tensor([[1,2,20,30],[40,20,80,60],[0,0,1,1]]),
             conf=Tensor([.9,.8,.99]),cls=Tensor([7,7,1]))
    model=NS(task='detect',names={1:'cat',7:'dog'},predict=lambda **kw: calls.append(kw) or [NS(boxes=boxes)])
    detector=YoloDogDetector(model=model)
    dogs=detector.detect(VideoFrame(0,0,Image()))
    assert len(dogs)==2 and dogs[1].confidence==.8
    assert calls[0]['classes']==[7] and calls[0]['batch']==1


@pytest.mark.parametrize('names', [{0:'person'}, {1:'dog',2:'dog'}])
def test_missing_or_ambiguous_dog_class(names):
    with pytest.raises(ValueError): YoloDogDetector(model=NS(task='detect',names=names))


def test_local_checkpoint_required(tmp_path):
    with pytest.raises(FileNotFoundError): YoloDogDetector(tmp_path/'absent.pt')
    with pytest.raises(FileNotFoundError): DeepLabCutRTMPose(tmp_path/'absent.pt',tmp_path/'absent.yaml')


def test_device_fallback():
    torch=NS(backends=NS(mps=NS(is_available=lambda:False)))
    assert select_device('auto',torch)=='cpu'
    assert select_device('cpu',torch)=='cpu'
    with pytest.raises(RuntimeError,match='cpu'): select_device('mps',torch)
    torch.backends.mps.is_available=lambda:True
    assert select_device('auto',torch)=='mps'
    assert 'cpu' in str(inference_error(RuntimeError('unsupported'),'mps'))


@pytest.mark.parametrize('suffix', ['.MOV','.mp4'])
def test_video_paths(tmp_path,suffix):
    p=tmp_path/('input'+suffix);p.touch()
    assert validate_video_paths(p,tmp_path/'output.mp4')[0]==p


def test_invalid_video_paths(tmp_path):
    p=tmp_path/'input.mp4';p.touch()
    with pytest.raises(ValueError): validate_video_paths(p,p)
    with pytest.raises(ValueError): validate_video_paths(p,tmp_path/'out.mov')
    with pytest.raises(ValueError): validate_video_paths(tmp_path/'file.txt',tmp_path/'out.mp4')
    with pytest.raises(FileNotFoundError): validate_video_paths(tmp_path/'absent.mov',tmp_path/'out.mp4')
    out=tmp_path/'out.mp4';out.touch()
    with pytest.raises(FileExistsError): validate_video_paths(p,out)


class CV:
    CAP_PROP_FPS=1; CAP_PROP_FRAME_WIDTH=2; CAP_PROP_FRAME_HEIGHT=3
    FONT_HERSHEY_SIMPLEX=0
    def __init__(self): self.drawn=[];self.written=0;self.released=[];self.reads=0
    def VideoCapture(self,path):
        return NS(isOpened=lambda:True,get=lambda k:{1:25,2:200,3:100}[k],read=self.read,
                  release=lambda:self.released.append('capture'))
    def read(self):
        self.reads+=1
        return (True,Image()) if self.reads<=3 else (False,None)
    def VideoWriter_fourcc(self,*args):return 1
    def VideoWriter(self,path,codec,fps,size):
        self.writer_args=(fps,size)
        return NS(isOpened=lambda:True,write=self.write,release=lambda:self.released.append('writer'))
    def write(self,image):self.written+=1
    def rectangle(self,*args):pass
    def putText(self,*args):pass
    def line(self,*args):self.drawn.append('line')
    def circle(self,*args):self.drawn.append('circle')


def test_skeleton_filters_both_endpoints():
    cv=CV()
    pose=DogPose(SCHEMA.names,((1,2),(3,4)),(.9,.1))
    draw_poses(Image(),[DogDetection((0,0,10,10),.9)],[pose],SCHEMA,cv_module=cv)
    assert cv.drawn==['circle']


def test_streaming_video_and_release(tmp_path):
    source=tmp_path/'in.mov';source.touch()
    cv=CV()
    def detect(frame):
        # A frame is processed before the next is decoded.
        assert cv.reads==frame.index+1
        return []
    estimator=NS(schema=SCHEMA,estimate=lambda frame,dogs:[])
    report=process_video(source,tmp_path/'out.mp4',NS(detect=detect),estimator,cv_module=cv)
    assert report==dict(frames=3,fps=25,width=200,height=100,audio_preserved=False)
    assert cv.written==3 and cv.writer_args==(25,(200,100))
    assert cv.released==['capture','writer']


def test_video_resources_released_on_inference_failure(tmp_path):
    p=tmp_path/'in.mp4';p.touch();cv=CV()
    def fail(frame):raise RuntimeError('unsupported operator')
    with pytest.raises(RuntimeError,match='--device cpu'):
        process_video(p,tmp_path/'out.mp4',NS(detect=fail),NS(schema=SCHEMA),device='mps',cv_module=cv)
    assert cv.released==['capture','writer']


def test_cli_help_has_no_ml_dependency():
    from src.pose_estimation.run_video import main
    with pytest.raises(SystemExit) as exc: main(['--help'])
    assert exc.value.code==0


def test_checkpoint_inspection_fails_closed_without_metadata(tmp_path, monkeypatch):
    import sys
    checkpoint=tmp_path/'fixture.pt';checkpoint.touch()
    config=tmp_path/'fixture.yaml';config.write_text('{}')
    calls=[]
    def load(path, **kwargs):
        calls.append(kwargs)
        return {'state_dict': {}}  # Synthetic structure; not a real DLC checkpoint.
    monkeypatch.setitem(sys.modules,'torch',NS(load=load))
    with pytest.raises(MetadataError,match='no verified metadata'):
        DeepLabCutRTMPose(checkpoint,config)
    assert calls==[{'map_location':'cpu','weights_only':True}]


def test_plausible_metadata_does_not_bypass_provenance_gate(tmp_path, monkeypatch):
    import sys
    import yaml
    names=[f'fixture_{i}' for i in range(39)]
    metadata={'bodyparts':names,'skeleton':[[names[0],names[1]]]}
    checkpoint=tmp_path/'fixture.pt';checkpoint.touch()
    config=tmp_path/'fixture.yaml';config.write_text(yaml.safe_dump({'metadata':metadata}))
    monkeypatch.setitem(sys.modules,'torch',NS(load=lambda *args,**kwargs:{'metadata':metadata}))
    with pytest.raises(MetadataError,match='provenance'):
        DeepLabCutRTMPose(checkpoint,config)


def test_skeleton_draws_verified_edge():
    cv=CV()
    pose=DogPose(SCHEMA.names,((1,2),(3,4)),(.9,.8))
    draw_poses(Image(),[DogDetection((0,0,10,10),.9)],[pose],SCHEMA,cv_module=cv)
    assert cv.drawn==['line','circle','circle']


def test_empty_video_releases_capture(tmp_path):
    p=tmp_path/'in.mp4';p.touch();cv=CV();cv.read=lambda:(False,None)
    with pytest.raises(ValueError,match='no decodable frames'):
        process_video(p,tmp_path/'out.mp4',NS(),NS(),cv_module=cv)
    assert cv.released==['capture'] and cv.written==0


def test_invalid_fps_releases_capture(tmp_path):
    p=tmp_path/'in.mp4';p.touch();cv=CV()
    original=cv.VideoCapture
    def capture(path):
        result=original(path)
        result.get=lambda key:0
        return result
    cv.VideoCapture=capture
    with pytest.raises(ValueError,match='invalid FPS'):
        process_video(p,tmp_path/'out.mp4',NS(),NS(),cv_module=cv)
    assert cv.released==['capture']


def test_all_pipeline_imports_are_lightweight():
    import subprocess
    import sys
    script='''
import sys
import src.pose_estimation.run_video
assert not ({'torch','ultralytics','cv2','numpy','deeplabcut'} & sys.modules.keys())
'''
    subprocess.run([sys.executable,'-B','-c',script],check=True)
