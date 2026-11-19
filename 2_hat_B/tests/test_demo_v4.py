import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")
from src.demo.stable_pose import FlowPreviewTracker
from src.demo.gallery import inventory
from src.demo.robust_detection import AngledDetection, OrientedPoseAdapter
from src.demo.tracking import Track
from src.preprocessing.interfaces import DogDetection, VideoFrame
from src.pose_estimation.interfaces import DogPose


def test_optical_flow_reduces_model_jitter_without_freezing_moving_point():
    tracker = FlowPreviewTracker()
    rng = np.random.default_rng(3)
    gray = rng.integers(0,255,(96,96),dtype=np.uint8)
    first = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    moved = cv2.warpAffine(first,np.float32([[1,0,2],[0,1,0]]),(96,96))
    tracker.set_frame(first)
    tracker.update([DogDetection((10,10,80,80),.9)],[DogPose(("nose",),((40,40),),(.9,))],0)
    tracker.set_frame(moved)
    point = tracker.update([DogDetection((12,10,82,80),.9)],[DogPose(("nose",),((45,40),),(.9,))],.1)[0].xy[0]
    assert abs(point[0]-42) < 1.5
    assert point[0] > 41


def test_implausibly_long_face_edges_are_never_drawn():
    tracker = FlowPreviewTracker()
    track = Track(1,np.array([0,0,100,100]),np.array([[10.,10.],[80.,80.]]),np.array([.9,.9]),0)
    assert not tracker.edge_allowed(track,0,1,("nose","left_eye"))


def test_edge_length_spike_is_rejected_after_stable_observations():
    tracker = FlowPreviewTracker()
    track = Track(1,np.array([0,0,100,100]),np.array([[30.,30.],[40.,40.]]),np.array([.9,.9]),0)
    assert not tracker.edge_allowed(track,0,1,("back_base","back_middle"))
    assert tracker.edge_allowed(track,0,1,("back_base","back_middle"))
    tracker.edge_allowed(track,0,1,("back_base","back_middle"))
    track.xy[1] = [60,60]
    assert not tracker.edge_allowed(track,0,1,("back_base","back_middle"))


def test_rotated_pose_returns_to_original_coordinates():
    class Adapter:
        schema = None
        def estimate(self, frame, dogs):
            assert frame.pixels.shape[:2] == (60,100)
            assert dogs[0].xyxy == (30,10,90,50)
            return [DogPose(("nose",),((70,20),),(.9,))]
    adapter = OrientedPoseAdapter(Adapter())
    result = adapter.estimate(VideoFrame(0,0,np.zeros((60,100,3),dtype=np.uint8)),
                              [AngledDetection((10,10,70,50),.9,180)])[0]
    assert result.xy == ((29,39),)


def test_gallery_keeps_all_videos_and_original_partitions(tmp_path,monkeypatch):
    import hashlib
    import src.demo.gallery as module
    videos = tmp_path/'videos'
    videos.mkdir()
    (videos/'one.mov').write_bytes(b'one')
    (videos/'two.mov').write_bytes(b'two')
    (videos/'extra.MOV').write_bytes(b'extra')
    trained = tmp_path/'trained'
    trained.mkdir()
    pd.DataFrame([{'source_sha256':hashlib.sha256(b'one').hexdigest(),'split':'test'}]).to_csv(trained/'split.csv',index=False)
    monkeypatch.setattr(module,'workbook_rows',lambda _:[{'A':'one.mov'},{'A':'two.mov'}])
    inventory(tmp_path/'unused.xlsx',videos,trained,tmp_path/'out')
    result = pd.read_csv(tmp_path/'out/manifest.csv')
    assert len(result)==3
    assert result.set_index('video_id').loc['one.mov','split']=='test'
    assert (result.split=='unlabelled').sum()==2


def test_detector_dropout_uses_image_motion_and_limits_expensive_rescue():
    from src.demo.robust_detection import RobustDogDetector
    detector = object.__new__(RobustDogDetector)
    detector.primary,detector.rescue='primary','rescue'
    detector.last_rescue=-100.
    detector.previous_gray=None
    detector.previous=[]
    detector.previous_time=None
    calls=[]
    def infer(pixels,model,*args):
        calls.append(model)
        return [AngledDetection((20,20,120,120),.8)] if model=='rescue' else []
    detector.infer=infer
    rng=np.random.default_rng(7)
    image=cv2.GaussianBlur(rng.integers(0,255,(160,160,3),dtype=np.uint8),(3,3),0)
    first=detector.detect(VideoFrame(0,0,image))
    shifted=cv2.warpAffine(image,np.float32([[1,0,2],[0,1,0]]),(160,160))
    second=detector.detect(VideoFrame(1,.1,shifted))
    assert len(first)==len(second)==1
    assert abs(second[0].xyxy[0]-22)<1
    assert calls.count('rescue')==1


def test_between_model_frames_follow_image_motion_instead_of_stale_coordinates():
    from src.demo.pose_flow import propagate
    rng=np.random.default_rng(3)
    gray=rng.integers(0,255,(96,96),dtype=np.uint8)
    image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    shifted=cv2.warpAffine(image,np.float32([[1,0,2],[0,1,0]]),(96,96))
    pose=DogPose(('a','b','c'),((40,40),(50,40),(40,50)),(.9,.9,.9))
    dogs,poses=propagate(image,shifted,[DogDetection((20,20,75,75),.9)],[pose])
    assert len(dogs)==len(poses)==1
    assert abs(poses[0].xy[0][0]-42)<.3
    assert abs(dogs[0].xyxy[0]-22)<.3
