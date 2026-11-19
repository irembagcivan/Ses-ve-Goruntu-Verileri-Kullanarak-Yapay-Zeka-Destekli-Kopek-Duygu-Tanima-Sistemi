import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from src.demo.tracking import PreviewTracker
from src.preprocessing.interfaces import DogDetection
from src.pose_estimation.interfaces import DogPose


def pose(x, y):
    return DogPose(("nose",), ((x, y),), (.9,))


def test_all_dogs_returned_and_track_ids_survive_detection_reordering():
    tracker = PreviewTracker()
    a, b = DogDetection((0, 0, 100, 100), .9), DogDetection((200, 0, 300, 100), .9)
    first = tracker.update([a, b], [pose(50,50), pose(250,50)], 0)
    second = tracker.update([b, a], [pose(250,50), pose(50,50)], .1)
    assert len(second) == 2
    assert [t.identifier for t in second] == [first[1].identifier, first[0].identifier]


def test_missing_dog_is_not_drawn_and_long_gap_creates_new_track():
    tracker = PreviewTracker()
    dog = DogDetection((0,0,100,100), .9)
    original = tracker.update([dog], [pose(50,50)], 0)[0]
    assert tracker.update([], [], .1) == []
    assert tracker.update([dog], [pose(50,50)], 1)[0].identifier != original.identifier


def test_teleporting_keypoint_is_hidden_instead_of_held_at_old_location():
    tracker = PreviewTracker()
    dog = DogDetection((0,0,100,100), .9)
    tracker.update([dog], [pose(5,5)], 0)
    result = tracker.update([dog], [pose(90,90)], .1)[0]
    assert np.isnan(result.xy[0]).all()
    assert result.confidence[0] == 0


def test_box_translation_does_not_leave_pose_lagging_behind_dog():
    tracker = PreviewTracker()
    tracker.update([DogDetection((0,0,100,100), .9)], [pose(50,50)], 0)
    moved = tracker.update([DogDetection((20,0,120,100), .9)], [pose(70,50)], .1)[0]
    assert moved.xy[0].tolist() == [70,50]


def test_preview_uses_fresh_multi_dog_inference_on_every_output_frame(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")
    from src.demo.dense_previews import render
    from src.pose_estimation.metadata import load_superanimal_metadata
    source, output = tmp_path / "input.mp4", tmp_path / "output.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160,120))
    assert writer.isOpened()
    for _ in range(18):
        writer.write(np.zeros((120,160,3),dtype=np.uint8))
    writer.release()
    _, schema = load_superanimal_metadata()
    calls = []
    class Detector:
        def detect(self, frame):
            calls.append(frame.index)
            return [DogDetection((0,0,70,100),.9),DogDetection((80,0,150,100),.9)]
    class Adapter:
        def __init__(self):
            self.schema = schema
        def estimate(self, frame, dogs):
            assert len(dogs) == 2
            return [DogPose(schema.names,tuple([(30,50)]*len(schema.names)),tuple([.9]*len(schema.names))),
                    DogPose(schema.names,tuple([(110,50)]*len(schema.names)),tuple([.9]*len(schema.names)))]
    metadata = render(source,output,Detector(),Adapter())
    assert metadata["frames"] == len(calls) == 9
    assert calls == list(range(0,18,2))
    assert metadata["max_dogs"] == 2
    assert metadata["multi_dog_frames"] == 9
    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened() and capture.read()[0]
    assert capture.get(cv2.CAP_PROP_FPS) == 15
    capture.release()
    calls.clear()
    stable_output=tmp_path/'stable.mp4'
    stable=render(source,stable_output,Detector(),Adapter(),stable=True)
    assert stable['frames']==9
    assert len(calls)==3
    assert not stable['fresh_inference_each_displayed_frame']
    assert stable['pose_model_fps']==5
