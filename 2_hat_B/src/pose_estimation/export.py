"""Streaming, strict-JSON pose export, independent of visualization profiles."""
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from src.pose_estimation.metadata import load_superanimal_metadata


TIMESTAMP_SOURCE = 'frame_index_div_fps'


def number_or_null(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def frame_record(frame_index, fps, width, height, detections, poses, names):
    if len(detections) != len(poses):
        raise ValueError('Detection/pose count mismatch during export')
    dogs=[]
    for index,(dog,pose) in enumerate(zip(detections,poses)):
        if pose.keypoint_names != names or len(pose.xy)!=len(names) or len(pose.confidence)!=len(names):
            raise ValueError('Pose export keypoint schema mismatch')
        if len(dog.xyxy)!=4:
            raise ValueError('Detection bounding box must contain four coordinates')
        points=[]
        for i,(name,xy,score) in enumerate(zip(names,pose.xy,pose.confidence)):
            if len(xy)!=2:raise ValueError('Keypoint must have two coordinates')
            x,y=map(number_or_null,xy)
            confidence=number_or_null(score)
            points.append({'index':i,'name':name,'x':x,'y':y,'confidence':confidence,
                           'in_frame':x is not None and y is not None and 0<=x<width and 0<=y<height,
                           'confidence_valid':confidence is not None and 0<=confidence<=1})
        dogs.append({'detection_index':index,'bbox_xyxy':[number_or_null(v) for v in dog.xyxy],
                     'detection_confidence':number_or_null(dog.confidence),'keypoints':points})
    return {'frame_index':frame_index,'timestamp_seconds':frame_index/fps,
            'timestamp_source':TIMESTAMP_SOURCE,'frame_width':width,'frame_height':height,'dogs':dogs}


def metadata_path(pose_output):
    return Path(pose_output).with_suffix('.metadata.json')


def validate_export_paths(pose_output, *protected_paths):
    path=Path(pose_output)
    if path.suffix.lower()!='.jsonl':raise ValueError('Pose output must end in .jsonl')
    companion=metadata_path(path)
    if {path.resolve(),companion.resolve()} & {Path(p).resolve() for p in protected_paths if p is not None}:
        raise ValueError('Pose export must not replace input or output artifacts')
    if path.exists() or companion.exists():
        raise FileExistsError('Pose JSONL or companion metadata already exists; choose a new path')
    return path,companion


def checkpoint_digest(path):
    if path is None:return {'sha256':None,'status':'not_provided'}
    try:
        with Path(path).open('rb') as stream:
            return {'sha256':hashlib.file_digest(stream,'sha256').hexdigest(),'status':'hashed'}
    except OSError:
        return {'sha256':None,'status':'unreadable'}


def run_metadata(detector_path=None, pose_path=None, device='cpu', detection_confidence=None,
                 keypoint_confidence=.5, margin=None, visualization=False):
    # Reuse the privacy-limited collector: no Git authors, remotes or user paths.
    from scripts.collect_environment import git_info
    evidence,schema=load_superanimal_metadata()
    mode='official'
    if visualization:
        from src.pose_estimation.dog_skeleton import visualization_profile
        mode=visualization_profile(visualization)
    return {'metadata_schema_version':1,
            'models':{'detection':{'id':'yolov8_detection_local',**checkpoint_digest(detector_path)},
                      'pose':{'id':evidence['model_name'],**checkpoint_digest(pose_path)}},
            'device':device,
            'thresholds':{'detection_confidence':number_or_null(detection_confidence),
                          'visualization_keypoint_confidence':number_or_null(keypoint_confidence)},
            'crop_margin':number_or_null(margin),'visualization_profile':mode,
            'git':git_info(),
            'keypoint_schema':{'id':'superanimal_quadruped_39','version':1,
                               'count':len(schema.names),'names':list(schema.names)},
            'timestamp_source':TIMESTAMP_SOURCE,'source_pts_verified':False,
            'coordinate_system':'original_frame_pixels_xy','tracking_enabled':False,
            'raw_export_confidence_filter_applied':False}


class PoseJsonlWriter:
    """Own two exclusive files; retain a failed run's prefix with status=failed."""
    def __init__(self, path, names, fps, width, height, metadata):
        _,official=load_superanimal_metadata()
        if names!=official.names:raise ValueError('Export requires original SuperAnimal 39-keypoint order')
        if not math.isfinite(fps) or fps<=0 or width<=0 or height<=0:
            raise ValueError('Export requires positive FPS and frame dimensions')
        self.names=names;self.fps=fps;self.width=width;self.height=height
        self.count=0;self.closed=False
        path,companion=validate_export_paths(path)
        self.metadata=dict(metadata)
        self.metadata.update(video={'fps':fps,'width':width,'height':height},
                             run_status='in_progress',frames_written=0)
        path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=path.open('x',encoding='utf-8')
        try:
            self.metadata_stream=companion.open('x',encoding='utf-8')
        except BaseException:
            self.stream.close();path.unlink(missing_ok=True)
            raise
        try:
            self._write_metadata()
        except BaseException:
            self.stream.close();self.metadata_stream.close()
            path.unlink(missing_ok=True);companion.unlink(missing_ok=True)
            raise

    def _write_metadata(self):
        content=json.dumps(self.metadata,indent=2,allow_nan=False)+'\n'
        self.metadata_stream.seek(0)
        self.metadata_stream.write(content)
        self.metadata_stream.truncate();self.metadata_stream.flush()

    def write(self, frame_index, detections, poses):
        if self.closed:raise ValueError('Pose exporter is closed')
        if type(frame_index) is not int or frame_index!=self.count:raise ValueError('Pose export frames must be sequential from zero')
        record=frame_record(frame_index,self.fps,self.width,self.height,detections,poses,self.names)
        self.stream.write(json.dumps(record,allow_nan=False)+'\n');self.stream.flush()
        self.count+=1

    def close(self, complete):
        if self.closed:return
        self.closed=True
        try:
            self.metadata.update(run_status='complete' if complete else 'failed',frames_written=self.count)
            self._write_metadata()
        finally:
            try:self.stream.close()
            finally:self.metadata_stream.close()
