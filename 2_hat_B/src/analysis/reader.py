"""Read one JSONL frame at a time, preserving an invalid row as a chain break."""
from dataclasses import dataclass, field
import hashlib
import json
import math


@dataclass
class Frame:
    line_number: int
    index: int | None = None
    timestamp: float | None = None
    width: int | None = None
    height: int | None = None
    timestamp_source: str = 'unspecified'
    dog_count: int | None = None
    bbox: tuple | None = None
    points: dict = field(default_factory=dict)
    point_errors: int = 0
    reason: str | None = None


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def parse_frame(data, line, names, confidence, detection_confidence):
    frame=Frame(line)
    if not isinstance(data,dict):frame.reason='invalid_record';return frame
    index=data.get('frame_index');stamp=data.get('timestamp_seconds')
    if type(index) is int and index>=0:frame.index=index
    if finite(stamp) and stamp>=0:frame.timestamp=float(stamp)
    width,height=data.get('frame_width'),data.get('frame_height')
    if type(width) is int and width>0:frame.width=width
    if type(height) is int and height>0:frame.height=height
    if data.get('timestamp_source')=='frame_index_div_fps':frame.timestamp_source='frame_index_div_fps'
    if None in (frame.index,frame.timestamp,frame.width,frame.height):
        frame.reason='invalid_frame_header';return frame
    dogs=data.get('dogs')
    if not isinstance(dogs,list):frame.reason='invalid_dogs';return frame
    frame.dog_count=len(dogs)
    if not dogs:frame.reason='no_dog';return frame
    if len(dogs)!=1:frame.reason='multiple_dogs';return frame
    dog=dogs[0]
    if not isinstance(dog,dict):frame.reason='invalid_detection';return frame
    score=dog.get('detection_confidence')
    if not finite(score) or not detection_confidence<=score<=1:
        frame.reason='unreliable_detection';return frame
    box=dog.get('bbox_xyxy')
    if (not isinstance(box,list) or len(box)!=4 or not all(finite(v) for v in box)
            or not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height):
        frame.reason='invalid_bbox';return frame
    frame.bbox=tuple(map(float,box))
    points=dog.get('keypoints')
    if not isinstance(points,list):frame.reason='invalid_keypoints';return frame
    seen=set();blocked=set()
    for point in points:
        if not isinstance(point,dict):frame.point_errors+=1;continue
        idx=point.get('index');name=point.get('name')
        if type(idx) is not int or not 0<=idx<len(names) or name!=names[idx]:
            frame.point_errors+=1;continue
        if idx in seen:
            frame.points.pop(name,None);blocked.add(idx);frame.point_errors+=1;continue
        seen.add(idx)
        if idx in blocked:continue
        x,y,score=point.get('x'),point.get('y'),point.get('confidence')
        if (not all(finite(v) for v in (x,y,score)) or not 0<=x<width or not 0<=y<height
                or not confidence<=score<=1 or point.get('in_frame') is not True
                or point.get('confidence_valid') is not True):
            frame.point_errors+=1;continue
        frame.points[name]=(float(x),float(y))
    return frame


class JsonlFrames:
    def __init__(self,path,names,confidence=.5,detection_confidence=0.0):
        self.path=path;self.names=names;self.confidence=confidence
        self.detection_confidence=detection_confidence
        self.blank_lines=0;self.nonstandard_constants=0;self.digest=hashlib.sha256()

    def __iter__(self):
        # Single pass: the hash is over the same input bytes that are parsed.
        high_index=-1
        with open(self.path,'rb') as stream:
            for line,raw in enumerate(stream,1):
                self.digest.update(raw)
                if not raw.strip():self.blank_lines+=1;continue
                def constant(value):
                    self.nonstandard_constants+=1
                    return None
                def unique(pairs):
                    result={}
                    for key,value in pairs:
                        if key in result:raise ValueError('Duplicate JSON key')
                        result[key]=value
                    return result
                try:
                    data=json.loads(raw.decode('utf-8'),parse_constant=constant,object_pairs_hook=unique)
                    frame=parse_frame(data,line,self.names,self.confidence,self.detection_confidence)
                except (ValueError,UnicodeError,RecursionError,OverflowError):
                    frame=Frame(line,reason='malformed_json')
                if frame.index is not None:
                    if frame.index<=high_index:
                        frame.reason='nonincreasing_frame_index';frame.points={}
                    else:high_index=frame.index
                yield frame
