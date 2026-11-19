"""Higher-capacity fallback for difficult dog detections; never relabel cats."""
from dataclasses import dataclass
import cv2
import numpy as np

from src.preprocessing.interfaces import DogDetection, VideoFrame
from src.preprocessing.detection import YoloDogDetector
from src.demo.tracking import iou


@dataclass(frozen=True)
class AngledDetection(DogDetection):
    rotation: int = 0


class RobustDogDetector:
    def __init__(self, primary, rescue):
        self.primary = YoloDogDetector(primary, "cpu", .3)
        self.rescue = YoloDogDetector(rescue, "cpu", .3)
        self.confidence = .3
        self.last_rescue = -100.
        self.previous_gray = None
        self.previous = []
        self.previous_time = None

    def infer(self, pixels, model, image_size=640, rotation=0, offset_y=0):
        height, width = pixels.shape[:2]
        image = cv2.rotate(pixels,cv2.ROTATE_180) if rotation == 180 else pixels
        results = model.model.predict(image, classes=[model.dog_id], conf=.3, imgsz=image_size,
                                      device="cpu", verbose=False, save=False)
        dogs = []
        for result in results:
            for box, confidence in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
                x1,y1,x2,y2 = box
                if rotation == 180:
                    x1,y1,x2,y2 = width-x2,height-y2,width-x1,height-y1
                dogs.append(AngledDetection((x1,y1+offset_y,x2,y2+offset_y),float(confidence),rotation))
        return dogs

    def detect(self, frame):
        pixels = frame.pixels
        timestamp = frame.timestamp_seconds
        gray = cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)
        if self.previous_time is not None and timestamp <= self.previous_time:
            self.last_rescue=-100.;self.previous=[];self.previous_gray=None
        dogs = self.infer(pixels, self.primary)
        from_detector = True
        if not dogs and timestamp-self.last_rescue >= .35:
            self.last_rescue = timestamp
            dogs += self.infer(pixels,self.rescue,960)
            if not dogs:
                dogs += self.infer(pixels,self.rescue,960,180)
                height = pixels.shape[0]
                low, high = int(height*.12),int(height*.86)
                dogs += self.infer(pixels[low:high],self.primary,960,0,low)
                dogs += self.infer(pixels[low:high],self.rescue,960,180,low)
        anchors = {}
        if not dogs and self.previous_gray is not None and gray.shape==self.previous_gray.shape:
            from_detector = False
            for previous,anchor in self.previous:
                if timestamp-anchor > .35:
                    continue
                x1,y1,x2,y2 = previous.xyxy
                mask = np.zeros(gray.shape,dtype=np.uint8)
                cx,cy = (x1+x2)/2,(y1+y2)/2
                cv2.ellipse(mask,(int(cx),int(cy)),(max(1,int((x2-x1)*.35)),max(1,int((y2-y1)*.35))),0,0,360,255,-1)
                points = cv2.goodFeaturesToTrack(self.previous_gray,maxCorners=35,qualityLevel=.02,minDistance=8,mask=mask)
                if points is None or len(points)<6:
                    continue
                moved,status,error = cv2.calcOpticalFlowPyrLK(self.previous_gray,gray,points,None,winSize=(21,21),maxLevel=2)
                if moved is None:
                    continue
                back,back_status,_ = cv2.calcOpticalFlowPyrLK(gray,self.previous_gray,moved,None,winSize=(21,21),maxLevel=2)
                if back is None:
                    continue
                valid=status.ravel().astype(bool)&back_status.ravel().astype(bool)
                valid &= np.linalg.norm(back.reshape(-1,2)-points.reshape(-1,2),axis=1)<1.5
                valid &= error.ravel()<22
                if valid.sum()<6:
                    continue
                shifts=(moved-points).reshape(-1,2)[valid]
                shift=np.median(shifts,axis=0)
                if np.linalg.norm(shift)>.15*np.hypot(x2-x1,y2-y1):
                    continue
                box=np.array(previous.xyxy)+np.tile(shift,2)
                box[[0,2]]=np.clip(box[[0,2]],0,gray.shape[1])
                box[[1,3]]=np.clip(box[[1,3]],0,gray.shape[0])
                if box[2]<=box[0] or box[3]<=box[1]:
                    continue
                dog=AngledDetection(tuple(map(float,box)),previous.confidence,previous.rotation)
                dogs.append(dog);anchors[id(dog)]=anchor
        kept = []
        for dog in sorted(dogs,key=lambda d:d.confidence,reverse=True):
            if all(iou(np.asarray(dog.xyxy),np.asarray(other.xyxy)) < .6 for other in kept):
                kept.append(dog)
        self.previous=[(dog,timestamp if from_detector else anchors[id(dog)]) for dog in kept]
        self.previous_gray=gray
        self.previous_time=timestamp
        return kept


class OrientedPoseAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
        self.schema = adapter.schema

    def estimate(self, frame, dogs):
        from src.pose_estimation.interfaces import DogPose
        poses = []
        height,width = frame.pixels.shape[:2]
        for dog in dogs:
            if getattr(dog,"rotation",0) != 180:
                poses.extend(self.adapter.estimate(frame,[dog]))
                continue
            x1,y1,x2,y2 = dog.xyxy
            rotated_box = DogDetection((width-x2,height-y2,width-x1,height-y1),dog.confidence)
            rotated_frame = VideoFrame(frame.index,frame.timestamp_seconds,cv2.rotate(frame.pixels,cv2.ROTATE_180))
            pose = self.adapter.estimate(rotated_frame,[rotated_box])[0]
            xy = tuple((width-1-x,height-1-y) for x,y in pose.xy)
            poses.append(DogPose(pose.keypoint_names,xy,pose.confidence))
        return poses
