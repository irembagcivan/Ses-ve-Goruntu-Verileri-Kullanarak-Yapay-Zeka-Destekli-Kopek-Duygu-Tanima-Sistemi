"""Local YOLOv8 dog instance masks, never bounding-box masks or implicit downloads."""
from pathlib import Path
import math
import numpy as np
from .interfaces import DogDetection,DogSegmentation

class YoloDogSegmenter:
    def __init__(self,weights=None,device='cpu',confidence=.3,model=None):
        if not 0<=confidence<=1:raise ValueError('Segmentation confidence must be in [0, 1]')
        if model is None:
            if weights is None or not Path(weights).is_file():raise FileNotFoundError('Provide a local YOLOv8 segmentation checkpoint')
            from ultralytics import YOLO
            model=YOLO(str(Path(weights).resolve()),task='segment')
            cfg=model.model.yaml;layers=cfg.get('backbone',[])+cfg.get('head',[])
            if not layers or not any(layer[2]=='C2f' for layer in layers) or layers[-1][2]!='Segment':raise ValueError('Expected a YOLOv8 segmentation architecture')
        if model.task!='segment':raise ValueError('A segmentation model is required')
        entries=model.names.items() if isinstance(model.names,dict) else enumerate(model.names)
        ids=[int(i) for i,name in entries if str(name).strip().lower()=='dog']
        if len(ids)!=1:raise ValueError('Exactly one dog class must be present')
        self.model,self.dog_id,self.device,self.confidence=model,ids[0],device,confidence
    def segment(self,frame):
        height,width=frame.pixels.shape[:2];instances=[]
        for result in self.model.predict(source=frame.pixels,classes=[self.dog_id],conf=self.confidence,
                device=self.device,retina_masks=True,batch=1,verbose=False,save=False):
            if result.masks is None or result.boxes is None:continue
            masks=result.masks.data.cpu().numpy()
            if masks.ndim!=3 or masks.shape[1:]!=(height,width):raise ValueError('Mask coordinates do not match the input image')
            boxes=result.boxes.xyxy.cpu().tolist();scores=result.boxes.conf.cpu().tolist();classes=result.boxes.cls.cpu().tolist()
            if not len(masks)==len(boxes)==len(scores)==len(classes):raise ValueError('Mask and instance counts differ')
            for box,score,cls,mask in zip(boxes,scores,classes,masks):
                if int(cls)!=self.dog_id or not math.isfinite(score) or score<self.confidence:continue
                if len(box)!=4 or not all(math.isfinite(v) for v in box) or box[2]<=box[0] or box[3]<=box[1]:continue
                if not np.isfinite(mask).all():continue
                binary=np.asarray(mask>=.5,dtype=bool)
                if not binary.any():continue
                instances.append(DogSegmentation(DogDetection(tuple(map(float,box)),float(score)),binary))
        return sorted(instances,key=lambda obj:(obj.detection.xyxy[0],obj.detection.xyxy[1]))
