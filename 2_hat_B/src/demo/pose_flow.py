"""Move measured pose points with checked image flow between model refreshes."""
import cv2
import numpy as np
from src.preprocessing.interfaces import DogDetection
from src.pose_estimation.interfaces import DogPose


def propagate(previous, current, dogs, poses):
    before=cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY)
    after=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)
    if before.shape!=after.shape:
        return [],[]
    histogram=lambda image:np.histogram(image,bins=32,range=(0,256))[0]/image.size
    if np.mean(cv2.absdiff(before,after))>45 and np.abs(histogram(before)-histogram(after)).sum()>.8:
        return [],[]
    new_dogs,new_poses=[],[]
    levels=min(3,max(0,int(np.log2(min(after.shape)/40))))
    for dog,pose in zip(dogs,poses):
        xy=np.asarray(pose.xy,dtype=float)
        scores=np.asarray(pose.confidence,dtype=float)
        indices=np.where(np.isfinite(xy).all(axis=1)&(scores>=.45))[0]
        if len(indices)<2:
            continue
        points=xy[indices].astype(np.float32).reshape(-1,1,2)
        moved,status,error=cv2.calcOpticalFlowPyrLK(before,after,points,None,winSize=(21,21),maxLevel=levels)
        if moved is None:
            continue
        back,back_status,_=cv2.calcOpticalFlowPyrLK(after,before,moved,None,winSize=(21,21),maxLevel=levels)
        if back is None:
            continue
        valid=status.ravel().astype(bool)&back_status.ravel().astype(bool)
        valid&=np.linalg.norm(back.reshape(-1,2)-points.reshape(-1,2),axis=1)<1.5
        valid&=error.ravel()<22
        if valid.sum()<2:
            continue
        shifted=moved.reshape(-1,2)
        translation=np.median(shifted[valid]-points.reshape(-1,2)[valid],axis=0)
        box=np.asarray(dog.xyxy,dtype=float)
        diagonal=max(1,np.linalg.norm(box[2:]-box[:2]))
        if np.linalg.norm(translation)>.15*diagonal:
            continue
        predicted=xy.copy();predicted[:]=np.nan
        confidence=np.zeros_like(scores)
        predicted[indices[valid]]=shifted[valid]
        confidence[indices[valid]]=scores[indices[valid]]*.99
        box+=np.tile(translation,2)
        box[[0,2]]=np.clip(box[[0,2]],0,after.shape[1])
        box[[1,3]]=np.clip(box[[1,3]],0,after.shape[0])
        new_dogs.append(DogDetection(tuple(map(float,box)),dog.confidence))
        new_poses.append(DogPose(pose.keypoint_names,tuple(map(tuple,predicted)),tuple(map(float,confidence))))
    return new_dogs,new_poses
