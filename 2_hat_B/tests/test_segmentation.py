from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.preprocessing.interfaces import VideoFrame
from src.preprocessing.segmentation import YoloDogSegmenter
from src.demo.segmentation import overlay

class FakeModel:
    task='segment';names={0:'cat',16:'dog'}
    def __init__(self,result):self.result=result;self.kwargs=None
    def predict(self,**kwargs):self.kwargs=kwargs;return [self.result]

def result(masks,classes=(16,),scores=(.9,)):
    return SimpleNamespace(masks=None if masks is None else SimpleNamespace(data=torch.tensor(masks,dtype=torch.float32)),boxes=SimpleNamespace(xyxy=torch.tensor([[0.,0.,8.,6.]]*len(classes)),conf=torch.tensor(scores),cls=torch.tensor(classes)))
def frame():return VideoFrame(0,0,np.full((6,8,3),100,np.uint8))

def test_real_mask_not_bbox_and_retina_coordinates():
    mask=np.zeros((1,6,8));mask[0,1:5,2:6]=1;mask[0,2,3]=0
    model=FakeModel(result(mask));segmenter=YoloDogSegmenter(model=model);objects=segmenter.segment(frame())
    assert len(objects)==1 and objects[0].mask.dtype==bool
    assert objects[0].mask.sum()==15 and not objects[0].mask[2,3]
    assert model.kwargs['retina_masks'] is True and model.kwargs['classes']==[16]
    rendered=overlay(frame().pixels,objects)
    assert np.array_equal(rendered[5,7],frame().pixels[5,7])

def test_no_mask_never_becomes_box_rectangle():
    assert YoloDogSegmenter(model=FakeModel(result(None))).segment(frame())==[]

def test_filters_other_species_and_low_confidence():
    mask=np.ones((2,6,8));model=FakeModel(result(mask,classes=(0,16),scores=(.99,.1)))
    assert YoloDogSegmenter(model=model).segment(frame())==[]

def test_rejects_letterbox_shape_mismatch():
    with pytest.raises(ValueError,match='coordinates'):YoloDogSegmenter(model=FakeModel(result(np.ones((1,8,8))))).segment(frame())

def test_segmentation_cannot_load_detection_model():
    model=FakeModel(result(None));model.task='detect'
    with pytest.raises(ValueError,match='segmentation model'):YoloDogSegmenter(model=model)


def test_upload_worker_segmenter_argument_is_accepted():
    import subprocess,sys
    response=subprocess.run([sys.executable,'-m','src.demo.analyze_upload','--help'],capture_output=True,text=True)
    assert response.returncode==0 and '--segmenter' in response.stdout
