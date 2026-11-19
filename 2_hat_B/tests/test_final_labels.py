import numpy as np
import pandas as pd
import pytest
import torch

from src.deep_learning.final_labels import audit_labels, TARGETS, CLASSES
from src.deep_learning.train import model_targets, FusionNet
from src.demo.analyze_upload import infer


def fixtures():
    old=pd.DataFrame(dict(video_id=['a.mov','b.mov','c.mov','d.mov'],source_sha256=['a','b','c','d'],
                          split=['train','unlabelled','test','unlabelled']))
    rows=[]
    for name in ['d.mov','b.mov','a.mov','c.mov']:
        rows.append(dict(video=name,durum='tam',guvensiz='0',degisken='0',konsensus_disi='',
                         **{t:CLASSES[t][0] for t in TARGETS}))
    labels=pd.DataFrame(rows)
    return old,labels


def test_final_labels_align_by_name_and_keep_test_out_of_training():
    old,labels=fixtures()
    labels.loc[labels.video.eq('b.mov'),'guvensiz']='1'
    labels.loc[labels.video.eq('c.mov'),'kulak']=''
    labels.loc[labels.video.eq('a.mov'),'konsensus_disi']='kuyruk_konum'
    annotations,y,train,test,audit=audit_labels(labels,old)
    assert annotations.video.tolist()==old.video_id.tolist()
    assert train.tolist()==[0,3] and test.tolist()==[2]
    assert (y[1]==-1).all()
    assert y[2,TARGETS.index('kulak')]==-1
    assert y[0,TARGETS.index('kuyruk_konum')]==-1
    assert y[0,TARGETS.index('hareket')]==0
    assert audit['newly_usable_training']==1


def test_reject_unknown_labels_instead_of_inventing_mapping():
    old,labels=fixtures();labels.loc[0,'ses']='mutlu'
    with pytest.raises(ValueError,match='Unknown label'):audit_labels(labels,old)


def test_reject_hash_leakage_and_duplicate_video_labels():
    old,labels=fixtures();old.loc[2,'source_sha256']='a'
    with pytest.raises(ValueError,match='leakage'):audit_labels(labels,old)
    old,labels=fixtures();labels.loc[0,'video']='b.mov'
    with pytest.raises(ValueError,match='Duplicate'):audit_labels(labels,old)


def test_unsafe_test_is_masked_without_moving_to_training():
    old,labels=fixtures();labels.loc[labels.video.eq('c.mov'),'guvensiz']='1'
    _,y,train,test,audit=audit_labels(labels,old)
    assert (y[test]==-1).all() and not set(train)&set(test)
    assert test.tolist()==[2] and audit['evaluable_test_videos']==0


def test_six_head_inference_uses_checkpoint_order_and_preserves_legacy():
    model=FusionNet(2,2,[len(CLASSES[t]) for t in TARGETS]).eval()
    stats=dict(median=[0,0],mean=[0,0],scale=[1,1])
    meta=dict(targets=TARGETS,protocol=dict(classes=CLASSES,visual_columns=['v_x','v_y']),
              visual_scaler=stats,audio_scaler=stats)
    result=infer(meta,model,dict(v_x=1,v_y=2),np.array([1,2]))
    assert list(result['predictions'])==TARGETS
    assert 'vucut' not in result['predictions'] and 'aktivite' not in result['predictions']
    assert len(model_targets({}))==5
    assert model_targets(dict(protocol=dict(targets=TARGETS)))==TARGETS
    for t in TARGETS:
        assert set(result['scores'][t])==set(CLASSES[t])
        assert sum(result['scores'][t].values())==pytest.approx(1,abs=1e-6)
