import numpy as np
import pytest
from src.affect.predict import vocal_segment,assessment,infer_affect
from src.affect.train import embedding_features
from src.evaluation.late_fusion import blend


def test_fusion_is_convex_and_rejects_missing_or_invalid_members():
    p={'v':[np.array([[.8,.2]])],'a':[np.array([[.2,.8]])]}
    np.testing.assert_allclose(blend(p,{'v':.75,'a':.25})[0],[[.65,.35]])
    with pytest.raises(ValueError):blend(p,{'v':-1})
    with pytest.raises(ValueError):blend(p,{'missing':1})
    with pytest.raises(ValueError):blend({'v':[np.array([[2.,0.]])]}, {'v':1})


def test_vocal_segment_excludes_speech_music_and_does_not_join_gaps():
    e=np.ones((12,1024),np.float32);e[8:]=3
    p=np.zeros((12,3));p[:,0]=.9;p[3:8,1]=.95
    out=vocal_segment(e,p,['Bark','Speech','Music'])
    assert out['patches']<=4
    assert out['end']<3 or out['start']>=8*.48
    p[:,2]=.8
    assert vocal_segment(e,p,['Bark','Speech','Music']) is None


def test_mean_std_embedding_matches_single_window_case():
    f=embedding_features(np.ones((1,1024),np.float32))
    assert f.shape==(2048,) and np.all(f[:1024]==1) and np.all(f[1024:]==0)
    with pytest.raises(ValueError):embedding_features(np.empty((0,1024)))


def test_missing_affect_never_becomes_neutral_or_happy(tmp_path):
    assert infer_affect(None,None,[],tmp_path)['status']=='model_unavailable'
    result=assessment({'audio_affect':{'status':'no_audio','reason':'Ses yok'}})
    assert result['label']=='Duygu durumu belirlenemedi'
    assert result['emotion_accuracy_on_local_videos_measured'] is False


def test_audio_affect_is_not_replaced_by_behavior_rule():
    analysis={'audio_affect':{'status':'limited_evidence','label':'Olumsuz · düşük uyarılma','support':'sınırlı'},
              'preferred_model':'x','model_comparisons':{'x':{'predictions':{'hareket':'kosuyor_oynuyor'}}},
              'quality':{'multi_dog_frames':2}}
    result=assessment(analysis)
    assert result['label']=='Olumsuz · düşük uyarılma' and result['multi_dog_warning']
