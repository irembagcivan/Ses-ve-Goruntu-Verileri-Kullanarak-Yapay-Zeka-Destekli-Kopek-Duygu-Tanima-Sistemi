import numpy as np
import pytest
from src.demo.temporal import intervals,visual_window,predict_window
from src.demo.extract import audio_features_from_wave


def record(time,x,dogs=1):
    return {'timestamp_seconds':time,'dog_count':dogs,'bbox_xyxy':[0,0,100,100],
            'xy':[[x,50]],'confidence':[1.],'keypoint_names':['nose']}


def test_window_bounds_and_tail():
    assert list(intervals(7))==[(0.,4.),(2.,6.),(4.,7)]
    assert list(intervals(.5))==[]
    with pytest.raises(ValueError):list(intervals(-1))


def test_motion_does_not_cross_window_boundary():
    records=[record(0,0),record(1,20),record(2,21)]
    features,_=visual_window(records,1,3,200,200,n=1)
    assert features['v_speed_0_mean']==pytest.approx(1/np.hypot(100,100))


def test_missing_detection_breaks_motion_chain():
    records=[record(0,0),{'timestamp_seconds':1,'dog_count':0},record(2,90)]
    features,quality=visual_window(records,0,3,200,200,n=1)
    assert np.isnan(features['v_speed_0_mean'])
    assert quality['detected_frames']==2


def test_multi_dog_window_does_not_invent_identity_or_velocity():
    features,quality=visual_window([record(0,0,2),record(1,90,2)],0,2,200,200,n=1)
    assert np.isnan(features['v_speed_0_mean'])
    assert quality['multi_dog_frames']==2
    assert 'no guaranteed identity' in quality['subject_scope']


def test_missing_modalities_prevent_imputed_false_confidence():
    entries={'aktivite':{'columns':['v_pose_available'],'model':None},
             'kulak':{'columns':['y_available'],'model':None}}
    pred,scores,missing=predict_window(entries,{'v_pose_available':0,'y_available':0})
    assert not pred and not scores
    assert missing=={'aktivite':'pose_missing','kulak':'audio_events_missing'}


def test_decoded_audio_rejects_nonfinite_short_or_stereo():
    for wave in [[],[np.nan]*512,np.zeros((512,2))]:
        features,status=audio_features_from_wave(wave)
        assert features['a_available']==0


def test_all_low_confidence_keypoints_do_not_count_as_available_pose():
    r=record(0,50);r['confidence']=[.1]
    features,_=visual_window([r],0,2,200,200,n=1)
    assert features['v_pose_available']==0
