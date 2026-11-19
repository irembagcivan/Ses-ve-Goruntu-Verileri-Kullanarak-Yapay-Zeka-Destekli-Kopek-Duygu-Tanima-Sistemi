import numpy as np
import pandas as pd
import pytest
from src.audio_analysis.behavior import event_features, aligned_table
from src.audio_analysis.run import CLASSES


def test_audio_scope_excludes_windows_crossing_60_seconds():
    def window(start,end,value):return {'start':start,'end':end,'scores':{name:value for name in CLASSES}}
    result=event_features({'status':'ok','windows':[window(0,.975,.2),window(59.5,60.475,1.)]})
    assert result['y_Bark_max']==.2
    assert result['y_Bark_fraction_ge_03']==0


def test_missing_audio_does_not_mean_silence():
    result=event_features({'status':'no_audio'})
    assert result['y_available']==0
    assert np.isnan(result['y_Bark_mean'])


def fixture():
    manifest=pd.DataFrame({'video_id':['a','b'],'source_sha256':['hash-a','hash-b'],'split':['train','test']})
    features=pd.DataFrame({'video_id':['a','b'],'v_test':[1.,2.]})
    report={'samples':[dict(video_id=r.video_id,source_sha256=r.source_sha256,split=r.split,
                            audio_analysis={'status':'no_audio'}) for r in manifest.itertuples()]}
    return manifest,features,report


def test_preserve_split_and_feature_alignment():
    manifest,features,report=fixture()
    result=aligned_table(manifest,features.iloc[::-1],report)
    assert result.v_test.tolist()==[1.,2.]
    assert result.split.tolist()==['train','test']


def test_reject_missing_features():
    manifest,features,report=fixture()
    with pytest.raises(ValueError,match='Missing legacy'):aligned_table(manifest,features.iloc[:1],report)


def test_reject_content_leakage():
    manifest,features,report=fixture();manifest.loc[1,'source_sha256']='hash-a'
    with pytest.raises(ValueError,match='crosses'):aligned_table(manifest,features,report)


def test_reject_split_drift():
    manifest,features,report=fixture();report['samples'][1]['split']='train'
    with pytest.raises(ValueError,match='mismatch'):aligned_table(manifest,features,report)
