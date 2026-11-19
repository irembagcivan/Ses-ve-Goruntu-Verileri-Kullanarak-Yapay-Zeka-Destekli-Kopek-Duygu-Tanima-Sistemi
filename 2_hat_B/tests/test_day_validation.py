import numpy as np
import pandas as pd
import pytest
from src.evaluation.recording_days import parse_tags
from src.evaluation.day_validation import day_folds
from src.evaluation.annotation_agreement import agreement
from src.deep_learning.final_labels import TARGETS,CLASSES


def test_camera_local_date_overrides_container_transcode_date():
    tags=parse_tags('    creation_time : 2026-07-01T14:00:00Z\n    com.apple.quicktime.creationdate: 2026-06-18T00:30:00+0530\nOutput #0\ncreation_time : 2026-08-01T00:00:00Z\n')
    assert tags['recording_day_candidate']=='2026-06-18'
    assert tags['container_camera_day_disagree'] is True
    assert tags['evidence']=='camera_creationdate'


def test_missing_or_invalid_timestamp_never_becomes_guessed_day():
    assert parse_tags('Input #0, mov, from IMG_5354.MOV')['recording_day_candidate']==''
    assert parse_tags('creation_time : 1904-01-01T00:00:00Z\n')['recording_day_candidate']==''
    assert parse_tags('com.apple.quicktime.creationdate: nope\n')['evidence']=='invalid_timestamp'


def test_timezone_midnight_is_not_a_metadata_conflict():
    tags=parse_tags('creation_time : 2026-06-17T19:00:00Z\ncom.apple.quicktime.creationdate: 2026-06-18T00:30:00+0530\n')
    assert tags['recording_day_candidate']=='2026-06-18'
    assert tags['container_camera_day_disagree'] is False


def test_nested_days_never_overlap_and_ignore_container_only_dates():
    metadata=pd.DataFrame(dict(video_id=list('abcdefg'),evidence=['camera_creationdate']*6+['container_creation_time_only'],
                               recording_day_candidate=['d1','d1','d2','d2','d3','d3','d4']))
    y=np.zeros((7,6),int);hashes=np.array(list('abcdefg'))
    eligible,groups,folds=day_folds(metadata,y,hashes)
    assert eligible.tolist()==list(range(6)) and len(folds)==3
    for fit,test,inner in folds:
        assert not set(groups[fit])&set(groups[test])
        for f,g in inner:
            assert set(f)|set(g)==set(fit)
            assert not set(groups[f])&set(groups[g])
            assert not set(test)&(set(f)|set(g))
    hashes[0]=hashes[2]
    with pytest.raises(ValueError,match='leakage'):day_folds(metadata,y,hashes)


def test_consensus_and_rule_files_cannot_generate_annotation_kappa():
    raw=pd.DataFrame(dict(video=['a','b'],**{t:[CLASSES[t][0],CLASSES[t][1]] for t in TARGETS}))
    for column in ['konsensus_disi','kural','kaynak']:
        wrong=raw.copy();wrong[column]=''
        with pytest.raises(ValueError,match='Independent'):agreement(wrong,raw)


def test_kappa_pairs_by_video_and_excludes_missing_without_imputing():
    raw=pd.DataFrame(dict(video=['a','b','c','d'],**{t:[CLASSES[t][0],CLASSES[t][0],CLASSES[t][1],CLASSES[t][1]] for t in TARGETS}))
    second=raw.iloc[::-1].copy();second.loc[second.video.eq('a'),'hareket']=CLASSES['hareket'][1]
    second.loc[second.video.eq('b'),'kulak']=''
    result=agreement(raw,second)
    assert result['targets'][0]['cohen_kappa']==pytest.approx(.5)
    assert result['targets'][0]['raw_agreement']==pytest.approx(.75)
    assert result['targets'][-1]['paired_labels']==3
    assert result['targets'][-1]['cohen_kappa']==pytest.approx(1)
