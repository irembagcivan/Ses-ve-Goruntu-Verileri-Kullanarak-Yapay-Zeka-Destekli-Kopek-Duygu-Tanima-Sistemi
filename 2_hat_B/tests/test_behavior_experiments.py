import pandas as pd
import pytest
from src.experiments.behavior import select_candidate, training_folds, feature_sets


def test_selection_cannot_use_test_outcomes():
    records=[{'candidate':'a','cv_macro_f1_mean':.4,'test':{'macro_f1':1.}},
             {'candidate':'b','cv_macro_f1_mean':.5,'test':{'macro_f1':0.}}]
    assert select_candidate(records)=='b'
    records[0]['test']['macro_f1']=0;records[1]['test']['macro_f1']=1
    assert select_candidate(records)=='b'


def test_tie_break_is_stable():
    records=[{'candidate':'z','cv_macro_f1_mean':.4},{'candidate':'a','cv_macro_f1_mean':.4}]
    assert select_candidate(records)==select_candidate(records[::-1])=='a'


def test_grouped_cv_never_accesses_test_or_incomplete_rows():
    table=pd.DataFrame({'video_id':list('abcdefghij'),'split':['train']*8+['test','unlabelled'],
                        'source_sha256':['same','same','c','d','e','f','g','h','i','j'],
                        'target':['a','a','b','b','a','b','gorunmuyor','a','test-only','']})
    indices,folds=training_folds(table,'target')
    assert set(indices)=={0,1,2,3,4,5,7}
    assert sorted(i for _,val in folds for i in val)==sorted(indices)
    for fit,val in folds:
        assert not set(table.iloc[fit].source_sha256)&set(table.iloc[val].source_sha256)
    table.loc[8,'target']='changed test answer'
    assert [[v.tolist() for v in pair] for pair in training_folds(table,'target')[1]]==[[v.tolist() for v in pair] for pair in folds]


def test_feature_sets_cannot_contain_labels_or_identifiers():
    table=pd.DataFrame(columns=['video_id','aktivite','vucut','split','source_sha256','v_pose','a_mfcc',
                                'y_Bark_max','y_Speech_max','y_Music_mean','y_available'])
    sets=feature_sets(table)
    assert sets['fusion_events']==['v_pose','a_mfcc','y_Bark_max','y_Speech_max','y_Music_mean','y_available']
    assert sets['fusion_dog_events']==['v_pose','a_mfcc','y_Bark_max','y_available']
    assert not set(['video_id','aktivite','vucut','split','source_sha256']) & set(sum(sets.values(),[]))
