import copy
import pandas as pd
import pytest
from src.demo.annotation_review import LABELS, manifest_digest, validate_review, review_rows


def fixture():
    labels={t:values[0] for t,values in LABELS.items()}
    manifest=pd.DataFrame([{'video_id':name,'source_sha256':name+'hash','split':split,**labels,'note':''}
                           for name,split in [('old','train'),('test','test'),('new','unlabelled')]])
    manifest.loc[2,'aktivite']=''
    record={'video_id':'new','source_sha256':'newhash','labels':labels,'note':'Human review','status':'reviewed_full'}
    payload={'schema_version':1,'manifest_digest':manifest_digest(manifest),'records':[record]}
    return manifest,payload


def test_human_import_preserves_all_splits_and_original_labelled_rows():
    manifest,payload=fixture();result,accepted=validate_review(manifest,payload)
    assert accepted==['new']
    pd.testing.assert_frame_equal(result.iloc[:2],manifest.iloc[:2])
    assert result.split.tolist()==manifest.split.tolist()
    assert result.iloc[2].aktivite==payload['records'][0]['labels']['aktivite']

@pytest.mark.parametrize('status',['pending','preview_only','blocked'])
def test_non_full_reviews_cannot_become_training_annotations(status):
    manifest,payload=fixture();payload['records'][0]['status']=status
    with pytest.raises(ValueError,match='No completed'):validate_review(manifest,payload)

@pytest.mark.parametrize('field,value',[('video_id','test'),('video_id','old'),('source_sha256','wrong')])
def test_protect_existing_data_and_content_identity(field,value):
    manifest,payload=fixture();payload['records'][0][field]=value
    with pytest.raises(ValueError):validate_review(manifest,payload)


def test_reject_stale_base_manifest():
    manifest,payload=fixture();manifest.loc[2,'note']='Changed note is not identity'
    validate_review(manifest,payload)
    manifest.loc[2,'vucut']='gergin'
    with pytest.raises(ValueError,match='different manifest'):validate_review(manifest,payload)


def test_model_suggestions_are_never_imported_as_labels():
    manifest,payload=fixture();payload['records'][0]['suggestions']={'aktivite':{'label':'oynuyor_kosuyor'}}
    result,_=validate_review(manifest,payload)
    assert result.iloc[2].aktivite=='hareketsiz'


def test_reject_missing_or_invalid_labels_and_duplicates():
    manifest,payload=fixture()
    for value in ['','not-a-label']:
        candidate=copy.deepcopy(payload);candidate['records'][0]['labels']['aktivite']=value
        with pytest.raises(ValueError):validate_review(manifest,candidate)
    payload['records'].append(copy.deepcopy(payload['records'][0]))
    with pytest.raises(ValueError,match='Duplicate'):validate_review(manifest,payload)


def test_digest_stable_under_row_order():
    manifest,_=fixture()
    assert manifest_digest(manifest)==manifest_digest(manifest.iloc[::-1])


def test_prioritization_keeps_existing_partial_labels_separate_from_predictions():
    manifest,_=fixture()
    pred={t:values[-1] for t,values in LABELS.items()}
    sample={'video_id':'new','source_sha256':'newhash','predictions':{'cv_selected':pred},
            'scores':{'cv_selected':{t:{value:1/len(values) for value in values} for t,values in LABELS.items()}}}
    rows=review_rows(manifest,{'samples':[sample]})
    assert len(rows)==1
    assert rows[0]['labels']['aktivite']==''
    assert rows[0]['labels']['vucut']==manifest.iloc[2].vucut
    assert rows[0]['suggestions']['aktivite']['label']==pred['aktivite']
    assert rows[0]['status']=='pending'

@pytest.mark.parametrize('payload',[[],{}, {'records':None}])
def test_reject_invalid_backup_structure(payload):
    manifest,_=fixture()
    with pytest.raises(ValueError):validate_review(manifest,payload)
