"""Compare 18 fixed candidates per target; select using training-only grouped CV."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.audio_analysis.behavior import aligned_table
from src.demo.train import TARGETS, emotion_rule


def feature_sets(table):
    visual=[c for c in table if c.startswith('v_')]
    audio=[c for c in table if c.startswith('a_')]
    events=[c for c in table if c.startswith('y_')]
    dog=[c for c in events if not c.startswith(('y_Speech_','y_Music_'))]
    return {'visual':visual,'audio':audio,'events':events,'fusion':visual+audio,
            'fusion_events':visual+audio+events,'fusion_dog_events':visual+audio+dog}


def estimator(name, labels):
    imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
    if len(set(labels))<2:return make_pipeline(imputer,DummyClassifier(strategy='most_frequent'))
    if name=='logistic':
        return make_pipeline(imputer,StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',
                              max_iter=3000,random_state=42))
    cls={'random_forest':RandomForestClassifier,'extra_trees':ExtraTreesClassifier}[name]
    return make_pipeline(imputer,cls(n_estimators=300,min_samples_leaf=2,max_depth=8,
                                    class_weight='balanced',random_state=42,n_jobs=2))


def select_candidate(records):
    """Test scores cannot participate: only CV mean and stable candidate ID."""
    if not records:raise ValueError('No candidates')
    return min(records,key=lambda r:(-r['cv_macro_f1_mean'],r['candidate']))['candidate']


def training_folds(table,target):
    valid=~table[target].isin(['','gorunmuyor','etiket_yok'])
    indices=np.flatnonzero((table.split.eq('train')&valid).to_numpy())
    if len(indices)<6:raise ValueError('Insufficient training rows')
    groups=table.iloc[indices].source_sha256.to_numpy()
    if len(set(groups))<3:raise ValueError('Insufficient training groups')
    folds=[(indices[a],indices[b]) for a,b in GroupKFold(3).split(indices,groups=groups)]
    for a,b in folds:
        if set(table.iloc[a].source_sha256)&set(table.iloc[b].source_sha256):raise ValueError('CV content leakage')
    return indices,folds


def metric(actual,predicted,labels):
    return {'accuracy':float(accuracy_score(actual,predicted)),
            'balanced_accuracy':float(balanced_accuracy_score(actual,predicted)),
            'macro_f1':float(f1_score(actual,predicted,labels=labels,average='macro',zero_division=0)),
            'classes':labels,'confusion_matrix':confusion_matrix(actual,predicted,labels=labels).tolist()}


def paired_interval(actual,baseline,predicted,groups,labels,iterations=1000):
    """Descriptive paired content-group bootstrap; not independent validation."""
    actual,baseline,predicted,groups=map(np.asarray,(actual,baseline,predicted,groups))
    unique=np.unique(groups);rng=np.random.default_rng(20261108);deltas=[]
    for _ in range(iterations):
        indices=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
        a=f1_score(actual[indices],baseline[indices],labels=labels,average='macro',zero_division=0)
        b=f1_score(actual[indices],predicted[indices],labels=labels,average='macro',zero_division=0)
        deltas.append(b-a)
    return list(map(float,np.quantile(deltas,[.025,.975])))


def run(args):
    report=json.loads(args.report.read_text())
    features=pd.concat([pd.read_csv(args.old_features),pd.read_csv(args.new_features)],ignore_index=True)
    table=aligned_table(pd.read_csv(args.manifest).fillna(''),features,report)
    x=table.replace([np.inf,-np.inf],np.nan)
    args.output.mkdir(parents=True,exist_ok=False)
    sets=feature_sets(table);records=[];choices={};models={};fold_manifest={};selected_metrics=[]
    samples={s['video_id']:s for s in report['samples']}
    # COMPLETE ALL training CV and freeze choices before accessing test outcomes.
    for target in TARGETS:
        train_indices,folds=training_folds(table,target)
        labels=sorted(set(table.iloc[train_indices][target]))
        fold_manifest[target]=[{'fit_ids':table.iloc[a].video_id.tolist(),'validation_ids':table.iloc[b].video_id.tolist(),
                               'unseen_validation_classes':sorted(set(table.iloc[b][target])-set(table.iloc[a][target]))}
                              for a,b in folds]
        for name in ['logistic','random_forest','extra_trees']:
            for features_name,columns in sets.items():
                scores=[];oof_actual=[];oof_predicted=[]
                for a,b in folds:
                    model=estimator(name,table.iloc[a][target])
                    model.fit(x.iloc[a][columns],table.iloc[a][target])
                    predicted=model.predict(x.iloc[b][columns]);actual=table.iloc[b][target]
                    scores.append(float(f1_score(actual,predicted,labels=labels,average='macro',zero_division=0)))
                    oof_actual.extend(actual);oof_predicted.extend(predicted)
                entry={'target':target,'candidate':name+' / '+features_name,'algorithm':name,'features':features_name,
                       'feature_count':len(columns),'cv_macro_f1_mean':float(np.mean(scores)),
                       'cv_macro_f1_std':float(np.std(scores)),'cv_fold_macro_f1':scores,
                       'oof':metric(oof_actual,oof_predicted,labels)}
                records.append(entry)
                (args.output/'cv_progress.json').write_text(json.dumps(records,ensure_ascii=False))
                print(f"CV {target} {entry['candidate']}: {entry['cv_macro_f1_mean']:.3f}",flush=True)
        choices[target]=select_candidate([r for r in records if r['target']==target])
    selection={'selected_by':'Training-only 3-fold GroupKFold mean macro F1; fixed lexical tie-break',
               'choices':choices,'folds':fold_manifest,'seed':42,'candidate_count_per_target':18}
    (args.output/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    # Frozen choices above: test evaluation cannot alter them.
    for target in TARGETS:
        train_indices,_=training_folds(table,target)
        test_indices=np.flatnonzero((table.split.eq('test') & ~table[target].isin(['','gorunmuyor','etiket_yok'])).to_numpy())
        labels=sorted(set(table.iloc[np.concatenate([train_indices,test_indices])][target]))
        actual=table.iloc[test_indices][target].to_numpy()
        for entry in [r for r in records if r['target']==target]:
            columns=sets[entry['features']];model=estimator(entry['algorithm'],table.iloc[train_indices][target])
            model.fit(x.iloc[train_indices][columns],table.iloc[train_indices][target])
            predicted=model.predict(x.iloc[test_indices][columns])
            entry['test']=metric(actual,predicted,labels)
            entry['train_accuracy']=float(accuracy_score(table.iloc[train_indices][target],model.predict(x.iloc[train_indices][columns])))
            entry['selected_on_training_cv']=entry['candidate']==choices[target]
            if not entry['selected_on_training_cv']:continue
            models[target]={'model':model,'columns':columns,'candidate':entry['candidate']}
            baseline=np.array([samples[table.iloc[i].video_id]['predictions']['fusion'][target] for i in test_indices])
            interval=paired_interval(actual,baseline,predicted,table.iloc[test_indices].source_sha256,labels)
            entry['test_delta_vs_old_fusion']=entry['test']['macro_f1']-f1_score(actual,baseline,labels=labels,average='macro',zero_division=0)
            entry['test_delta_bootstrap_95_interval']=interval
            majority=DummyClassifier(strategy='most_frequent').fit(x.iloc[train_indices][columns],table.iloc[train_indices][target]).predict(x.iloc[test_indices][columns])
            selected_metrics.append({'modality':'cv_selected','target':target,'train_videos':len(train_indices),
                'test_videos':len(test_indices),**entry['test'],
                'majority_accuracy':float(accuracy_score(actual,majority)),
                'majority_macro_f1':float(f1_score(actual,majority,labels=labels,average='macro',zero_division=0))})
            for video,pred,score in zip(table.video_id,model.predict(x[columns]),model.predict_proba(x[columns])):
                samples[video]['predictions'].setdefault('cv_selected',{})[target]=pred
                samples[video]['scores'].setdefault('cv_selected',{})[target]=dict(zip(model.classes_,map(float,score)))
            entry['test_errors']=[{'video_id':table.iloc[i].video_id,'actual':str(truth),'predicted':str(pred),
                                   'old_fusion':samples[table.iloc[i].video_id]['predictions']['fusion'][target]}
                                  for i,truth,pred in zip(test_indices,actual,predicted) if truth!=pred]
            print(f"SELECTED {target}: {entry['candidate']}; test F1={entry['test']['macro_f1']:.3f}; delta interval={interval}",flush=True)
    for sample in report['samples']:
        sample['emotion_rule']['cv_selected']=emotion_rule(sample['predictions']['cv_selected'],sample['scores']['cv_selected'])
    report['metrics'].extend(selected_metrics)
    report['cv_experiment']={'selection':selection['selected_by'],'choices':choices,'results':[
        {k:v for k,v in r.items() if k not in {'test_errors','oof'}} for r in records if r['selected_on_training_cv']]}
    report['limitations'].append('Yeni CV modeli yalnız eğitimde seçilir. Önceden incelenmiş 25 videoluk test bağımsız yeni doğrulama değildir; fark aralıkları keşifseldir.')
    audit={target:{split:table.loc[(table.split==split),target].value_counts().to_dict() for split in ['train','test','unlabelled']} for target in TARGETS}
    experiment={'protocol':selection,'candidates':records,'class_counts':audit,
        'partitions':table.split.value_counts().to_dict(),'dataset_split_sha256':hashlib.sha256(table[['video_id','source_sha256','split']].to_csv(index=False).encode()).hexdigest(),
        'limits':['Exploratory: holdout previously inspected, not a fresh independent test.',
                  'No dog identities; exact-content grouping cannot prevent same-dog/background leakage.',
                  '18 candidates per target; CV winner scores have selection optimism.',
                  'Bootstrap intervals resample test content groups, fixed class set, 1000 draws; descriptive only.',
                  'No emotion or temporal behavior ground truth; all behavior metrics are video-level.']}
    (args.output/'experiments.json').write_text(json.dumps(experiment,ensure_ascii=False,indent=2))
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False))
    pd.DataFrame([{k:v for k,v in e.items() if k not in {'oof','test_errors','cv_fold_macro_f1','test'}} | {'test_macro_f1':e['test']['macro_f1'],'test_accuracy':e['test']['accuracy']} for e in records]).to_csv(args.output/'experiments.csv',index=False)
    joblib.dump({'models':{'cv_selected':models},'fit_split':'train_only','selection':selection},args.output/'models.joblib')
    print('Experiment complete:',args.output,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','old-features','new-features','report','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
