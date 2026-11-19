"""Frozen-split behavior experiments using cached YAMNet event features.

All settings are fixed before evaluation. Existing models/metrics remain intact;
new variants are experimental comparisons, never selected using test scores.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from src.demo.train import TARGETS, emotion_rule
from .run import CLASSES


def event_features(analysis, seconds=60):
    """Match legacy first-60s scope; missing audio stays missing, never silence."""
    windows=[w for w in analysis.get('windows',[]) if w['end']<=seconds]
    result={'y_available':float(analysis.get('status')=='ok' and bool(windows))}
    for name in CLASSES:
        values=np.array([w['scores'][name] for w in windows],dtype=float)
        for statistic in ['mean','max','p90','fraction_ge_03']:
            key=f'y_{name}_{statistic}'
            result[key]=float({'mean':lambda:values.mean(),'max':lambda:values.max(),
                              'p90':lambda:np.quantile(values,.9),
                              'fraction_ge_03':lambda:(values>=.3).mean()}[statistic]()) if len(values) else np.nan
    return result


def aligned_table(manifest, features, report):
    if manifest.video_id.duplicated().any() or features.video_id.duplicated().any():
        raise ValueError('One row per video required')
    table=manifest.merge(features,on='video_id',validate='one_to_one',how='left',indicator=True)
    if (table['_merge']!='both').any():raise ValueError('Missing legacy feature rows')
    table=table.drop(columns='_merge')
    samples={s['video_id']:s for s in report['samples']}
    if set(samples)!=set(table.video_id):raise ValueError('Audio report and manifest must contain identical video IDs')
    if not set(table.split)<= {'train','test','unlabelled'}:raise ValueError('Unknown split')
    train=set(table.loc[table.split=='train','source_sha256'])
    test=set(table.loc[table.split=='test','source_sha256'])
    if train & test:raise ValueError('Duplicate content crosses train/test split')
    for row in table.itertuples():
        sample=samples[row.video_id]
        if sample['source_sha256']!=row.source_sha256 or sample['split']!=row.split:
            raise ValueError('Frozen report/manifest split or source mismatch')
    event=pd.DataFrame([event_features(samples[v]['audio_analysis']) for v in table.video_id])
    return pd.concat([table.reset_index(drop=True),event],axis=1)


def run(manifest, old_features, new_features, report_path, output):
    report=json.loads(report_path.read_text())
    features=pd.concat([pd.read_csv(old_features),pd.read_csv(new_features)],ignore_index=True)
    table=aligned_table(pd.read_csv(manifest).fillna(''),features,report)
    audio=[c for c in features if c.startswith('a_')]
    visual=[c for c in features if c.startswith('v_')]
    events=[c for c in table if c.startswith('y_')]
    dog_events=[c for c in events if not c.startswith(('y_Speech_','y_Music_'))]
    variants={'audio_yamnet':events,'fusion_yamnet':visual+audio+events,
              'fusion_dogevents':visual+audio+dog_events}
    output.mkdir(parents=True,exist_ok=False)
    models={};new_metrics=[]
    table[['video_id','source_sha256','split']].to_csv(output/'split.csv',index=False)
    table[['video_id']+events].to_csv(output/'event_features.csv',index=False)
    by_id={s['video_id']:s for s in report['samples']}
    for modality,columns in variants.items():
        x=table[columns].replace([np.inf,-np.inf],np.nan)
        models[modality]={}
        for target in TARGETS:
            valid=~table[target].isin(['','gorunmuyor','etiket_yok'])
            training=(table.split=='train')&valid
            testing=(table.split=='test')&valid
            if not training.any() or not testing.any():raise ValueError('No labelled train/test rows')
            classifier=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                RandomForestClassifier(n_estimators=300,min_samples_leaf=2,max_depth=8,
                                       class_weight='balanced',random_state=42,n_jobs=4))
            classifier.fit(x.loc[training],table.loc[training,target])
            actual=table.loc[testing,target];predicted=classifier.predict(x.loc[testing])
            labels=sorted(set(table.loc[valid & table.split.isin(['train','test']),target]))
            majority=DummyClassifier(strategy='most_frequent').fit(x.loc[training],table.loc[training,target]).predict(x.loc[testing])
            metric={'modality':modality,'target':target,'train_videos':int(training.sum()),'test_videos':int(testing.sum()),
                    'accuracy':accuracy_score(actual,predicted),'balanced_accuracy':balanced_accuracy_score(actual,predicted),
                    'macro_f1':f1_score(actual,predicted,labels=labels,average='macro',zero_division=0),
                    'majority_accuracy':accuracy_score(actual,majority),
                    'majority_macro_f1':f1_score(actual,majority,labels=labels,average='macro',zero_division=0),
                    'classes':labels,'confusion_matrix':confusion_matrix(actual,predicted,labels=labels).tolist()}
            new_metrics.append(metric);models[modality][target]={'model':classifier,'columns':columns}
            for video,pred,scores in zip(table.video_id,classifier.predict(x),classifier.predict_proba(x)):
                sample=by_id[video]
                sample['predictions'].setdefault(modality,{})[target]=pred
                sample['scores'].setdefault(modality,{})[target]=dict(zip(classifier.classes_,map(float,scores)))
            print(f"{modality} {target}: F1={metric['macro_f1']:.3f} accuracy={metric['accuracy']:.3f}",flush=True)
        for sample in report['samples']:
            sample['emotion_rule'][modality]=emotion_rule(sample['predictions'][modality],sample['scores'][modality])
    report['metrics'].extend(new_metrics)
    report['audio_behavior_experiment']={'scope':'first 60s; complete 0.975s windows only',
        'split':'unchanged 72 train / 25 test; 30 incomplete videos never fitted',
        'features':'per-class mean, max, p90, fraction score>=0.3; availability indicator',
        'variants':{key:len(value) for key,value in variants.items()},
        'selection':'No hyperparameter search or test-based model selection; all variants displayed',
        'limitations':'Small previously inspected holdout; no dog-identity split; no calibrated probabilities or emotion labels.'}
    report['limitations'].append('Yeni YAMNet davranış modelleri deneyseldir; aynı küçük test kümesi tekrar kullanılmıştır. Anlık davranış veya duygu doğruluğu ölçülmemiştir.')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False))
    pd.DataFrame(new_metrics).to_csv(output/'metrics.csv',index=False)
    joblib.dump({'models':models,'fit_split':'train_only','experiment':report['audio_behavior_experiment'],
                 'audio_event_model':report['audio_event_model']},output/'models.joblib')
    print('Saved:',output)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','old-features','new-features','report','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.manifest,args.old_features,args.new_features,args.report,args.output)
