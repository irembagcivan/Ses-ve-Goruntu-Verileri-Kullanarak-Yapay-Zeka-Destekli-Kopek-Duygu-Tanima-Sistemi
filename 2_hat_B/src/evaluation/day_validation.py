"""Retrospective nested leave-camera-day-out validation on frozen features.

Only embedded camera creationdate is eligible. Container dates/unknown dates
never become guessed sessions. This does not manufacture a fresh independent test.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneGroupOut

from src.deep_learning.final_labels import (audit_labels, TARGETS, CLASSES, metric,
                                          mean_f1, neural_fit, classical_fit)

KINDS=['neural_visual','neural_audio','neural_fusion','sequence_cnn','sequence_gru',
       'sequence_bigru','sequence_lstm_lstm_gru']


def day_folds(metadata,y,source_hashes):
    if len(metadata)!=len(y) or metadata.video_id.duplicated().any():
        raise ValueError('Unaligned day metadata')
    eligible=np.flatnonzero(metadata.evidence.eq('camera_creationdate').to_numpy() &
                            metadata.recording_day_candidate.ne('').to_numpy() & (y>=0).any(1))
    groups=metadata.recording_day_candidate.to_numpy(str)
    if len(set(groups[eligible]))<3:
        raise ValueError('Need at least three camera days for nested grouped validation')
    folds=[]
    for a,b in LeaveOneGroupOut().split(eligible,groups=groups[eligible]):
        fit,test=eligible[a],eligible[b]
        if set(groups[fit])&set(groups[test]) or set(source_hashes[fit])&set(source_hashes[test]):
            raise ValueError('Day or source leakage')
        inner=[]
        for f,g in LeaveOneGroupOut().split(fit,groups=groups[fit]):
            train,val=fit[f],fit[g]
            if any(not (y[train,j]>=0).any() for j in range(len(TARGETS))):
                raise ValueError('No training labels for a task in inner day fold')
            inner.append((train,val))
        folds.append((fit,test,inner))
    return eligible,groups,folds


def run(args):
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=False)
    manifest=pd.read_csv(args.manifest).fillna('')
    final=pd.read_csv(args.labels).fillna('');ann,y,_,_,audit=audit_labels(final,manifest)
    meta=pd.read_csv(args.days).fillna('')
    if set(meta.video_id)!=set(manifest.video_id):raise ValueError('Day inventory mismatch')
    meta=meta.set_index('video_id').loc[manifest.video_id].reset_index()
    hashes=manifest.source_sha256.to_numpy(str)
    if not np.array_equal(meta.source_sha256.to_numpy(str),hashes):raise ValueError('Day source hash mismatch')
    with np.load(args.snapshot,allow_pickle=False) as d:
        if not np.array_equal(d['source_sha256'],hashes) or not np.array_equal(d['labels'],y):
            raise ValueError('Snapshot final label/order mismatch')
        v,a,x,lengths=[d[k].copy() for k in ['visual','audio','sequences','lengths']]
    eligible,groups,folds=day_folds(meta,y,hashes)
    protocol=dict(name='Nested leave-camera-day-out; retrospective validation',
                  source='com.apple.quicktime.creationdate, camera-local date',
                  videos=int(len(eligible)),excluded=int(len(y)-len(eligible)),
                  day_counts={day:int((groups[eligible]==day).sum()) for day in sorted(set(groups[eligible]))},
                  date_metadata_sha256=hashlib.sha256(args.days.read_bytes()).hexdigest(),
                  feature_snapshot_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
                  seed=42,epochs=[20,40,60,80],targets=TARGETS,classes=CLASSES,
                  encoders='Frozen pretrained YOLOv8n/RTMPose/YAMNet, unchanged features',
                  limitations=['Camera dates are metadata evidence, not manually verified sessions or dog identities.',
                               'All files were used in earlier development; grouped revalidation is not a new independent test.',
                               'Only three camera days; one has four usable videos. Inner training can have only four videos.',
                               'Same dog/environment may recur on different days. Rare/absent training classes remain reported.',
                               'No emotion ground truth; rule-derived emotion file never trains or evaluates a model.'])
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    outputs={k:[np.zeros((len(y),len(CLASSES[t])),np.float32) for t in TARGETS]
             for k in KINDS+['classical_visual','classical_audio','classical_fusion','majority_baseline']}
    fold_reports=[]
    for fold,(fit,test,inner) in enumerate(folds,1):
        report=dict(held_out_day=groups[test[0]],fit_videos=manifest.iloc[fit].video_id.tolist(),
                    test_videos=manifest.iloc[test].video_id.tolist(),train_count=len(fit),test_count=len(test),
                    inner=[dict(fit=manifest.iloc[f].video_id.tolist(),validation=manifest.iloc[g].video_id.tolist()) for f,g in inner],
                    class_support={t:dict(train={c:int((y[fit,j]==k).sum()) for k,c in enumerate(CLASSES[t])},
                                           test={c:int((y[test,j]==k).sum()) for k,c in enumerate(CLASSES[t])}) for j,t in enumerate(TARGETS)},
                    selection={},metrics={})
        selections={}
        for kind in KINDS:
            traces=[]
            for f,g in inner:
                _,_,trace,_=neural_fit(v,a,x,lengths,y,f,g,kind,80);traces.append(trace)
            means={e:float(np.mean([next(z['macro_f1'] for z in trace if z['epoch']==e) for trace in traces])) for e in [20,40,60,80]}
            epoch=min(means,key=lambda e:(-means[e],e));selections[kind]=dict(epochs=epoch,inner_cv=means,trace=traces)
            print(f'Outer {fold}/{len(folds)} inner CV {kind}',flush=True)
        for modality,data in [('visual',v),('audio',a),('fusion',np.concatenate([v,a],1))]:
            means={k:float(np.mean([mean_f1(classical_fit(data,y,f,g,k)[0],y[g]) for f,g in inner])) for k in ['lr','rf']}
            selections['classical_'+modality]=dict(classifier=max(means,key=means.get),inner_cv=means)
        report['selection']=selections
        # Freeze outer-fold choices before accessing this day's performance.
        (args.output/f'fold_{fold}_selection.json').write_text(json.dumps(report,indent=2))
        for kind in KINDS:
            _,_,_,p=neural_fit(v,a,x,lengths,y,fit,None,kind,selections[kind]['epochs'])
            for j in range(len(TARGETS)):outputs[kind][j][test]=p[j][test]
            report['metrics'][kind]=metric([q[test] for q in p],y[test])
        for modality,data in [('visual',v),('audio',a),('fusion',np.concatenate([v,a],1))]:
            kind='classical_'+modality;p,_=classical_fit(data,y,fit,test,selections[kind]['classifier'])
            for j in range(len(TARGETS)):outputs[kind][j][test]=p[j]
            report['metrics'][kind]=metric(p,y[test])
        p,_=classical_fit(v,y,fit,test,'dummy')
        for j in range(len(TARGETS)):outputs['majority_baseline'][j][test]=p[j]
        report['metrics']['majority_baseline']=metric(p,y[test])
        fold_reports.append(report)
        (args.output/'folds.json').write_text(json.dumps(fold_reports,indent=2))
        print(f'Outer {fold} complete; held out {groups[test[0]]}: {len(test)} videos',flush=True)
    models=[]
    for name,p in outputs.items():
        metrics=metric([q[eligible] for q in p],y[eligible])
        per_day=[float(np.mean([m['macro_f1'] for m in f['metrics'][name] if m['macro_f1'] is not None])) for f in fold_reports]
        models.append(dict(model=name,pooled_oof_macro_f1=mean_f1([q[eligible] for q in p],y[eligible]),
                           equal_day_mean_macro_f1=float(np.mean(per_day)),per_day_macro_f1=per_day,metrics=metrics))
    result=dict(protocol=protocol,models=models,folds=fold_reports,
                video_ids=manifest.iloc[eligible].video_id.tolist(),
                dates=groups[eligible].tolist(),
                predictions={name:{t:p[j][eligible].argmax(1).tolist() for j,t in enumerate(TARGETS)} for name,p in outputs.items()})
    (args.output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(models,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key,default in [('manifest','results/demo_v4/manifest.csv'),('labels','results/labels_v13/source_labels_extracted.csv'),
                        ('days','results/recording_day_audit_v14/recording_days.csv'),('snapshot','results/demo_v13_release/feature_snapshot.npz')]:p.add_argument('--'+key,type=Path,default=Path(default))
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
