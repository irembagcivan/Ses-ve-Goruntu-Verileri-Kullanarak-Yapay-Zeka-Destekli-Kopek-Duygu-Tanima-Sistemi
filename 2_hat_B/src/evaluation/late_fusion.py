"""Training-selected decision fusion; nested camera-day evaluation is kept separate."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.model_selection import GroupKFold
from src.deep_learning.final_labels import neural_fit,classical_fit,mean_f1,metric,TARGETS,CLASSES
from .day_validation import day_folds

RECIPES={
 'classical_fusion':{'cf':1.},'neural_fusion':{'nf':1.},
 'cf75_nf25':{'cf':.75,'nf':.25},'cf50_nf50':{'cf':.5,'nf':.5},'cf25_nf75':{'cf':.25,'nf':.75},
 'visual75_audio25':{'v':.75,'a':.25},'visual50_audio50':{'v':.5,'a':.5},'visual25_audio75':{'v':.25,'a':.75},
 'four_way':{'cf':.25,'nf':.25,'v':.25,'a':.25}}


def blend(outputs,weights):
    if not weights or any(not np.isfinite(w) or w<0 for w in weights.values()) or sum(weights.values())<=0:raise ValueError('Invalid fusion weights')
    if set(weights)-set(outputs):raise ValueError('Missing fusion member')
    n=len(next(iter(outputs.values())))
    result=[]
    for j in range(n):
        values=[np.asarray(outputs[k][j],float) for k in weights]
        if any(v.shape!=values[0].shape or not np.isfinite(v).all() or (v<0).any() for v in values):raise ValueError('Invalid member probabilities')
        if any(not np.allclose(v.sum(1),1,atol=1e-5) for v in values):raise ValueError('Member probabilities do not sum to one')
        result.append(sum(weights[k]*np.asarray(outputs[k][j]) for k in weights)/sum(weights.values()))
    return result


def members(v,a,x,lengths,y,fit,epochs,classifiers):
    p={'nf':neural_fit(v,a,x,lengths,y,fit,None,'neural_fusion',epochs)[3]}
    for key,data in [('v',v),('a',a),('cf',np.concatenate([v,a],1))]:p[key]=classical_fit(data,y,fit,np.arange(len(y)),classifiers[key])[0]
    return p


def choose(v,a,x,lengths,y,fit,folds,epochs,classifiers):
    oof={k:[np.zeros((len(y),len(CLASSES[t]))) for t in TARGETS] for k in ['nf','cf','v','a']}
    for f,g in folds:
        p=members(v,a,x,lengths,y,f,epochs,classifiers)
        for key in p:
            for j in range(len(TARGETS)):oof[key][j][g]=p[key][j][g]
    inner={key:[p[fit] for p in values] for key,values in oof.items()}
    scores={recipe:mean_f1(blend(inner,weights),y[fit]) for recipe,weights in RECIPES.items()}
    selected=max(scores,key=scores.get)
    return dict(recipe=selected,weights=RECIPES[selected],cv_macro_f1=scores)


def run(args):
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=False)
    with np.load('results/demo_v13_release/feature_snapshot.npz',allow_pickle=False) as data:
        v,a,x,lengths,y=[data[k].copy() for k in ['visual','audio','sequences','lengths','labels']];train=data['train_indices'];test=data['test_indices'];hashes=data['source_sha256']
    old=json.loads(Path('results/day_validation_v14/report.json').read_text());meta=pd.read_csv('results/recording_day_audit_v14/recording_days.csv').fillna('')
    manifest=pd.read_csv('results/demo_v4/manifest.csv');meta=meta.set_index('video_id').loc[manifest.video_id].reset_index()
    eligible,groups,folds=day_folds(meta,y,hashes);all_oof=[np.zeros((len(y),len(CLASSES[t]))) for t in TARGETS];records=[]
    for i,(fit,held,inner) in enumerate(folds):
        previous=old['folds'][i]
        if previous['held_out_day']!=groups[held[0]]:raise ValueError('Fold alignment mismatch')
        epochs=previous['selection']['neural_fusion']['epochs']
        classifiers={k:previous['selection']['classical_'+name]['classifier'] for k,name in [('v','visual'),('a','audio'),('cf','fusion')]}
        choice=choose(v,a,x,lengths,y,fit,inner,epochs,classifiers)
        (args.output/f'fold_{i+1}_selection.json').write_text(json.dumps(choice,indent=2))
        p=blend(members(v,a,x,lengths,y,fit,epochs,classifiers),choice['weights'])
        for j in range(len(TARGETS)):all_oof[j][held]=p[j][held]
        records.append(dict(day=groups[held[0]],choice=choice,macro_f1=mean_f1([p[j][held] for j in range(len(TARGETS))],y[held]),metrics=metric([p[j][held] for j in range(len(TARGETS))],y[held])))
        print('Late fusion day',groups[held[0]],choice['recipe'],records[-1]['macro_f1'],flush=True)
    result=dict(protocol='Global recipe selected inside each training-day set; no outer-day recipe tuning. Same 103 camera-day examples as v14; retrospective, one seed.',
                mean_macro_f1=mean_f1([p[eligible] for p in all_oof],y[eligible]),metrics=metric([p[eligible] for p in all_oof],y[eligible]),folds=records,
                predictions={t:all_oof[j][eligible].argmax(1).tolist() for j,t in enumerate(TARGETS)},video_ids=manifest.iloc[eligible].video_id.tolist())
    (args.output/'report.json').write_text(json.dumps(result,indent=2));print('Nested late fusion',result['mean_macro_f1'],flush=True)
    deployment=json.loads(Path('results/demo_v13_release/selection.json').read_text())
    epochs=deployment['models']['neural_fusion']['epochs'];classifiers={k:deployment['models']['classical_'+name]['classifier'] for k,name in [('v','visual'),('a','audio'),('cf','fusion')]}
    inner=[(train[f],train[g]) for f,g in GroupKFold(3).split(train,groups=hashes[train])]
    choice=choose(v,a,x,lengths,y,train,inner,epochs,classifiers)
    choice.update(targets=TARGETS,classes=CLASSES,selection_scope='Only original 93 training videos; no test-score recipe selection',
                  components={'nf':'FusionNet','cf':'Klasik · birleşik','v':'Klasik · görüntü','a':'Klasik · ses'},
                  model_hashes={name:hashlib.sha256(Path('results/demo_v13_release',name).read_bytes()).hexdigest() for name in ['neural_fusion.pt','classical.joblib']})
    (args.output/'deployment.json').write_text(json.dumps(choice,ensure_ascii=False,indent=2));print('Deployment',choice['recipe'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args())
