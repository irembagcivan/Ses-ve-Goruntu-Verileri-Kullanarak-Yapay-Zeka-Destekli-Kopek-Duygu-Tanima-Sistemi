"""Learn perceived arousal/valence from pinned external human annotations.

The official source train/test split is preserved. Author-reported dog separation
cannot be independently audited because released CSVs contain no dog identities.
"""
import argparse,hashlib,json,math
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from src.audio_analysis.yamnet import Yamnet,logmel_patches
from src.deep_learning.train import normalize_fit,normalize_apply,seed

CLASSES={'valence':['Negative','Neutral','Positive'],'arousal':['Low','Medium','High']}
TARGETS=list(CLASSES)


class AffectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(2048,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.5),nn.Linear(64,32),nn.GELU(),nn.Dropout(.3))
        self.heads=nn.ModuleList([nn.Linear(32,3) for _ in TARGETS])
    def forward(self,x):
        h=self.shared(x);return [head(h) for head in self.heads]


def embedding_features(embedding):
    e=np.asarray(embedding,dtype=np.float32)
    if e.ndim!=2 or e.shape[1]!=1024 or not len(e) or not np.isfinite(e).all():raise ValueError('Invalid vocal embeddings')
    return np.concatenate([e.mean(0),e.std(0)]).astype(np.float32)


def metrics(p,y):
    result=[]
    for j,t in enumerate(TARGETS):
        predicted=p[j].argmax(1)
        result.append(dict(target=t,macro_f1=float(f1_score(y[:,j],predicted,labels=[0,1,2],average='macro',zero_division=0)),
                           accuracy=float(accuracy_score(y[:,j],predicted)),classes=CLASSES[t],support=len(y),
                           confusion_matrix=confusion_matrix(y[:,j],predicted,labels=[0,1,2]).tolist()))
    return result


def average(p,y):return float(np.mean([m['macro_f1'] for m in metrics(p,y)]))


def neural(x,y,fit,val,epochs,trace=False):
    seed();stats=normalize_fit(x[fit]);xt=normalize_apply(x,stats);yt=torch.tensor(y)
    model=AffectNet();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    weights=[torch.tensor(len(fit)/(3*np.maximum(1,np.bincount(y[fit,j],minlength=3))),dtype=torch.float32) for j in range(2)]
    history=[]
    for e in range(1,epochs+1):
        model.train();optimizer.zero_grad();outputs=model(xt[fit]);loss=torch.stack([nn.functional.cross_entropy(o,yt[fit,j],weight=weights[j]) for j,o in enumerate(outputs)]).mean()
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if trace and e in [20,40,80]:
            model.eval()
            with torch.inference_mode():p=[o.numpy() for o in model(xt[val])]
            history.append(dict(epoch=e,loss=float(loss.detach()),macro_f1=average(p,y[val])))
    model.eval()
    with torch.inference_mode():p=[torch.softmax(o,1).numpy() for o in model(xt)]
    return model,stats,p,history


def classical(x,y,fit,kind):
    models=[];outputs=[]
    for j in range(2):
        estimator=LogisticRegression(C=.1,class_weight='balanced',max_iter=2000) if kind=='lr' else RandomForestClassifier(n_estimators=200,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=2)
        model=make_pipeline(StandardScaler(),estimator);model.fit(x[fit],y[fit,j]);p=np.zeros((len(x),3));p[:,model.classes_.astype(int)]=model.predict_proba(x)
        models.append(model);outputs.append(p)
    return models,outputs


def features(data,weights,output):
    records=[];patches=[];counts=[]
    provenance=json.loads((data/'provenance.json').read_text());file_hashes={f['path']:f['sha256'] for f in provenance['files']}
    for breed in ['husky','shiba']:
        for split in ['train','test']:
            table=pd.read_csv(data/f'{breed}_{split}_labels.csv')
            if len(table)!=(600 if split=='train' else 100):raise ValueError('Unexpected label count')
            for r in table.itertuples():
                path=f'{breed}/{split}/{r.audio_id}.wav';source=data/path
                if hashlib.sha256(source.read_bytes()).hexdigest()!=file_hashes[path]:raise ValueError('Audio changed')
                rate,wave=wavfile.read(source)
                if wave.dtype.kind in 'iu':wave=wave.astype(np.float32)/float(2**(8*wave.dtype.itemsize-1))
                else:wave=wave.astype(np.float32)
                if wave.ndim==2:wave=wave.mean(1)
                wave=resample_poly(wave,16000//math.gcd(rate,16000),rate//math.gcd(rate,16000)).astype(np.float32)
                if not len(wave) or not np.isfinite(wave).all() or len(wave)>960000:raise ValueError('Invalid source audio')
                _,p=logmel_patches(wave);patches.append(p);counts.append(len(p))
                records.append(dict(audio_id=r.audio_id,breed=breed,split=split,sha256=file_hashes[path],valence=r.valence,arousal=r.arousal,seconds=len(wave)/16000))
    model=Yamnet(weights);all_patches=np.concatenate(patches);vectors=[]
    with torch.inference_mode():
        for i in range(0,len(all_patches),64):
            _,embedding=model(torch.from_numpy(all_patches[i:i+64]));vectors.append(embedding.numpy())
            if i%640==0:print(f'External YAMNet {i}/{len(all_patches)} patches',flush=True)
    embedding=np.concatenate(vectors);x=[];offset=0
    for count in counts:x.append(embedding_features(embedding[offset:offset+count]));offset+=count
    table=pd.DataFrame(records);table.to_csv(output/'manifest.csv',index=False)
    y=np.array([[CLASSES[t].index(row[t]) for t in TARGETS] for row in records]);x=np.array(x)
    np.savez_compressed(output/'features.npz',x=x,y=y)
    return table,x,y


def run(args):
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=False)
    if args.feature_cache:
        table=pd.read_csv(args.feature_cache/'manifest.csv')
        with np.load(args.feature_cache/'features.npz',allow_pickle=False) as saved:x,y=saved['x'].copy(),saved['y'].copy()
        expected=np.array([[CLASSES[t].index(row[t]) for t in TARGETS] for row in table.to_dict('records')])
        provenance=json.loads((args.data/'provenance.json').read_text());expected_hashes={r['sha256'] for r in provenance['files']}
        if not np.array_equal(y,expected) or x.shape!=(len(table),2048) or not np.isfinite(x).all() or not set(table.sha256)<=expected_hashes:raise ValueError('Invalid feature cache')
        table.to_csv(args.output/'manifest.csv',index=False);np.savez_compressed(args.output/'features.npz',x=x,y=y)
    else:table,x,y=features(args.data,args.weights,args.output)
    test=np.flatnonzero(table.split.eq('test'));train=np.flatnonzero(table.split.eq('train'))
    overlap=set(table.iloc[train].sha256)&set(table.iloc[test].sha256)
    # Source release has exact duplicates. Never allow an official-test audio
    # into training; inspect conflicting training labels without test labels.
    raw_train=train.copy();train=train[~table.iloc[train].sha256.isin(overlap).to_numpy()]
    conflicting=set(key for key,rows in table.iloc[train].groupby('sha256') if rows.valence.nunique()>1 or rows.arousal.nunique()>1)
    train=train[~table.iloc[train].sha256.isin(conflicting).to_numpy()]
    train=train[~table.iloc[train].sha256.duplicated().to_numpy()]
    audit=dict(raw_train=len(raw_train),clean_train=len(train),official_test=len(test),test_unique_audio=table.iloc[test].sha256.nunique(),
               overlap_hashes=sorted(overlap),quarantined_train_audio_ids=table.iloc[sorted(set(raw_train)-set(train))].audio_id.tolist(),
               conflicting_training_hashes=sorted(conflicting),duration_min=float(table.seconds.min()),duration_max=float(table.seconds.max()),
               explanation='Training excludes test-matching hashes, contradictory training hash labels and repeated training hashes. Official 200 test rows retained; two test rows share one hash.')
    if set(table.iloc[train].sha256)&set(table.iloc[test].sha256):raise ValueError('Leakage after quarantine')
    (args.output/'data_audit.json').write_text(json.dumps(audit,indent=2))
    print('External data audit',json.dumps(audit),flush=True)
    strata=table.iloc[train].breed+'_'+table.iloc[train].valence+'_'+table.iloc[train].arousal
    folds=[(train[f],train[g]) for f,g in StratifiedGroupKFold(3,shuffle=True,random_state=42).split(train,strata,groups=table.iloc[train].sha256)]
    traces=[]
    for i,(f,g) in enumerate(folds):
        _,_,_,trace=neural(x,y,f,g,80,True);traces.append(trace);print('Affect neural CV',i+1,flush=True)
    values={e:float(np.mean([next(v['macro_f1'] for v in h if v['epoch']==e) for h in traces])) for e in [20,40,80]}
    epoch=min(values,key=lambda e:(-values[e],e));oof={name:[np.zeros((len(y),3)) for _ in range(2)] for name in ['lr','rf','neural']}
    for f,g in folds:
        for name in ['lr','rf','neural']:
            p=neural(x,y,f,None,epoch)[2] if name=='neural' else classical(x,y,f,name)[1]
            for j in range(2):oof[name][j][g]=p[j][g]
    oof['soft_vote']=[(oof['lr'][j]+oof['neural'][j])/2 for j in range(2)]
    cv={name:average([p[train] for p in outputs],y[train]) for name,outputs in oof.items()}
    selected=max(cv,key=cv.get)
    if args.weights.exists():encoder_sha=hashlib.sha256(args.weights.read_bytes()).hexdigest()
    elif args.feature_cache and (args.feature_cache/'model.json').exists():encoder_sha=json.loads((args.feature_cache/'model.json').read_text())['yamnet_sha256']
    else:raise ValueError('Need encoder weights or trusted cached encoder fingerprint')
    if args.feature_cache and (args.feature_cache/'model.json').exists() and encoder_sha!=json.loads((args.feature_cache/'model.json').read_text())['yamnet_sha256']:raise ValueError('Cached encoder fingerprint mismatch')
    meta=dict(version='external-affect-v15',source_commit=json.loads((args.data/'provenance.json').read_text())['commit'],
              classes=CLASSES,targets=TARGETS,selected=selected,cv_macro_f1=cv,epochs=epoch,epoch_cv=values,
              yamnet_sha256=encoder_sha,
              features='Mean and std of frozen YAMNet 1024D embeddings; 2048D',
              source='External human-perceived valence/arousal labels, EmotionalCanines; never local rule-derived emotions',
              data_audit=audit,
              limits=['Only Husky/Shiba bark clips; source annotations are human perception, not verified internal mood.',
                      'Official test retained; contaminated/conflicting/duplicate training audio quarantined. Authors report dog-separated test; released IDs do not allow independent identity audit.',
                      'Official test has 200 rows but only 199 unique audio hashes; test labels include repeated audio.',
                      'Inner CV groups exact audio hashes, not dogs; validation may be optimistic.',
                      'Our 127-video dataset has no independent affect labels: target-domain accuracy is unknown.',
                      'No synchronized visual emotion labels: this head is audio-only, not trained multimodal emotion fusion.'])
    (args.output/'selection.json').write_text(json.dumps(meta,indent=2))
    # Selection is frozen before official held-out metrics.
    network,stats,p,_=neural(x,y,train,None,epoch);torch.save(network.state_dict(),args.output/'neural.pt');meta['scaler']=stats
    final={'neural':p};models={}
    for name in ['lr','rf']:models[name],final[name]=classical(x,y,train,name)
    final['soft_vote']=[(final['lr'][j]+final['neural'][j])/2 for j in range(2)]
    joblib.dump(models,args.output/'classical.joblib');(args.output/'model.json').write_text(json.dumps(meta,indent=2))
    results=[]
    for name,p in final.items():
        results.append(dict(model=name,mean_test_macro_f1=average([v[test] for v in p],y[test]),metrics=metrics([v[test] for v in p],y[test]),
                            joint_accuracy=float((np.stack([v[test].argmax(1) for v in p],1)==y[test]).all(1).mean()),
                            per_breed={b:metrics([v[test[table.iloc[test].breed.eq(b)]] for v in p],y[test[table.iloc[test].breed.eq(b)]]) for b in ['husky','shiba']}))
    result=dict(protocol={k:v for k,v in meta.items() if k!='scaler'},train_clips=len(train),test_clips=len(test),results=results,
                test_predictions={name:{t:outputs[j][test].argmax(1).tolist() for j,t in enumerate(TARGETS)} for name,outputs in final.items()},
                test_audio_ids=table.iloc[test].audio_id.tolist())
    (args.output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps([{k:v for k,v in m.items() if k not in ['metrics','per_breed']} for m in results]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('results/affect_v15_external/data'));p.add_argument('--weights',type=Path,default=Path('models/yamnet.h5'));p.add_argument('--output',type=Path,required=True);p.add_argument('--feature-cache',type=Path)
    run(p.parse_args())
