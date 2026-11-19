"""Train/evaluate neural pose/audio ablations using training-only epoch selection."""
import argparse,csv,hashlib,json,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from src.audio_analysis.yamnet import Yamnet,logmel_patches
from src.audio_analysis.run import extract as decode_audio
from src.audio_analysis.behavior import aligned_table
from src.demo.train import TARGETS,emotion_rule

CHECKPOINTS=[20,40,60,80]


def model_targets(metadata):
    """Use checkpoint task order; legacy checkpoints retain the five old tasks."""
    return metadata.get('targets', metadata.get('protocol', {}).get('targets', TARGETS))


def seed(value=42):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)


class FusionNet(nn.Module):
    def __init__(self,visual_dim,audio_dim,class_counts,modality='fusion'):
        super().__init__();self.modality=modality
        def branch(size):return nn.Sequential(nn.Linear(size,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.35),nn.Linear(64,32),nn.GELU())
        self.visual=branch(visual_dim);self.audio=branch(audio_dim)
        self.shared=nn.Sequential(nn.Linear(64 if modality=='fusion' else 32,32),nn.GELU(),nn.Dropout(.35))
        self.heads=nn.ModuleList([nn.Linear(32,size) for size in class_counts])
        inactive=self.audio if modality=='visual' else self.visual if modality=='audio' else None
        if inactive is not None:
            for parameter in inactive.parameters():parameter.requires_grad_(False)
    def forward(self,visual,audio):
        if self.modality=='fusion':hidden=torch.cat([self.visual(visual),self.audio(audio)],dim=1)
        elif self.modality=='visual':hidden=self.visual(visual)
        else:hidden=self.audio(audio)
        return [head(self.shared(hidden)) for head in self.heads]


def normalize_fit(array):
    array=np.asarray(array,float);median=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in array.T])
    clean=np.where(np.isfinite(array),array,median);mean=clean.mean(0);scale=clean.std(0);scale[scale<1e-6]=1.
    return {'median':median.tolist(),'mean':mean.tolist(),'scale':scale.tolist()}


def normalize_apply(array,stats):
    array=np.asarray(array,float);median=np.array(stats['median']);clean=np.where(np.isfinite(array),array,median)
    return torch.tensor(np.clip((clean-np.array(stats['mean']))/np.array(stats['scale']),-10,10),dtype=torch.float32)


def losses(outputs,labels,weights):
    terms=[]
    for index,output in enumerate(outputs):
        valid=labels[:,index]>=0
        if valid.any():terms.append(nn.functional.cross_entropy(output[valid],labels[valid,index],weight=weights[index]))
    if not terms:raise ValueError('No labeled examples')
    return torch.stack(terms).mean()


def weights_for(labels,sizes):
    return [torch.tensor(len(labels)/(size*np.maximum(1,np.bincount(labels[:,index][labels[:,index]>=0],minlength=size))),dtype=torch.float32) for index,size in enumerate(sizes)]


def scores(outputs,labels,sizes):
    values=[]
    for index,output in enumerate(outputs):
        valid=labels[:,index]>=0;predicted=output.argmax(1).numpy()
        values.append(float(f1_score(labels[valid,index],predicted[valid],labels=list(range(sizes[index])),average='macro',zero_division=0)))
    return values


def train_one(v,a,labels,fit_indices,validation_indices,sizes,modality,epochs,collect=True):
    seed();vstats=normalize_fit(v[fit_indices]);astats=normalize_fit(a[fit_indices])
    vt=normalize_apply(v,vstats);at=normalize_apply(a,astats);yt=torch.tensor(labels,dtype=torch.long)
    model=FusionNet(v.shape[1],a.shape[1],sizes,modality)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    weights=weights_for(labels[fit_indices],sizes);history=[]
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad();output=model(vt[fit_indices],at[fit_indices]);loss=losses(output,yt[fit_indices],weights)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if collect and epoch%10==0:
            model.eval()
            with torch.inference_mode():
                validation=model(vt[validation_indices],at[validation_indices]);f1=scores(validation,labels[validation_indices],sizes)
                history.append({'epoch':epoch,'train_loss':float(loss.detach()),'validation_loss':float(losses(validation,yt[validation_indices],weights)),
                                'validation_f1':f1,'validation_mean_f1':float(np.mean(f1))})
    model.eval()
    return model,vstats,astats,history,vt,at


def embeddings(manifest,weights,cache):
    cache.mkdir(parents=True,exist_ok=True);model=Yamnet(weights);fingerprint=hashlib.sha256(weights.read_bytes()).hexdigest();rows=[]
    for index,row in manifest.iterrows():
        path=cache/(row.source_sha256+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                if str(saved['weights_sha256'])!=fingerprint:raise ValueError('Embedding checkpoint mismatch')
                vector=saved['embedding']
        else:
            wave=decode_audio(row.video_path)
            if wave is None:vector=np.full(1024,np.nan,np.float32)
            else:
                _,patches=logmel_patches(wave[:960000]);vectors=[]
                with torch.inference_mode():
                    for start in range(0,len(patches),32):
                        _,embedding=model(torch.from_numpy(patches[start:start+32]));vectors.append(embedding.numpy())
                vector=np.concatenate(vectors).mean(0)
            np.savez_compressed(path,embedding=vector,weights_sha256=fingerprint)
        rows.append(vector)
        print(f'Embedding {index+1}/{len(manifest)}: {row.video_id}',flush=True)
    return np.asarray(rows)


def main(args):
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=False)
    report=json.loads(args.report.read_text());manifest=pd.read_csv(args.manifest).fillna('')
    features=pd.concat([pd.read_csv(args.old_features),pd.read_csv(args.new_features)],ignore_index=True)
    table=aligned_table(manifest,features,report)
    embedding=embeddings(manifest,args.weights,args.output/'embedding_cache')
    visual_columns=[c for c in table if c.startswith('v_')];event_columns=[c for c in table if c.startswith('y_')]
    v=table[visual_columns].to_numpy(float);a=np.concatenate([embedding,table[event_columns].to_numpy(float)],axis=1)
    train=np.flatnonzero(table.split.eq('train'));test=np.flatnonzero(table.split.eq('test'))
    assert not set(table.iloc[train].source_sha256)&set(table.iloc[test].source_sha256)
    classes={target:sorted(set(table.iloc[train][target])-{'','gorunmuyor','etiket_yok'}) for target in TARGETS}
    sizes=[len(classes[t]) for t in TARGETS];labels=np.array([[classes[t].index(row[t]) if row[t] in classes[t] else -1 for t in TARGETS] for row in table.to_dict('records')])
    folds=[(train[fit],train[val]) for fit,val in GroupKFold(3).split(train,groups=table.iloc[train].source_sha256)]
    np.savez_compressed(args.output/'feature_snapshot.npz',visual=v,audio=a,labels=labels,train_indices=train,test_indices=test,video_ids=table.video_id.to_numpy(str),source_sha256=table.source_sha256.to_numpy(str))
    histories={};choices={}
    # Training CV only: freeze epoch choices for all modalities before any test evaluation.
    for modality in ['visual','audio','fusion']:
        history=[]
        for fold,(fit,val) in enumerate(folds):
            _,_,_,trace,_,_=train_one(v,a,labels,fit,val,sizes,modality,80)
            history.append(trace);print(f'CV {modality} fold {fold+1} complete',flush=True)
        histories[modality]=history
        candidates={epoch:float(np.mean([next(e['validation_mean_f1'] for e in trace if e['epoch']==epoch) for trace in history])) for epoch in CHECKPOINTS}
        choices[modality]={'epochs':min(candidates,key=lambda e:(-candidates[e],e)),'cv_epoch_mean_f1':candidates}
    protocol={'epoch_choices':choices,'epochs_considered':CHECKPOINTS,'seed':42,'optimizer':'AdamW lr=.001 weight_decay=.01',
        'dropout':.35,'gradient_clip':1.,'loss':'Equal-task mean of class-weighted cross entropy; invisible labels masked',
        'encoders':'YOLO/RTMPose and YAMNet frozen; only FusionNet is fitted',
        'input':'Normalized pose/movement aggregates; 1024 mean YAMNet embeddings + event summaries, first 60 seconds',
        'split':'72 training / 25 previously inspected test; 30 incomplete never fit',
        'classes':classes,'visual_columns':visual_columns,'event_columns':event_columns,
        'folds':[{'fit':table.iloc[f].video_id.tolist(),'validation':table.iloc[val].video_id.tolist()} for f,val in folds]}
    source_signature=json.loads((args.old_features.parent/(table.iloc[train[0]].source_sha256+'.json')).read_text())['signature']
    protocol.update({'visual_feature_extraction_signature':source_signature,'visual_feature_detector':'YOLOv8n cached training features; gallery previews separately use YOLOv8s/m','yamnet_weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest()})
    (args.output/'selection.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    results=[]
    for modality in ['visual','audio','fusion']:
        model,vs,ast,_,vt,at=train_one(v,a,labels,train,test,sizes,modality,choices[modality]['epochs'],False)
        with torch.inference_mode():outputs=model(vt,at)
        mode='neural_'+modality;metrics=[]
        for index,target in enumerate(TARGETS):
            valid=test[labels[test,index]>=0];predicted=outputs[index].argmax(1).numpy();actual=labels[valid,index]
            metric={'modality':mode,'target':target,'accuracy':float(accuracy_score(actual,predicted[valid])),
                'macro_f1':float(f1_score(actual,predicted[valid],labels=list(range(sizes[index])),average='macro',zero_division=0)),
                'classes':classes[target],'confusion_matrix':confusion_matrix(actual,predicted[valid],labels=list(range(sizes[index]))).tolist(),
                'train_videos':int((labels[train,index]>=0).sum()),'test_videos':len(valid)}
            metrics.append(metric)
            probabilities=torch.softmax(outputs[index],dim=1).numpy()
            for sample,pred,s in zip(report['samples'],predicted,probabilities):
                sample['predictions'].setdefault(mode,{})[target]=classes[target][pred]
                sample['scores'].setdefault(mode,{})[target]=dict(zip(classes[target],map(float,s)))
        for sample in report['samples']:sample['emotion_rule'][mode]=emotion_rule(sample['predictions'][mode],sample['scores'][mode])
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
        torch.save(model.state_dict(),args.output/(mode+'.pt'))
        (args.output/(mode+'.json')).write_text(json.dumps({'protocol':protocol,'visual_scaler':vs,'audio_scaler':ast,'modality':modality,
                'visual_dim':v.shape[1],'audio_dim':a.shape[1],'class_counts':sizes,'trainable_parameters':parameters},ensure_ascii=False))
        results.append({'modality':mode,'mean_test_macro_f1':float(np.mean([m['macro_f1'] for m in metrics])),
                        'epochs':choices[modality]['epochs'],'trainable_parameters':parameters,'metrics':metrics})
        report['metrics'].extend(metrics);print(mode,results[-1]['mean_test_macro_f1'],flush=True)
    experiment={'protocol':protocol,'history':histories,'results':results,
        'limits':['Frozen pretrained encoders, not end-to-end fine-tuning from pixels/waveforms.',
                  'Small previously inspected holdout; one neural training seed. No emotion ground truth.',
                  'Neural video-level models are not used for the existing temporal behavior windows.']}
    report['deep_learning_experiment']={k:v for k,v in experiment.items() if k!='history'}
    report['limitations'].append('Yeni sinir ağları donmuş ön-eğitimli özellikler üzerinde eğitildi. Tek başlangıç tohumu ve küçük, önceden incelenmiş test kümesi kullanıldı; daha iyi sonuç garantisi yoktur.')
    (args.output/'experiment.json').write_text(json.dumps(experiment,ensure_ascii=False,indent=2))
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','report','old-features','new-features','weights','output']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
