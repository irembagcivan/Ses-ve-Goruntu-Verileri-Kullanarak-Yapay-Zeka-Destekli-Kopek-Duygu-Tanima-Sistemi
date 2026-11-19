"""Training-only CV comparison of temporal CNN, GRU and bidirectional GRU."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from src.audio_analysis.yamnet import Yamnet,logmel_patches
from src.demo.train import emotion_rule
from src.audio_analysis.run import extract as decode_audio
from .train import seed,normalize_fit,normalize_apply,weights_for,losses,scores,TARGETS

class SequenceNet(nn.Module):
    def __init__(self,classes,kind):
        super().__init__();self.kind=kind
        self.visual=nn.Sequential(nn.Linear(117,32),nn.LayerNorm(32),nn.GELU())
        self.audio=nn.Sequential(nn.Linear(1024,32),nn.LayerNorm(32),nn.GELU())
        if kind=='cnn':self.temporal=nn.Sequential(nn.Conv1d(64,32,3,padding=1),nn.GELU(),nn.Conv1d(32,32,3,padding=1),nn.GELU())
        else:self.temporal=nn.GRU(64,32,batch_first=True,bidirectional=kind=='bigru')
        self.dropout=nn.Dropout(.35);self.heads=nn.ModuleList([nn.Linear(64 if kind=='bigru' else 32,size) for size in classes])
    def forward(self,x,lengths):
        features=torch.cat([self.visual(x[:,:,:117]),self.audio(x[:,:,117:])],-1)
        mask=torch.arange(x.shape[1])[None,:]<lengths[:,None]
        if self.kind=='cnn':
            hidden=features.transpose(1,2)*mask[:,None,:]
            for layer in self.temporal:hidden=layer(hidden)*mask[:,None,:]
            hidden=hidden.sum(-1)/lengths[:,None]
        else:
            _,hidden=self.temporal(pack_padded_sequence(features,lengths.cpu(),batch_first=True,enforce_sorted=False))
            hidden=torch.cat([hidden[-2],hidden[-1]],-1) if self.kind=='bigru' else hidden[-1]
        hidden=self.dropout(hidden)
        return [head(hidden) for head in self.heads]

def scale_fit(x,lengths,indices):
    return normalize_fit(np.concatenate([x[i,:lengths[i]] for i in indices]))
def scale_apply(x,lengths,stats):
    transformed=normalize_apply(x.reshape(-1,x.shape[-1]),stats).reshape(x.shape)
    for i,length in enumerate(lengths):transformed[i,length:]=0
    return transformed

def sequence_from_parts(records,embedding,length):
    seq=np.zeros((60,1141),np.float32)
    for second in range(length):
        vectors=[]
        for r in records:
            if not second<=r['timestamp_seconds']<second+1 or 'xy' not in r:continue
            xy=np.array(r['xy'],float);conf=np.array(r['confidence'],float);box=np.array(r['bbox_xyxy']);scale=np.linalg.norm(box[2:]-box[:2])
            if scale<=0:continue
            valid=np.isfinite(xy).all(1)&np.isfinite(conf)&(conf>=.3)&np.array(['antler' not in n for n in r['keypoint_names']])
            norm=(xy-(box[:2]+box[2:])/2)/scale;norm[~valid]=0
            vectors.append(np.concatenate([norm.reshape(-1),valid.astype(float)]))
        if vectors:seq[second,:117]=np.mean(vectors,axis=0)
        if embedding is not None:
            indices=[i for i in range(len(embedding)) if second<=i*.48+.4875<second+1]
            if indices:seq[second,117:]=embedding[indices].mean(0)
    return seq

def build_sequence(row,model):
    roots=[Path('results/demo_v1/features'),Path('results/demo_v4/new_features')]
    root=next(p for p in roots if (p/(row.source_sha256+'.pose.jsonl')).exists())
    records=[json.loads(line) for line in (root/(row.source_sha256+'.pose.jsonl')).read_text().splitlines()]
    wave=decode_audio(row.video_path);embedding=None
    length=min(60,max(1,int(np.ceil(len(wave)/16000)))) if wave is not None else min(60,max(1,int(np.ceil(records[-1]['timestamp_seconds']+.5))))
    if wave is not None:
        _,patches=logmel_patches(wave[:960000]);chunks=[]
        with torch.inference_mode():
            for i in range(0,len(patches),32):
                _,values=model(torch.from_numpy(patches[i:i+32]));chunks.append(values.numpy())
        embedding=np.concatenate(chunks)
    return sequence_from_parts(records,embedding,length),length

def train_model(x,lengths,labels,fit,val,sizes,kind,epochs,trace=True):
    seed();stats=scale_fit(x,lengths,fit);xt=scale_apply(x,lengths,stats);lt=torch.tensor(lengths);yt=torch.tensor(labels)
    model=SequenceNet(sizes,kind);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);weights=weights_for(labels[fit],sizes);history=[]
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad();loss=losses(model(xt[fit],lt[fit]),yt[fit],weights);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if trace and epoch in [10,20,40]:
            model.eval()
            with torch.inference_mode():f1=scores(model(xt[val],lt[val]),labels[val],sizes)
            history.append({'epoch':epoch,'train_loss':float(loss.detach()),'validation_mean_f1':float(np.mean(f1)),'validation_f1':f1})
    model.eval();return model,stats,history,xt,lt

def main(args):
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=False);report=json.loads(args.report.read_text());manifest=pd.read_csv(args.manifest).fillna('');yamnet=Yamnet(args.weights);sequences=[];lengths=[]
    for index,row in manifest.iterrows():
        seq,length=build_sequence(row,yamnet);sequences.append(seq);lengths.append(length);print(f'Sequence {index+1}/{len(manifest)}',flush=True)
    x=np.array(sequences);lengths=np.array(lengths);classes=report['deep_learning_experiment']['protocol']['classes'];sizes=[len(classes[t]) for t in TARGETS]
    labels=np.array([[classes[t].index(row[t]) if row[t] in classes[t] else -1 for t in TARGETS] for row in manifest.to_dict('records')]);train=np.flatnonzero(manifest.split.eq('train'));test=np.flatnonzero(manifest.split.eq('test'))
    assert not set(manifest.iloc[train].source_sha256)&set(manifest.iloc[test].source_sha256)
    folds=[(train[a],train[b]) for a,b in GroupKFold(3).split(train,groups=manifest.iloc[train].source_sha256)];choices={};histories={}
    np.savez_compressed(args.output/'sequence_snapshot.npz',sequences=x,lengths=lengths,labels=labels,train_indices=train,test_indices=test)
    for kind in ['cnn','gru','bigru']:
        histories[kind]=[]
        for index,(fit,val) in enumerate(folds):
            _,_,history,_,_=train_model(x,lengths,labels,fit,val,sizes,kind,40);histories[kind].append(history);print(f'CV {kind} fold {index+1}',flush=True)
        values={epoch:float(np.mean([next(h['validation_mean_f1'] for h in history if h['epoch']==epoch) for history in histories[kind]])) for epoch in [10,20,40]}
        choices[kind]={'epochs':min(values,key=lambda e:(-values[e],e)),'cv_mean_f1':values}
    (args.output/'selection.json').write_text(json.dumps(choices,indent=2))
    results=[]
    for kind in ['cnn','gru','bigru']:
        model,stats,_,xt,lt=train_model(x,lengths,labels,train,test,sizes,kind,choices[kind]['epochs'],False)
        with torch.inference_mode():outputs=model(xt,lt)
        metrics=[];mode='sequence_'+kind
        for index,target in enumerate(TARGETS):
            valid=test[labels[test,index]>=0];prediction=outputs[index].argmax(1).numpy();actual=labels[valid,index]
            metric={'target':target,'modality':mode,'macro_f1':float(f1_score(actual,prediction[valid],labels=list(range(sizes[index])),average='macro',zero_division=0)),'accuracy':float(accuracy_score(actual,prediction[valid])),'classes':classes[target],'confusion_matrix':confusion_matrix(actual,prediction[valid],labels=list(range(sizes[index]))).tolist(),'test_videos':len(valid),'train_videos':int((labels[train,index]>=0).sum())};metrics.append(metric)
            probabilities=torch.softmax(outputs[index],1).numpy()
            for sample,pred,score in zip(report['samples'],prediction,probabilities):
                sample['predictions'].setdefault(mode,{})[target]=classes[target][pred];sample['scores'].setdefault(mode,{})[target]=dict(zip(classes[target],map(float,score)))
        torch.save(model.state_dict(),args.output/(mode+'.pt'));(args.output/(mode+'.json')).write_text(json.dumps({'kind':kind,'classes':classes,'sizes':sizes,'scaler':stats,'epochs':choices[kind]['epochs']},ensure_ascii=False));results.append({'kind':kind,'epochs':choices[kind]['epochs'],'mean_test_macro_f1':float(np.mean([m['macro_f1'] for m in metrics])),'metrics':metrics,'parameters':sum(p.numel() for p in model.parameters())});report['metrics'].extend(metrics)
        print(results[-1],flush=True)
    for sample in report['samples']:
        for kind in ['cnn','gru','bigru']:
            mode='sequence_'+kind;sample['emotion_rule'][mode]=emotion_rule(sample['predictions'][mode],sample['scores'][mode])
    report['sequence_experiment']={'choices':choices,'results':results,'protocol':'1 Hz aligned sequences, first 60s, pose+confidence mask and 1024D YAMNet; train-only 3-fold CV epochs; no test selection; single seed; previously inspected small holdout'}
    (args.output/'experiment.json').write_text(json.dumps({**report['sequence_experiment'],'histories':histories},indent=2));(args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','report','weights','output']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
