"""Cache full-recording YAMNet event candidates and attach them to a demo report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
import imageio_ffmpeg
from .yamnet import Yamnet

CLASSES = {'Dog':'Köpek sesi','Bark':'Havlama','Growling':'Hırlama',
           'Whimper (dog)':'İnleme','Howl':'Ulama','Speech':'Konuşma','Music':'Müzik'}


def summarize(wave, scores, names, threshold=.3):
    duration=len(wave)/16000
    selected={name:names.index(name) for name in CLASSES if name in names}
    windows=[]
    for index,row in enumerate(scores):
        start=index*.48
        if start>=duration:break
        windows.append({'start':round(start,3),'end':round(min(duration,start+.975),3),
                        'scores':{name:round(float(row[column]),4) for name,column in selected.items()}})
    events=[]
    for name in selected:
        current=None
        for window in windows:
            score=window['scores'][name]
            if score>=threshold:
                if current is not None and window['start']<=current['end']:
                    current['end']=window['end'];current['score']=max(current['score'],score)
                else:
                    current={'class':name,'label':CLASSES[name],'start':window['start'],
                             'end':window['end'],'score':score};events.append(current)
            else:current=None
    bins=np.array_split(np.abs(wave),min(600,len(wave)))
    envelope=[round(float(np.max(chunk)),4) for chunk in bins]
    return {'status':'ok','duration':round(duration,3),'threshold':threshold,
            'windows':windows,'events':sorted(events,key=lambda e:(e['start'],e['class'])),
            'envelope':envelope,'top_classes':[{'class':names[i],'score':round(float(scores[:,i].mean()),4)}
                                             for i in scores.mean(0).argsort()[-5:][::-1]]}


def extract(path):
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),
                           '-vn','-ac','1','-ar','16000','-f','f32le','pipe:1'],capture_output=True,timeout=120)
    if result.returncode or not result.stdout:
        return None
    return np.frombuffer(result.stdout,dtype='<f4').copy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=Path('models/yamnet.h5'))
    args=parser.parse_args();torch.set_num_threads(2)
    model=Yamnet(args.weights)
    names=[r['display_name'] for r in csv.DictReader(open('configs/audio/yamnet_class_map.csv'))]
    fingerprint=hashlib.sha256(args.weights.read_bytes()+Path(__file__).read_bytes()+Path(__file__).with_name('yamnet.py').read_bytes()).hexdigest()
    cache=args.output.parent/'audio_cache';cache.mkdir(parents=True,exist_ok=True)
    analyses={}
    rows=list(csv.DictReader(args.manifest.open()))
    for index,row in enumerate(rows,1):
        target=cache/(row['source_sha256']+'.json')
        analysis=json.loads(target.read_text()) if target.exists() else {}
        if analysis.get('fingerprint')!=fingerprint:
            wave=extract(row['video_path'])
            analysis={'status':'no_audio'} if wave is None else summarize(wave,model.score(wave),names)
            analysis['fingerprint']=fingerprint
            target.write_text(json.dumps(analysis,ensure_ascii=False))
        analyses[row['video_id']]=analysis
        print(f"{index}/{len(rows)} {row['video_id']}: {analysis['status']}, {len(analysis.get('events',[]))} candidates",flush=True)
    report=json.loads(args.report.read_text())
    for sample in report['samples']:sample['audio_analysis']=analyses[sample['video_id']]
    report['audio_event_model']={'name':'YAMNet','implementation':'Inference-only PyTorch port of official HDF5',
        'weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        'window_seconds':.975,'hop_seconds':.48,'candidate_threshold':.3,
        'limits':'Mixed recording; no source separation or attribution to individual dogs. Scores are not calibrated probabilities. No temporal ground-truth evaluation.'}
    args.output.write_text(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
