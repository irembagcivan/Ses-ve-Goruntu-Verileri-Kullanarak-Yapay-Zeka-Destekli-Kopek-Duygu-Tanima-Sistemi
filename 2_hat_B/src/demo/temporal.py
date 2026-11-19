"""Experimental window behavior inference; never evaluate against video labels."""
import argparse
import json
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
from src.audio_analysis.behavior import event_features
from src.audio_analysis.run import extract as decode_audio
from src.demo.extract import aggregate, audio_features_from_wave, clean_json, digest


def intervals(duration,width=4.,hop=2.):
    if not np.isfinite(duration) or duration<=0 or width<=0 or hop<=0:raise ValueError('Invalid window parameters')
    start=0.
    while start<duration:
        end=min(duration,start+width)
        if end-start<1.:break
        yield start,end
        if end==duration:break
        start+=hop


def visual_window(records,start,end,image_width,image_height,n=39,confidence=.3):
    selected=[r for r in records if start<=r['timestamp_seconds']<end]
    poses=[];movements=[];previous=None;detections=multi=0
    for r in selected:
        if 'xy' not in r or not r['dog_count']:
            previous=None;continue
        detections+=1;multi+=int(r['dog_count']>1)
        xy=np.asarray(r['xy'],float);scores=np.asarray(r['confidence'],float)
        box=np.asarray(r['bbox_xyxy'],float);scale=np.linalg.norm(box[2:]-box[:2])
        if scale<=0:previous=None;continue
        valid=np.isfinite(xy).all(1)&np.isfinite(scores)&(scores>=confidence)
        valid&=(xy[:,0]>=0)&(xy[:,0]<image_width)&(xy[:,1]>=0)&(xy[:,1]<image_height)
        valid&=np.array(['antler' not in name for name in r['keypoint_names']])
        normalized=(xy-(box[:2]+box[2:])/2)/scale;normalized[~valid]=np.nan
        poses.append(normalized.reshape(-1));time=r['timestamp_seconds']
        if previous is not None and r['dog_count']==1:
            prior,prior_time=previous
            if time>prior_time:movements.append(np.linalg.norm((normalized-prior)/(time-prior_time),axis=1))
        previous=(normalized,time) if r['dog_count']==1 else None
    count=len(selected)
    features={'v_detection_rate':detections/count if count else 0.,'v_multi_rate':multi/count if count else 0.,
              'v_pose_available':int(any(np.isfinite(pose).any() for pose in poses))}
    features.update(aggregate(poses if poses else np.full((1,n*2),np.nan),'v_pose'))
    features.update(aggregate(movements if movements else np.full((1,n),np.nan),'v_speed'))
    return features,{'sampled_frames':count,'detected_frames':detections,'multi_dog_frames':multi,
                     'subject_scope':'largest_box_per_frame; no guaranteed identity' if multi else 'video_window; identity not verified'}


def predict_window(entries,features):
    predictions={};scores={};unavailable={}
    for target,entry in entries.items():
        columns=entry['columns']
        if any(c.startswith('v_') for c in columns) and not features.get('v_pose_available'):
            unavailable[target]='pose_missing';continue
        if any(c.startswith('a_') for c in columns) and not features.get('a_available'):
            unavailable[target]='audio_missing';continue
        if any(c.startswith('y_') for c in columns) and not features.get('y_available'):
            unavailable[target]='audio_events_missing';continue
        missing=set(columns)-set(features)
        if missing:raise ValueError('Missing feature schema columns: '+','.join(sorted(missing)))
        frame=pd.DataFrame([{c:features[c] for c in columns}])
        model=entry['model'];predictions[target]=str(model.predict(frame)[0])
        scores[target]=dict(zip(model.classes_,map(float,model.predict_proba(frame)[0])))
    return predictions,scores,unavailable


def run(args):
    report=json.loads(args.report.read_text());manifest=pd.read_csv(args.manifest).set_index('video_id')
    models=joblib.load(args.model)['models']['cv_selected']
    if args.output.exists():raise FileExistsError('Choose a new output directory')
    args.output.mkdir(parents=True)
    for index,sample in enumerate(report['samples'],1):
        row=manifest.loc[sample['video_id']];key=sample['source_sha256']
        if row.source_sha256!=key:raise ValueError('Manifest/report content mismatch')
        root=next((p for p in [args.old_features,args.new_features] if (p/(key+'.pose.jsonl')).exists()),None)
        if root is None:raise ValueError('Cached pose sequence missing: '+sample['video_id'])
        metadata=json.loads((root/(key+'.json')).read_text())
        if metadata['source_sha256']!=key:raise ValueError('Cached source mismatch')
        records=[json.loads(line) for line in (root/(key+'.pose.jsonl')).read_text().splitlines()]
        cap=cv2.VideoCapture(str(row.video_path))
        try:
            fps=cap.get(cv2.CAP_PROP_FPS);frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width=cap.get(cv2.CAP_PROP_FRAME_WIDTH);height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if not cap.isOpened() or fps<=0:raise ValueError('Video metadata unavailable')
            duration=frames/fps
        finally:cap.release()
        limit=min(duration,float(metadata['signature']['seconds']))
        wave=decode_audio(row.video_path);windows=[]
        for start,end in intervals(limit):
            visual,quality=visual_window(records,start,end,width,height,confidence=metadata['signature']['confidence'])
            audio,_=audio_features_from_wave(wave[int(start*16000):int(end*16000)] if wave is not None else np.array([]))
            analysis=sample['audio_analysis'];event_windows=[w for w in analysis.get('windows',[]) if w['start']>=start and w['end']<=end]
            events=event_features({'status':analysis['status'],'windows':event_windows},seconds=end)
            predicted,scores,unavailable=predict_window(models,{**visual,**audio,**events})
            windows.append({'start':round(start,3),'end':round(end,3),'predictions':predicted,'scores':scores,
                            'unavailable':unavailable,'quality':quality})
        sample['temporal_behavior']={'window_seconds':4,'hop_seconds':2,'analyzed_until':round(limit,3),
                                     'source_duration':round(duration,3),'windows':windows,
                                     'experimental':True,'accuracy_measured':False}
        (args.output/(key+'.windows.json')).write_text(json.dumps(clean_json(sample['temporal_behavior']),ensure_ascii=False,allow_nan=False))
        print(f'{index}/{len(report["samples"])} {sample["video_id"]}: {len(windows)} windows',flush=True)
    report['temporal_protocol']={'models':'Frozen training-CV-selected video-level models, applied to 4s windows every 2s',
        'pose':'Cached 2fps largest-box pose features; first 60 seconds only. No per-dog identity guarantee.',
        'validation':'No temporal labels; window behavior accuracy is unmeasured. Short windows differ from training distribution.',
        'audio':'Complete YAMNet patches contained within each behavior window; mixed recording.',
        'model_sha256':digest(args.model)}
    report['limitations'].append('4 saniyelik pencere davranışları deneyseldir; model video etiketleriyle eğitilmiştir. Zaman bazlı doğruluk ölçülmedi. İlk 60 saniye incelenir; çoklu köpekte birey eşlemesi yoktur.')
    (args.output/'report.json').write_text(json.dumps(clean_json(report),ensure_ascii=False,allow_nan=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','report','model','old-features','new-features','output']:parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
