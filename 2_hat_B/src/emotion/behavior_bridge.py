"""Four-class emotion estimate trained on human-annotated emotion labels (two annotators + adjudication)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[2] / 'configs/demo/four_emotions.json'
NAMES = {'Happy': 'Mutlu', 'Sad': 'Üzgün', 'Angry': 'Öfkeli', 'Relaxed': 'Rahat'}


EVENTS = ['Dog', 'Bark', 'Growling', 'Whimper (dog)', 'Howl', 'Speech', 'Music']


def audio_features(analysis):
    values=[]
    windows=analysis.get('windows', []) if analysis else []
    for name in EVENTS:
        raw=[float(w.get('scores',{}).get(name,0)) for w in windows]
        if not np.isfinite(raw).all() or any(v<0 or v>1 for v in raw):
            raise ValueError('Invalid event scores')
        values.extend([float(np.mean(raw)) if raw else 0.,
                       float(np.mean(sorted(raw, reverse=True)[:3])) if raw else 0.,
                       float(np.mean(np.asarray(raw)>=.3)) if raw else 0.])
    return values+[float(bool(windows))]


def train(report_path, labels_path, output=MODEL):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score
    report=json.loads(Path(report_path).read_text())
    source={r['video'].casefold():r for r in csv.DictReader(open(labels_path))}
    labels={k:r['label'] for k,r in source.items()}
    targets=['kuyruk_konum','kuyruk_hareket','kulak','durus','hareket']
    columns=[(t,c) for t in targets for c in report['classes'][t]]+ [('vucut',c) for c in ['cekingen_kapali','gergin','gevsek']]
    x,y,ids,groups=[],[],[],[]
    for row in report['samples']:
        label=labels.get(row['video_id'].casefold())
        if row['split']!='train' or label not in NAMES or sum(row['valid_targets'].values())<3:continue
        cache=Path('results/demo_v5/audio_cache')/(row['source_sha256']+'.json')
        if not cache.exists():raise ValueError('Missing training audio cache')
        behavior=[float(source[row['video_id'].casefold()]['vucut']==c) if t=='vucut' else float(row['actual'][t]==c) if row['valid_targets'].get(t) else 0. for t,c in columns]
        x.append(behavior+audio_features(json.loads(cache.read_text())))
        y.append(label);ids.append(row['video_id']);groups.append(row['source_sha256'])
    x=np.asarray(x);y=np.asarray(y);n=len(columns)
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(x,y,groups))
    candidates=[];models={}
    for mode,indices in [('visual',list(range(n))),('audio',list(range(n,x.shape[1]))),('multimodal',list(range(x.shape[1])))]:
        best=None
        for c in [.1,.3,1.]:
            predictions=np.empty(len(y),dtype=object)
            for tr,va in folds:
                model=LogisticRegression(C=c,class_weight='balanced',max_iter=2000).fit(x[tr][:,indices],y[tr])
                predictions[va]=model.predict(x[va][:,indices])
            f1=float(f1_score(y,predictions,labels=list(NAMES),average='macro',zero_division=0))
            candidates.append(dict(mode=mode,C=c,proxy_cv_macro_f1=f1))
            if best is None or f1>best[0]:best=(f1,c)
        model=LogisticRegression(C=best[1],class_weight='balanced',max_iter=2000).fit(x[:,indices],y)
        models[mode]=dict(indices=indices,coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist(),C=best[1],proxy_cv_macro_f1=best[0])
    data=dict(version='human-emotion-labels-body-audio-v4',columns=columns,classes=list(model.classes_),models=models,
              train_videos=ids,class_counts={c:int(sum(y==c)) for c in NAMES},candidates=candidates,
              provenance='emotion_human_labels.csv; human emotion labels from two independent annotators plus adjudication (Cohen kappa 0.553)',
              limitations=['Labels are human-perceived emotions from video+audio; annotator agreement is moderate (kappa 0.553), so labels are subjective.',
                           'Scores are uncalibrated; day-grouped emotion accuracy is weak (macro-F1 about 0.40-0.44, chance about 0.25, 125 videos).',
                           'Training-time CV groups exact file hashes, not recording day; its macro-F1 is optimistic and not a test result.',
                           'Emotion labels are partly confounded with recording day (e.g. no Angry in the first day).',
                           'Human categorical behaviors train bridge; predicted soft behaviors at inference differ.'])
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');return data


def infer(comparison, model_path=MODEL, audio_analysis=None, quality=None):
    unavailable=dict(status='unavailable',label='Duygu durumu belirlenemedi',scores={})
    if not model_path.exists():return unavailable
    meta=json.loads(model_path.read_text());scores=(comparison or {}).get('scores',{})
    targets={t for t,_ in meta['columns'] if t!='vucut' or t in scores}
    visual_ok=bool(comparison) and all(t in scores and scores[t] and np.isfinite(list(scores[t].values())).all()
        and min(scores[t].values())>=0 and abs(sum(scores[t].values())-1)<.001 for t in targets)
    audio_ok=bool(audio_analysis and audio_analysis.get('status')=='ok' and any(
        w.get('scores',{}).get('Dog',0)>=.3
        and max(w.get('scores',{}).get(k,0) for k in ['Bark','Growling','Whimper (dog)','Howl'])>=.3
        and max(w.get('scores',{}).get(k,0) for k in ['Bark','Growling','Whimper (dog)','Howl'])>w.get('scores',{}).get('Speech',0)
        and w.get('scores',{}).get('Music',0)<.5
        for w in audio_analysis.get('windows',[])))
    ratio=(quality.get('detected_frames',0)/max(1,quality.get('sampled_frames',0))) if quality else 1.
    # Low detection coverage cannot support precise posture or hidden tail claims.
    visual_ok=visual_ok and ratio>=.5
    if not visual_ok and not audio_ok:return unavailable
    mode='multimodal' if visual_ok and audio_ok else 'visual' if visual_ok else 'audio'
    x=np.array([scores.get(t,{}).get(c,0.) for t,c in meta['columns']]+audio_features(audio_analysis))
    model=meta['models'][mode];z=np.asarray(model['coefficients'])@x[model['indices']]+np.asarray(model['intercept'])
    p=np.exp(z-z.max());p/=p.sum();index=int(p.argmax());rank=np.sort(p)
    decisive=bool(p[index]>=.5 and rank[-1]-rank[-2]>=.15)
    evidence={t:comparison['predictions'].get(t) for t in targets} if visual_ok else {}
    sound=[]
    for name in ['Growling','Bark','Whimper (dog)']:
        candidates=[w for w in (audio_analysis or {}).get('windows',[]) if w['scores'].get(name,0)>=.3
                    and w['scores'].get(name,0)>w['scores'].get('Speech',0) and w['scores'].get('Music',0)<.5]
        if candidates:sound.append(dict(event=name,peak=max(w['scores'][name] for w in candidates),windows=len(candidates)))
    conflict=bool(sound and any(v['event']=='Growling' for v in sound) and meta['classes'][index] in ['Sad','Relaxed'])
    return dict(status='human_label_estimate',label=NAMES[meta['classes'][index]],class_id=meta['classes'][index],
                scores=dict(zip(meta['classes'],map(float,p))),support='sınırlı' if not decisive or conflict else 'aday belirgin; doğrulanmadı',
                decision_status='uncertain' if not decisive or conflict else 'candidate',modality=mode,
                conflicting_evidence=conflict,visual_coverage=ratio,scores_calibrated=False,
                independent_emotion_accuracy_measured=False,source=meta['provenance'],behavior_evidence=evidence,audio_evidence=sound,
                reason={'multimodal':'Vücut, kuyruk, kulak, duruş ve hareket tahminleri ile YAMNet ses olayları birlikte değerlendirildi.',
                        'visual':'Ses kanıtı yok; vücut, kuyruk, kulak, duruş ve hareket tahminleri değerlendirildi.',
                        'audio':'Görüntü kanıtı yetersiz; dört duygu adayı yalnız ses olaylarından üretildi.'}[mode],
                limitations=meta['limitations'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    args=parser.parse_args()
    data=train(args.report,args.labels)
    print(json.dumps({'training_rows':len(data['train_videos']), 'counts':data['class_counts']}))
