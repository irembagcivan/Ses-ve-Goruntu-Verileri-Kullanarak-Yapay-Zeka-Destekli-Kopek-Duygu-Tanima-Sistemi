"""Auxiliary body-state head from the original worksheet's vucut column."""
import csv
import json
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[2] / 'configs/demo/body_state.json'
CLASSES = ['cekingen_kapali', 'gergin', 'gevsek']


def train():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score
    from src.deep_learning.train import normalize_fit, normalize_apply
    rows={r['video'].casefold():r for r in csv.DictReader(open('results/v14_input_review/emotion_rules_extracted.csv'))}
    data=np.load('results/demo_v13_release/feature_snapshot.npz')
    ids=[int(i) for i in data['train_indices'] if rows[str(data['video_ids'][i]).casefold()]['vucut'] in CLASSES]
    x=data['visual'][ids]; y=np.array([rows[str(data['video_ids'][i]).casefold()]['vucut'] for i in ids])
    groups=data['source_sha256'][ids]
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(x,y,groups))
    candidates=[]
    for c in [.01,.1,1.]:
        pred=np.empty(len(y),dtype=object)
        for tr,va in folds:
            stats=normalize_fit(x[tr]);model=LogisticRegression(C=c,class_weight='balanced',max_iter=2000)
            model.fit(normalize_apply(x[tr],stats).numpy(),y[tr]);pred[va]=model.predict(normalize_apply(x[va],stats).numpy())
        candidates.append({'C':c,'training_cv_macro_f1':float(f1_score(y,pred,labels=CLASSES,average='macro',zero_division=0))})
    chosen=max(candidates,key=lambda v:v['training_cv_macro_f1']);stats=normalize_fit(x)
    model=LogisticRegression(C=chosen['C'],class_weight='balanced',max_iter=2000).fit(normalize_apply(x,stats).numpy(),y)
    meta=json.load(open('results/demo_v13_release/neural_fusion.json'))
    result=dict(version='worksheet-body-v1',classes=list(model.classes_),columns=meta['protocol']['visual_columns'],
                scaler=stats,coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist(),
                train_videos=[str(data['video_ids'][i]) for i in ids],candidates=candidates,selection=chosen,
                source='Original worksheet body labels retained in previously supplied rule-derived CSV; screenshot schema matches.',
                limitations=['Not final consensus body labels; source versions differ.', 'Exact-file-hash CV does not validate unseen dogs.', 'Latest Excel binary was not accessible.'])
    MODEL.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result


def infer(visual,quality):
    if not MODEL.exists() or quality.get('detected_frames',0)/max(1,quality.get('sampled_frames',0))<.5:
        return {'status':'unavailable','reason':'Vücut için görüntü kapsamı yetersiz.'}
    from src.deep_learning.train import normalize_apply
    meta=json.loads(MODEL.read_text());x=normalize_apply([[visual[c] for c in meta['columns']]],meta['scaler']).numpy()[0]
    z=np.asarray(meta['coefficients'])@x+meta['intercept'];p=np.exp(z-z.max());p/=p.sum()
    return dict(status='estimated',prediction=meta['classes'][int(p.argmax())],scores=dict(zip(meta['classes'],map(float,p))),
                source=meta['source'],scores_calibrated=False)


if __name__=='__main__':
    result=train();print(json.dumps({'train':len(result['train_videos']),'selection':result['selection']}))
