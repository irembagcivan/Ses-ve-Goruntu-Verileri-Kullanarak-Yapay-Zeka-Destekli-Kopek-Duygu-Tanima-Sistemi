"""Audio-only perceived affect; never infer mood from a behavior-rule label."""
import json
from pathlib import Path
import joblib,numpy as np,torch
from src.deep_learning.train import normalize_apply
from .train import AffectNet,CLASSES,embedding_features


def vocal_segment(embeddings,patch_scores,class_names,threshold=.3):
    e=np.asarray(embeddings);p=np.asarray(patch_scores)
    if e.ndim!=2 or e.shape[1]!=1024 or p.ndim!=2 or len(e)!=len(p):raise ValueError('Unaligned audio evidence')
    bark=p[:,class_names.index('Bark')];speech=p[:,class_names.index('Speech')];music=p[:,class_names.index('Music')]
    valid=np.flatnonzero((bark>=threshold)&(bark>speech)&(music<.5))
    if not len(valid):return None
    runs=np.split(valid,np.where(np.diff(valid)>1)[0]+1);candidates=[]
    for run in runs:
        for start in range(max(1,len(run)-7)):
            part=run[start:start+8];candidates.append(part)
    ids=max(candidates,key=lambda ids:float(bark[ids].mean()))
    return dict(features=embedding_features(e[ids]),start=float(ids[0]*.48),end=float(ids[-1]*.48+.975),
                patches=len(ids),bark_score=float(bark[ids].mean()))


def infer_affect(embeddings,patch_scores,class_names,root=Path('results/affect_v15_clean'),weights_sha256=None):
    if not (root/'model.json').exists():return dict(status='model_unavailable',reason='Ses duygu modeli bu pakette bulunmuyor.')
    if embeddings is None:return dict(status='no_audio',reason='Ses kanalı yok; sesten duygu durumu tahmin edilemedi.')
    segment=vocal_segment(embeddings,patch_scores,class_names)
    if segment is None:return dict(status='no_clear_bark',reason='Belirgin havlama bölümü bulunamadı. Sessizlik nötr veya mutlu duygu kanıtı değildir.')
    meta=json.loads((root/'model.json').read_text())
    if weights_sha256 is not None and weights_sha256!=meta['yamnet_sha256']:raise ValueError('Affect encoder checkpoint mismatch')
    x=segment.pop('features')[None];selected=meta['selected'];outputs={}
    if selected in ['neural','soft_vote']:
        net=AffectNet();net.load_state_dict(torch.load(root/'neural.pt',weights_only=True,map_location='cpu'));net.eval()
        with torch.inference_mode():outputs['neural']=[torch.softmax(o,1).numpy()[0] for o in net(normalize_apply(x,meta['scaler']))]
    if selected in ['lr','rf','soft_vote']:
        models=joblib.load(root/'classical.joblib');kind='lr' if selected=='soft_vote' else selected;values=[]
        for estimator in models[kind]:
            p=np.zeros(3);p[estimator.classes_.astype(int)]=estimator.predict_proba(x)[0];values.append(p)
        outputs[kind]=values
    p=[(outputs['neural'][j]+outputs['lr'][j])/2 for j in range(2)] if selected=='soft_vote' else outputs[selected]
    predictions={t:CLASSES[t][int(values.argmax())] for t,values in zip(CLASSES,p)}
    scores={t:dict(zip(CLASSES[t],map(float,values))) for t,values in zip(CLASSES,p)}
    margins=[float(np.sort(values)[-1]-np.sort(values)[-2]) for values in p]
    supported=all(float(values.max())>=.5 for values in p) and min(margins)>=.15
    val={'Positive':'Olumlu','Neutral':'Nötr','Negative':'Olumsuz'}[predictions['valence']]
    arousal={'Low':'düşük uyarılma','Medium':'orta uyarılma','High':'yüksek uyarılma'}[predictions['arousal']]
    return dict(status='estimated' if supported else 'limited_evidence',label=f'{val} · {arousal}',
                version=meta['version'],source_commit=meta['source_commit'],
                predictions=predictions,scores=scores,support='belirgin' if supported else 'sınırlı',
                scores_calibrated=False,source='EmotionalCanines 2025 insan algısı etiketleri; ses temelli deneysel aktarım',
                model=selected,segment=segment,
                interpretation='Bu, havlama sesinin insan etiketlerine benzerliğidir; korku/öfke veya mutlu/üzgün ayrımını doğrudan eğitilmiş sınıflar gibi vermez.',
                limitations=meta['limits'])


def assessment(analysis):
    affect=analysis.get('audio_affect',{});status=affect.get('status','not_run')
    comparison=analysis.get('model_comparisons',{}).get(analysis.get('preferred_model','FusionNet'))
    available=status in ['estimated','limited_evidence']
    return dict(label=affect.get('label') if available else 'Duygu durumu belirlenemedi',
                status=status,scope='Kaydın sesinden algılanan duygu boyutları; tek köpeğin kesin ruh hali değil',
                support=affect.get('support','yetersiz'),reason=affect.get('interpretation') if available else affect.get('reason','Bu eski analizde duygu modeli çalıştırılmadı; videoyu yeniden analiz et.'),
                behavior_evidence=comparison.get('predictions',{}) if comparison else {},
                multi_dog_warning=bool(analysis.get('quality',{}).get('multi_dog_frames',0)),
                emotion_accuracy_on_local_videos_measured=False)
