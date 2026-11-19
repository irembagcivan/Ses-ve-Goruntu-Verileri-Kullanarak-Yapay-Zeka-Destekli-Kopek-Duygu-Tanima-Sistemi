"""Audit final consensus labels and refit six tasks without reusing old labels.

Frozen pretrained features are reused; the original 25 test source videos never
enter training. Missing/non-consensus target cells and unsafe rows are masked.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .train import FusionNet, losses, normalize_apply, normalize_fit, seed, weights_for
from .sequence import SequenceNet, scale_apply, scale_fit
from .paper_hybrid import HybridNet
from src.demo.extract import clean_json

TARGETS = ['hareket', 'ses', 'durus', 'kuyruk_konum', 'kuyruk_hareket', 'kulak']
CLASSES = {
    'hareket': ['hareketsiz', 'kosuyor_oynuyor', 'yuruyor'],
    'ses': ['havlama', 'hirlama', 'inleme', 'yok'],
    'durus': ['ayakta', 'oturuyor', 'sirtustu', 'yatiyor'],
    'kuyruk_konum': ['asagi_bacak_arasi', 'notr', 'yukari'],
    'kuyruk_hareket': ['sallamiyor', 'salliyor'],
    'kulak': ['dik_one', 'geride_yassi', 'notr'],
}
NEURAL_KINDS = ['neural_visual', 'neural_audio', 'neural_fusion', 'sequence_cnn',
                'sequence_gru', 'sequence_bigru', 'sequence_lstm_lstm_gru']
EPOCHS = [20, 40, 60, 80]


def audit_labels(labels, old):
    required = ['video', 'durum', 'guvensiz', 'degisken', 'konsensus_disi', *TARGETS]
    if set(required) - set(labels):
        raise ValueError('Missing columns: ' + str(set(required) - set(labels)))
    labels = labels.copy().fillna('')
    labels['key'] = labels.video.str.strip().str.casefold()
    old = old.copy().fillna(''); old['key'] = old.video_id.str.strip().str.casefold()
    if labels.key.duplicated().any() or old.key.duplicated().any():
        raise ValueError('Duplicate video key')
    if set(labels.key) != set(old.key):
        raise ValueError('Final label/video inventory mismatch')
    labels = labels.set_index('key').loc[old.key].reset_index(drop=True)
    for flag in ['guvensiz', 'degisken']:
        if not set(labels[flag].astype(str)) <= {'0', '1'}:
            raise ValueError('Invalid quality flag: ' + flag)
    y = np.full((len(old), len(TARGETS)), -1, dtype=np.int64)
    reasons = []
    for i, row in labels.iterrows():
        consensus = set(filter(None, str(row.konsensus_disi).split(',')))
        if consensus - set(TARGETS):
            raise ValueError('Unknown consensus target')
        for j, t in enumerate(TARGETS):
            value = str(row[t]).strip()
            if value not in CLASSES[t] and value not in {'', 'gorunmuyor'}:
                raise ValueError(f'Unknown label: {row.video} {t}={value}')
            reason = ('unsafe' if str(row.guvensiz) == '1' else
                      'incomplete' if row.durum != 'tam' else
                      'no_consensus' if t in consensus else
                      'missing_or_invisible' if value in {'', 'gorunmuyor'} else None)
            if reason: reasons.append(dict(video_id=row.video, target=t, reason=reason))
            else: y[i, j] = CLASSES[t].index(value)
    # Preserve test membership even for unsafe rows; mask their target labels.
    test = np.flatnonzero(old.split.eq('test'))
    train = np.flatnonzero(~old.split.eq('test') & (y >= 0).any(axis=1))
    if set(old.iloc[train].source_sha256) & set(old.iloc[test].source_sha256):
        raise ValueError('Source hash leakage')
    audit = dict(total=len(old), formerly_unlabelled=int(old.split.eq('unlabelled').sum()),
                 newly_usable_training=int((old.iloc[train].split == 'unlabelled').sum()),
                 unsafe=int(labels.guvensiz.astype(str).eq('1').sum()),
                 variable_videos=int(labels.degisken.astype(str).eq('1').sum()),
                 training_videos=len(train), original_test_videos=len(test),
                 evaluable_test_videos=int((y[test] >= 0).any(axis=1).sum()),
                 masked_cells=reasons, targets=TARGETS, classes=CLASSES,
                 per_target={t: dict(training=int((y[train, j] >= 0).sum()),
                                     test=int((y[test, j] >= 0).sum()),
                                     train_class_counts={c: int((y[train, j] == k).sum()) for k, c in enumerate(CLASSES[t])})
                             for j, t in enumerate(TARGETS)},
                 source_library_id='libfile_493a44e079cc81918ff2db14060922e2',
                 source_access='Full CSV text extraction; source-byte download unavailable',
                 limits=['Behavior/sound annotations, no emotion ground truth.',
                         'Largest-box visual features may not match the annotators focal dog.',
                         'Sound labels may refer to the focal dog; YAMNet observes the mixed recording.',
                         'Video labels do not validate temporal window predictions.',
                         'Previously inspected holdout, not an independent new test; no dog identity split.'])
    return labels, y, train, test, audit


def metric(outputs, y):
    result = []
    for j, t in enumerate(TARGETS):
        actual = y[:, j]; valid = actual >= 0; n = len(CLASSES[t])
        pred = np.asarray(outputs[j]).argmax(1)
        if not valid.any():
            result.append(dict(target=t, macro_f1=None, accuracy=None, support=0, classes=CLASSES[t]))
            continue
        result.append(dict(target=t, macro_f1=float(f1_score(actual[valid], pred[valid], labels=list(range(n)), average='macro', zero_division=0)),
                           accuracy=float(accuracy_score(actual[valid], pred[valid])), support=int(valid.sum()),
                           classes=CLASSES[t], confusion_matrix=confusion_matrix(actual[valid], pred[valid], labels=list(range(n))).tolist()))
    return result


def mean_f1(outputs, y):
    values = [m['macro_f1'] for m in metric(outputs, y) if m['macro_f1'] is not None]
    return float(np.mean(values))


def neural_fit(v, a, x, lengths, y, fit, val, kind, epochs):
    seed(); sizes = [len(CLASSES[t]) for t in TARGETS]
    if kind.startswith('neural_'):
        vs, ast = normalize_fit(v[fit]), normalize_fit(a[fit])
        vt, at = normalize_apply(v, vs), normalize_apply(a, ast)
        model = FusionNet(v.shape[1], a.shape[1], sizes, kind.removeprefix('neural_'))
        def forward(ids): return model(vt[ids], at[ids])
        metadata = dict(visual_dim=v.shape[1], audio_dim=a.shape[1], class_counts=sizes,
                        modality=kind.removeprefix('neural_'), visual_scaler=vs, audio_scaler=ast)
    else:
        stats = scale_fit(x, lengths, fit); xt = scale_apply(x, lengths, stats); lt = torch.tensor(lengths)
        name = kind.removeprefix('sequence_')
        model = HybridNet(sizes, name) if name == 'lstm_lstm_gru' else SequenceNet(sizes, name)
        def forward(ids): return model(xt[ids], lt[ids])
        metadata = dict(scaler=stats, sizes=sizes, kind=name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    weights = weights_for(y[fit], sizes); yt = torch.tensor(y); trace = []
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(); loss = losses(forward(fit), yt[fit], weights)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if val is not None and epoch in EPOCHS:
            model.eval()
            with torch.inference_mode(): outputs = [o.numpy() for o in forward(val)]
            trace.append(dict(epoch=epoch, loss=float(loss.detach()), macro_f1=mean_f1(outputs, y[val])))
    model.eval()
    with torch.inference_mode(): probabilities = [torch.softmax(o, 1).numpy() for o in forward(np.arange(len(y)))]
    return model, metadata, trace, probabilities


def classical_fit(data, y, fit, val, classifier):
    outputs, models = [], []
    for j, t in enumerate(TARGETS):
        valid = fit[y[fit, j] >= 0]; labels = y[valid, j]
        estimator = DummyClassifier(strategy='most_frequent') if len(set(labels)) < 2 or classifier == 'dummy' else (
            LogisticRegression(C=1., class_weight='balanced', max_iter=2000) if classifier == 'lr' else
            RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=2))
        model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), estimator)
        model.fit(data[valid], labels); p = np.zeros((len(val), len(CLASSES[t])))
        p[:, model.classes_.astype(int)] = model.predict_proba(data[val]); outputs.append(p); models.append(model)
    return outputs, models


def run(args):
    torch.set_num_threads(2); args.output.mkdir(parents=True, exist_ok=False)
    old = pd.read_csv(args.manifest).fillna(''); source = pd.read_csv(args.labels).fillna('')
    annotations, y, train, test, audit = audit_labels(source, old)
    with np.load(args.features, allow_pickle=False) as d:
        if not np.array_equal(d['source_sha256'], old.source_sha256.to_numpy(str)):
            raise ValueError('Feature snapshot order mismatch')
        v, a = d['visual'].copy(), d['audio'].copy()
    with np.load(args.sequences, allow_pickle=False) as d:
        x, lengths = d['sequences'].copy(), d['lengths'].copy()
        if len(x) != len(old) or not np.array_equal(d['test_indices'], test):
            raise ValueError('Sequence snapshot order/split mismatch')
    if not np.isfinite(x).all(): raise ValueError('Invalid sequence features')
    audit['extracted_labels_sha256'] = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    (args.output/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    manifest = old[['video_id','video_path','source_sha256']].copy()
    for t in annotations.columns:
        if t != 'video': manifest[t] = annotations[t].to_numpy()
    manifest['split'] = 'excluded'; manifest.loc[train, 'split'] = 'train'; manifest.loc[test, 'split'] = 'test'
    manifest.to_csv(args.output/'manifest.csv', index=False)
    np.savez_compressed(args.output/'feature_snapshot.npz', visual=v, audio=a, sequences=x, lengths=lengths,
                        labels=y, train_indices=train, test_indices=test, video_ids=old.video_id.to_numpy(str), source_sha256=old.source_sha256.to_numpy(str))
    print(json.dumps({k:audit[k] for k in ['total','training_videos','evaluable_test_videos','newly_usable_training','unsafe']}), flush=True)
    folds = [(train[f], train[g]) for f, g in GroupKFold(3).split(train, groups=old.iloc[train].source_sha256)]
    choices, histories = {}, {}
    for kind in NEURAL_KINDS:
        histories[kind] = []
        for fold, (f, g) in enumerate(folds):
            _, _, trace, _ = neural_fit(v,a,x,lengths,y,f,g,kind,80)
            histories[kind].append(trace); print(f'CV {kind} {fold+1}/3', flush=True)
        values = {e: float(np.mean([next(z['macro_f1'] for z in trace if z['epoch'] == e) for trace in histories[kind]])) for e in EPOCHS}
        epoch = min(values, key=lambda e: (-values[e],e)); choices[kind] = dict(epochs=epoch, cv_macro_f1=values[epoch], candidates=values)
    classical = {}
    for name, data in [('visual',v),('audio',a),('fusion',np.concatenate([v,a],1))]:
        values = {}
        for kind in ['lr','rf']:
            values[kind] = float(np.mean([mean_f1(classical_fit(data,y,f,g,kind)[0],y[g]) for f,g in folds]))
        winner = max(values,key=values.get); classical[name] = dict(classifier=winner, candidates=values)
        choices['classical_'+name] = dict(cv_macro_f1=values[winner], classifier=winner)
    selected = max(choices,key=lambda name: choices[name]['cv_macro_f1'])
    selection = dict(models=choices, selected_by_training_cv=selected, seed=42,
                     folds=[dict(fit=old.iloc[f].video_id.tolist(),validation=old.iloc[g].video_id.tolist()) for f,g in folds])
    (args.output/'selection.json').write_text(json.dumps(selection, indent=2))
    # Only now evaluate test performance. No test score selects a model.
    legacy = json.loads(args.metadata.read_text())
    protocol = dict(legacy['protocol'], targets=TARGETS, classes=CLASSES,
                    split=f'{len(train)} training / {len(test)} original held-out files, unsafe target rows masked',
                    annotation_version='labels_final-v13', emotion_ground_truth=False,
                    label_source_sha256=audit['extracted_labels_sha256'],
                    epoch_choices=choices, folds=selection['folds'])
    samples = [dict(video_id=row.video_id, source_sha256=row.source_sha256, split=manifest.iloc[i]['split'],
                    original_split=row.split, actual={t: annotations.iloc[i][t] for t in TARGETS},
                    valid_targets={t: bool(y[i,j]>=0) for j,t in enumerate(TARGETS)},
                    annotation_quality={k:annotations.iloc[i][k] for k in ['guvensiz','degisken','konsensus_disi','kopek_sayisi','gorunurluk','not']},
                    predictions={},scores={}) for i,row in old.iterrows()]
    results=[]; bundle={}
    def record(name, probabilities):
        metrics=metric([p[test] for p in probabilities],y[test])
        single=test[annotations.iloc[test].kopek_sayisi.astype(str).eq('1').to_numpy()]
        multiple=test[~annotations.iloc[test].kopek_sayisi.astype(str).eq('1').to_numpy()]
        results.append(dict(model=name, mean_test_macro_f1=mean_f1([p[test] for p in probabilities],y[test]), metrics=metrics,
                            single_dog_metrics=metric([p[single] for p in probabilities],y[single]),
                            multi_dog_metrics=metric([p[multiple] for p in probabilities],y[multiple])))
        for j,t in enumerate(TARGETS):
            for i,p in enumerate(probabilities[j]):
                samples[i]['predictions'].setdefault(name,{})[t]=CLASSES[t][int(p.argmax())]
                samples[i]['scores'].setdefault(name,{})[t]=dict(zip(CLASSES[t],map(float,p)))
        print(name, results[-1]['mean_test_macro_f1'],flush=True)
    for kind in NEURAL_KINDS:
        model, metadata, _, probabilities=neural_fit(v,a,x,lengths,y,train,None,kind,choices[kind]['epochs'])
        metadata.update(targets=TARGETS, classes=CLASSES, epochs=choices[kind]['epochs'], annotation_version='labels_final-v13')
        if kind.startswith('neural_'): metadata['protocol']=protocol
        torch.save(model.state_dict(),args.output/(kind+'.pt'))
        (args.output/(kind+'.json')).write_text(json.dumps(metadata,ensure_ascii=False))
        record(kind,probabilities)
    import joblib
    for name,data in [('visual',v),('audio',a),('fusion',np.concatenate([v,a],1))]:
        p,models=classical_fit(data,y,train,np.arange(len(y)),classical[name]['classifier'])
        bundle[name]=models; record('classical_'+name,p)
    p,_=classical_fit(v,y,train,np.arange(len(y)),'dummy'); record('majority_baseline',p)
    joblib.dump(dict(models=bundle,targets=TARGETS,classes=CLASSES),args.output/'classical.joblib')
    report=dict(version='labels_final-v13', targets=TARGETS, classes=CLASSES, audit=audit, selection=selection,
                models=results,samples=samples,limits=audit['limits'],histories=histories)
    (args.output/'report.json').write_text(json.dumps(clean_json(report),ensure_ascii=False,indent=2,allow_nan=False,
                                                    default=lambda value: value.item() if isinstance(value,np.generic) else str(value)))
    print('Final labels refit complete:', args.output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key,default in [('labels','results/labels_v13/source_labels_extracted.csv'),('manifest','results/demo_v4/manifest.csv'),
                        ('features','results/demo_v9/feature_snapshot.npz'),('sequences','results/demo_v11/sequence_snapshot.npz'),
                        ('metadata','results/demo_v9/neural_fusion.json')]: p.add_argument('--'+key,type=Path,default=Path(default))
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
