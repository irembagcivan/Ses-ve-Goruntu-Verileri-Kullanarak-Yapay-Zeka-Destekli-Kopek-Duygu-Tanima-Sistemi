"""Gün gruplu (leave-one-group-out) duygu değerlendirmesi: insan duygu etiketleri.

Girdi: dondurulmuş özellikler (results/demo_v13_final/feature_snapshot.npz),
insan duygu etiketleri (results/v14_input_review/emotion_human_labels.csv) ve
kayıt günü grupları (results/v14_input_review/emotion_day_groups.csv).
Her grup (A: 18 Haz, B: 22 Haz, C: diğer/tarihsiz) sırayla test dışı bırakılır.
Ölçekleme ve eksik değer doldurma yalnız eğitim katında öğrenilir.
C değerlerinin hepsi raporlanır; en iyisi test skoruna bakılarak seçilmez.
Kullanım: python -m src.evaluation.human_emotion_day_eval
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
LABELS = ['Relaxed', 'Happy', 'Angry', 'Sad']
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/human_emotion_eval'


def load():
    snap = np.load(ROOT / 'results/demo_v13_final/feature_snapshot.npz', allow_pickle=True)
    lab = pd.read_csv(ROOT / 'results/v14_input_review/emotion_human_labels.csv')
    grp = pd.read_csv(ROOT / 'results/v14_input_review/emotion_day_groups.csv')
    table = lab.merge(grp, on='video')
    table = table[table.label.isin(LABELS)].copy()
    index = {v.casefold(): i for i, v in enumerate(snap['video_ids'])}
    table = table[table.video.str.casefold().isin(index)]
    rows = [index[v.casefold()] for v in table.video]
    visual, audio = snap['visual'][rows], snap['audio'][rows]
    return table.label.values, table.grup.values, {
        'gorsel': visual, 'ses': audio, 'birlesik': np.hstack([visual, audio])}


def predict(x, y, g, c):
    pred = np.empty(len(y), dtype=object)
    for group in np.unique(g):
        test = g == group
        model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                              LogisticRegression(C=c, class_weight='balanced', max_iter=3000))
        pred[test] = model.fit(x[~test], y[~test]).predict(x[test])
    return pred


def main():
    y, g, features = load()
    rng = np.random.default_rng(0)
    result = {'n': int(len(y)), 'siniflar': {k: int((y == k).sum()) for k in LABELS},
              'gruplar': {k: int((g == k).sum()) for k in np.unique(g)}, 'tesadufi_seviye_yaklasik': 0.25, 'modeller': []}
    for name, x in features.items():
        for c in [.01, .1, 1.]:
            p = predict(x, y, g, c)
            f1 = f1_score(y, p, labels=LABELS, average='macro', zero_division=0)
            boot = [f1_score(y[b], p[b], labels=LABELS, average='macro', zero_division=0)
                    for b in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            if name == 'gorsel' and c == .1:  # rapordaki Tablo 3 / Şekil 2 (sınıf bazında P/R/F1 ve karışıklık matrisi)
                pr, rc, ff, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
                result['sinif_bazinda_gorsel_C0.1'] = {k: dict(precision=round(float(a), 2), recall=round(float(b), 2), f1=round(float(d), 2))
                                                      for k, a, b, d in zip(LABELS, pr, rc, ff)}
                result['karisiklik_gorsel_C0.1'] = dict(siralama=LABELS, matris=confusion_matrix(y, p, labels=LABELS).tolist())
            result['modeller'].append(dict(ozellik=name, C=c, macro_f1=round(float(f1), 3),
                                           ci95=[round(float(lo), 3), round(float(hi), 3)],
                                           dogruluk=round(float(accuracy_score(y, p)), 3)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# İnsan duygu etiketleriyle gün gruplu değerlendirme', '',
             f"{result['n']} video, sınıflar {result['siniflar']}, gruplar {result['gruplar']}.", '',
             '| Özellik | C | macro-F1 | %95 GA | Doğruluk |', '|---|---:|---:|---|---:|']
    lines += [f"| {m['ozellik']} | {m['C']} | {m['macro_f1']} | {m['ci95']} | {m['dogruluk']} |" for m in result['modeller']]
    lines += ['', 'Tüm C değerleri raporlanır; en iyisi test skoruna göre seçilmedi. Tesadüf seviyesi yaklaşık 0.25.',
              'Bu bir bağımsız test değildir; üç kayıt grubu vardır ve etiketler gün ile kısmen karışıktır.']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
