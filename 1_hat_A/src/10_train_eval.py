"""
Aşama 6 - Eğitim ve değerlendirme (sızıntısız)
----------------------------------------------
Girdi : outputs/features_all.npz (09), outputs/targets_and_splits.csv (08)
Çıktı : outputs/results/  (metrics.csv, per_fold.csv, predictions.csv, summary.md, confusion_*.csv, fig_*.png)

Hedefler : ses3, durus3, hareket3, kuyruk2, duygu4 (duygu: iki etiketleyicinin insan etiketi, ortak kararla)
Özellik grupları:
  ses (55) | goruntu_hareket (2560 + 9) | erken_birlesim (hepsi tek vektör) | gec_birlesim (iki modelin olasılık ortalaması)
Sınıflandırıcılar (ÖNCEDEN belirlenmiş sabit ayarlar, test üzerinde ayar YAPILMAZ):
  LR : standardize + L2 lojistik regresyon (C=0.1, sınıf ağırlıklı)
  RF : 300 ağaçlı rastgele orman (sınıf ağırlıklı)
Referanslar: baz_cogunluk (eğitimdeki en sık sınıf), gun_bilgisi (yalnızca rastgele bölmede: kaydın günü grubunun
  eğitimdeki çoğunluk sınıfı; oturum bölmesinde uygulanamaz çünkü test günü eğitimde yok).
Değerlendirme protokolleri:
  oturum   : kayıt günü grubu (A/B/C) sırayla test, kalanlar eğitim (asıl sonuç)
  rastgele : 5 rastgele fold (karşılaştırma: sızıntı varken skor nasıl şişiyor)
Metrikler: doğruluk, macro-F1 (ana metrik: sınıflar dengesiz), %95 bootstrap aralığı (1000 yeniden örnekleme),
  birleşik (out-of-fold) karışıklık matrisi. Ön-belirlenmiş birincil model: gec_birlesim_LR.

Çalıştırma:  python src/10_train_eval.py            (sonuçlar)
             python src/10_train_eval.py --save-final   (demo için tüm veriyle eğitilmiş modeller)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TARGETS = ["ses3", "durus3", "hareket3", "kuyruk2", "duygu4"]
PRIMARY = "gec_birlesim_LR"
FEATS = ["ses", "goruntu_hareket", "erken_birlesim", "gec_birlesim"]
CLFS = ["LR", "RF"]
OUT = Path("outputs/results")


def make_clf(kind):
    if kind == "LR":
        est = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    est = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2,
                                 random_state=42, n_jobs=-1)
    return make_pipeline(SimpleImputer(strategy="median"), est)


def fit_predict(feat, kind, B, y, tr, te):
    """B: dict blok adı -> dizi. tr/te: indeks dizileri. Dönüş: tahmin etiketleri."""
    ytr = y[tr]
    if len(set(ytr)) < 2:
        return np.array([ytr[0]] * len(te))
    if feat == "ses":
        blocks = [B["aud"]]
    elif feat == "goruntu_hareket":
        blocks = [B["vm"]]
    elif feat == "erken_birlesim":
        blocks = [np.hstack([B["aud"], B["vm"]])]
    else:  # gec_birlesim
        blocks = [B["aud"], B["vm"]]
    probas, classes = [], None
    for X in blocks:
        m = make_clf(kind).fit(X[tr], ytr)
        probas.append(m.predict_proba(X[te]))
        classes = m.classes_
    return classes[np.mean(probas, axis=0).argmax(axis=1)]


def boot_ci(y, p, labels, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        vals.append(f1_score(y[s], p[s], labels=labels, average="macro", zero_division=0))
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def folds_for(protocol, df, mask):
    ids = np.where(mask)[0]
    if protocol == "oturum":
        return [(ids[df.grup.values[ids] != g], ids[df.grup.values[ids] == g], g) for g in "ABC"
                if (df.grup.values[ids] == g).any()]
    f = df.rastgele_fold.values
    return [(ids[f[ids] != k], ids[f[ids] == k], str(k)) for k in range(5)]


def run_all(df, B):
    rows, pf, preds, cms = [], [], [], {}
    for tgt in TARGETS:
        y_all = df[tgt].astype(str).values
        mask = y_all != ""
        labels = sorted(set(y_all[mask]))
        for protocol in ["oturum", "rastgele"]:
            models = [("baz_cogunluk", None, None)] + [(f, c, f"{f}_{c}") for f in FEATS for c in CLFS]
            if protocol == "rastgele":
                models.insert(1, ("gun_bilgisi", None, None))
            for name, kind, key in models:
                key = key or name
                pred = np.empty(len(df), dtype=object)
                for tr, te, fid in folds_for(protocol, df, mask):
                    if name == "baz_cogunluk":
                        pred[te] = pd.Series(y_all[tr]).value_counts().idxmax()
                    elif name == "gun_bilgisi":
                        for g in set(df.grup.values[te]):
                            tg = tr[df.grup.values[tr] == g]
                            fallback = pd.Series(y_all[tr]).value_counts().idxmax()
                            val = pd.Series(y_all[tg]).value_counts().idxmax() if len(tg) else fallback
                            pred[te[df.grup.values[te] == g]] = val
                    else:
                        pred[te] = fit_predict(name, kind, B, y_all, tr, te)
                    fl = sorted(set(y_all[te]))
                    pf.append(dict(protokol=protocol, hedef=tgt, model=key, fold=fid, n=len(te),
                                   macro_f1=f1_score(y_all[te], pred[te].astype(str), labels=fl, average="macro",
                                                     zero_division=0),
                                   acc=accuracy_score(y_all[te], pred[te].astype(str))))
                yy, pp = y_all[mask], pred[mask].astype(str)
                lo, hi = boot_ci(yy, pp, labels)
                rows.append(dict(protokol=protocol, hedef=tgt, model=key, n=int(mask.sum()),
                                 acc=accuracy_score(yy, pp),
                                 macro_f1=f1_score(yy, pp, labels=labels, average="macro", zero_division=0),
                                 ci_lo=lo, ci_hi=hi))
                preds.append(pd.DataFrame(dict(protokol=protocol, hedef=tgt, model=key, video=df.video.values[mask],
                                               y_true=yy, y_pred=pp)))
                if key == PRIMARY:
                    cms[(protocol, tgt)] = (labels, confusion_matrix(yy, pp, labels=labels))
        print(f"  tamam: {tgt}")
    return pd.DataFrame(rows), pd.DataFrame(pf), pd.concat(preds), cms


# ----------------------------- grafikler -----------------------------
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SER = {"ses": "#2a78d6", "goruntu_hareket": "#eb6834", "erken_birlesim": "#1baf7a", "gec_birlesim": "#eda100"}
NAMES = {"ses": "Ses", "goruntu_hareket": "Görüntü+hareket", "erken_birlesim": "Erken birleşim",
         "gec_birlesim": "Geç birleşim"}
TNAMES = {"ses3": "Ses", "durus3": "Duruş", "hareket3": "Hareket", "kuyruk2": "Kuyruk\nhareketi", "duygu4": "Duygu"}


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#bdbcb5")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=.8)
    ax.set_axisbelow(True)


def fig_compare(M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True, facecolor="white")
    for ax, protocol, title in zip(axes, ["oturum", "rastgele"],
                                   ["(a) Oturuma göre bölme (asıl sonuç)", "(b) Rastgele bölme (sızıntı var)"]):
        style(ax)
        d = M[(M.protokol == protocol) & (M.model.str.endswith("_LR"))]
        w = 0.19
        for j, f in enumerate(FEATS):
            xs, ys, lo, hi = [], [], [], []
            for i, t in enumerate(TARGETS):
                r = d[(d.hedef == t) & (d.model == f"{f}_LR")].iloc[0]
                xs.append(i + (j - 1.5) * w); ys.append(r.macro_f1); lo.append(r.macro_f1 - r.ci_lo); hi.append(r.ci_hi - r.macro_f1)
            ax.bar(xs, ys, w * 0.92, color=SER[f], label=NAMES[f], zorder=3)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=.8, capsize=2, zorder=4)
        for i, t in enumerate(TARGETS):
            b = M[(M.protokol == protocol) & (M.hedef == t) & (M.model == "baz_cogunluk")].iloc[0].macro_f1
            ax.hlines(b, i - .45, i + .45, colors=MUTED, linestyles="--", lw=1.4, zorder=5,
                      label="Çoğunluk sınıfı referansı" if i == 0 else None)
            if protocol == "rastgele":
                g = M[(M.protokol == protocol) & (M.hedef == t) & (M.model == "gun_bilgisi")].iloc[0].macro_f1
                ax.plot([i], [g], marker="D", ms=5, color=INK, zorder=6, ls="none",
                        label="Yalnızca kayıt günü" if i == 0 else None)
        ax.set_xticks(range(len(TARGETS)))
        ax.set_xticklabels([TNAMES[t] for t in TARGETS])
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Macro-F1 (%95 bootstrap aralığı)", color=MUTED)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=9, labelcolor=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig_f1_karsilastirma.png", dpi=200)
    plt.close(fig)


def fig_confusion(cms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7), facecolor="white")
    axes = axes.ravel()
    for ax, t in zip(axes, TARGETS):
        labels, cm = cms[("oturum", t)]
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if norm[i, j] > .55 else INK)
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8, color=MUTED)
        ax.set_yticklabels(labels, fontsize=8, color=MUTED)
        ax.set_xlabel("Tahmin", color=MUTED, fontsize=9); ax.set_ylabel("Gerçek", color=MUTED, fontsize=9)
        ax.set_title(TNAMES[t].replace("\n", " "), loc="left", fontsize=11, color=INK)
    axes[-1].axis("off")
    axes[-1].text(0, .6, "Birincil model: geç birleşim (LR)\nOturuma göre bölme, birleşik test tahminleri\n"
                         "Hücre rengi: satır oranı; sayı: video adedi\nDuygu: insan etiketi (iki etiketleyici + ortak karar)", fontsize=9,
                  color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(OUT / "fig_karisiklik_matrisleri.png", dpi=200)
    plt.close(fig)


def fig_distribution(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), facecolor="white")
    for ax, t, title in zip(axes, ["duygu4", "hareket3"], ["Duygu (insan etiketi)", "Hareket"]):
        style(ax)
        d = df[df[t] != ""]
        cls = sorted(d[t].unique())
        w = 0.8 / len(cls)
        for j, c in enumerate(cls):
            vals = [(d[(d.grup == g)][t] == c).mean() * 100 for g in "ABC"]
            ax.bar(np.arange(3) + (j - (len(cls) - 1) / 2) * w, vals, w * .92, color=pal[j], label=c, zorder=3)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{g}\n(n={int((d.grup == g).sum())})" for g in "ABC"])
        ax.set_ylabel("Video oranı (%)", color=MUTED)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, ncol=2)
    fig.suptitle("Kayıt günü grubuna göre sınıf dağılımı (A: 18 Haz, B: 22 Haz, C: diğer)", fontsize=10, color=MUTED, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / "fig_etiket_dagilimi.png", dpi=200)
    plt.close(fig)


def fig_kappa():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = Path("outputs/agreement_irem_afra.txt")
    if not p.exists():
        return
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) == 4 and parts[0] in {"kopek_sayisi", "gorunurluk", "kuyruk_konum", "kuyruk_hareket", "kulak", "durus",
                                             "hareket", "ses", "degisken", "guvensiz"}:
            rows.append((parts[0], float(parts[2])))
    if not rows:
        return
    names = {"kopek_sayisi": "Köpek sayısı", "gorunurluk": "Görünürlük", "kuyruk_konum": "Kuyruk konumu",
             "kuyruk_hareket": "Kuyruk hareketi", "kulak": "Kulak", "durus": "Duruş", "hareket": "Hareket", "ses": "Ses",
             "degisken": "Davranış değişiyor", "guvensiz": "Emin değilim"}
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.8, 3.8), facecolor="white")
    style(ax)
    ax.yaxis.grid(False); ax.xaxis.grid(True, color=GRID, lw=.8)
    ax.barh([names[r[0]] for r in rows], [r[1] for r in rows], color="#2a78d6", height=.62, zorder=3)
    for i, r in enumerate(rows):
        ax.text(max(r[1], 0) + .015, i, f"{r[1]:.2f}", va="center", fontsize=8.5, color=INK)
    for x in (0.4, 0.6, 0.8):
        ax.axvline(x, color="#bdbcb5", lw=.7, ls=":", zorder=1)
    ax.set_xlim(-.05, 1)
    ax.set_xlabel("Cohen kappa (iki etiketleyici, 127 video; çözüm öncesi)", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_kappa.png", dpi=200)
    plt.close(fig)


def fmt_table(M, protocol):
    d = M[M.protokol == protocol]
    models = ["baz_cogunluk"] + (["gun_bilgisi"] if protocol == "rastgele" else []) + \
             [f"{f}_{c}" for f in FEATS for c in CLFS]
    lines = ["| Hedef | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for t in TARGETS:
        cells = []
        for m in models:
            r = d[(d.hedef == t) & (d.model == m)].iloc[0]
            cells.append(f"{r.macro_f1:.2f} [{r.ci_lo:.2f}-{r.ci_hi:.2f}] / acc {r.acc:.2f}")
        lines.append(f"| {t} (n={int(d[d.hedef == t].n.iloc[0])}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_final(df, B):
    import joblib
    out = {}
    for t in TARGETS:
        y = df[t].astype(str).values
        idx = np.where(y != "")[0]
        a = make_clf("LR").fit(B["aud"][idx], y[idx])
        v = make_clf("LR").fit(B["vm"][idx], y[idx])
        out[t] = dict(aud=a, vm=v, classes=list(a.classes_), n=len(idx))
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(out, "outputs/models/final_models.joblib")
    print("Kaydedildi: outputs/models/final_models.joblib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-final", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    z = np.load("outputs/features_all.npz", allow_pickle=False)
    tg = pd.read_csv("outputs/targets_and_splits.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    pos = {v: i for i, v in enumerate(z["video"])}
    df = tg[tg.video.isin(pos)].reset_index(drop=True)
    order = [pos[v] for v in df.video]
    df["rastgele_fold"] = df.rastgele_fold.astype(int)
    B = dict(aud=z["aud"][order], vm=np.hstack([z["vis"][order], z["mot"][order]]))
    print(f"{len(df)} video, ses={B['aud'].shape}, görüntü+hareket={B['vm'].shape}")

    if args.save_final:
        save_final(df, B)
        return

    M, PF, P, cms = run_all(df, B)
    M.to_csv(OUT / "metrics.csv", index=False)
    PF.to_csv(OUT / "per_fold.csv", index=False)
    P.to_csv(OUT / "predictions.csv", index=False)
    for (protocol, t), (labels, cm) in cms.items():
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(OUT / f"confusion_{protocol}_{t}.csv")

    fig_compare(M); fig_confusion(cms); fig_distribution(df); fig_kappa()

    prim = M[M.model == PRIMARY].pivot(index="hedef", columns="protokol", values="macro_f1")
    prim["fark_(rastgele-oturum)"] = prim["rastgele"] - prim["oturum"]
    txt = ["# Sonuçlar (otomatik üretildi)", "", f"Birincil model: {PRIMARY}. Hücre: macro-F1 [%95 aralık] / doğruluk.", "",
           "## Oturuma göre bölme (asıl sonuç)", fmt_table(M, "oturum"), "",
           "## Rastgele bölme (karşılaştırma)", fmt_table(M, "rastgele"), "",
           "## Birincil model: rastgele - oturum macro-F1 farkı (sızıntı şişmesi)", prim.round(3).to_string(), "",
           "## Fold bazında macro-F1 (oturum, birincil model)",
           PF[(PF.protokol == "oturum") & (PF.model == PRIMARY)].round(3).to_string(index=False)]
    (OUT / "summary.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt[:30]))
    print(f"\nYazıldı: {OUT}/ (metrics.csv, summary.md, fig_*.png ...)")


if __name__ == "__main__":
    main()
