"""
Aşama 4 - Hedefler ve veri bölme (sızıntısız)
---------------------------------------------
Girdi : labeling/labels_final.csv, labeling/duygu_final.csv, outputs/inventory.csv
Çıktı : outputs/targets_and_splits.csv  (her video: model hedefleri + oturum grubu + rastgele fold)
        outputs/split_report.txt        (grup boyutları ve sınıf dağılımları)

Hedefler (nadir sınıflar birleştirildi):
  ses3     : yok / havlama / diger (inleme + hirlama + diger)
  durus3   : ayakta / yatiyor / diger (oturuyor + sirtustu)
  hareket3 : hareketsiz / yuruyor / kosuyor_oynuyor
  kuyruk2  : salliyor / sallamiyor   (gorunmuyor -> dışarıda)
  duygu4   : Happy / Sad / Angry / Relaxed  (İNSAN etiketi, duygu_final.csv; boş = belirsiz, dışarıda)

Oturum grubu (kayıt günü): A = 18 Haz, B = 22 Haz, C = diğerleri (20 Haz + tarihsiz dog*.mov).
Değerlendirme: grup-dışarıda-bırak (leave-one-session-out) + karşılaştırma için rastgele 5-fold.

Çalıştırma: python src/08_make_splits.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

LAB = Path("labeling")
OUT = Path("outputs")

fin = pd.read_csv(LAB / "labels_final.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
inv = pd.read_csv(OUT / "inventory.csv")
emo = pd.read_csv(LAB / "duygu_final.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")  # İNSAN etiketi (2 etiketleyici + ortak karar)

inv["gun"] = pd.to_datetime(inv.creation_time, errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
df = fin.merge(inv[["file", "gun", "duration_s", "has_audio", "audio_flat_flag"]], left_on="video", right_on="file", how="left")
df = df.merge(emo[["video", "duygu_insan"]].rename(columns={"duygu_insan": "duygu4"}), on="video", how="left")
df["duygu4"] = df["duygu4"].fillna("")


def grp(g):
    return {"2026-06-18": "A", "2026-06-22": "B"}.get(g, "C")


df["grup"] = df.gun.map(grp)
df["ses3"] = df.ses.map({"yok": "yok", "havlama": "havlama"}).fillna("diger").where(df.ses != "", "")
df["durus3"] = df.durus.map({"ayakta": "ayakta", "yatiyor": "yatiyor"}).fillna("diger").where(df.durus != "", "")
df["hareket3"] = df.hareket
df["kuyruk2"] = df.kuyruk_hareket.where(df.kuyruk_hareket.isin(["salliyor", "sallamiyor"]), "")

rng = np.random.default_rng(42)
df["rastgele_fold"] = rng.permutation(np.arange(len(df)) % 5)

keep = ["video", "grup", "gun", "rastgele_fold", "ses3", "durus3", "hareket3", "kuyruk2", "duygu4",
        "kopek_sayisi", "gorunurluk", "degisken", "guvensiz", "duration_s", "has_audio", "audio_flat_flag"]
df[keep].to_csv(OUT / "targets_and_splits.csv", index=False, encoding="utf-8-sig")

lines = [f"Toplam video: {len(df)}", "Grup boyutları: " + str(df.grup.value_counts().sort_index().to_dict()), ""]
for t in ["ses3", "durus3", "hareket3", "kuyruk2", "duygu4"]:
    ct = pd.crosstab(df[t].replace("", "(dışarıda)"), df.grup)
    ct["toplam"] = ct.sum(axis=1)
    lines += [f"[{t}]", ct.to_string(), ""]
    for g in "ABC":
        test_cls = set(df[(df.grup == g) & (df[t] != "")][t])
        train_cls = set(df[(df.grup != g) & (df[t] != "")][t])
        miss = test_cls - train_cls
        if miss:
            lines.append(f"  UYARI: {t}, test={g}: eğitimde hiç olmayan sınıf {sorted(miss)}")
    lines.append("")
txt = "\n".join(lines)
(OUT / "split_report.txt").write_text(txt, encoding="utf-8")
print(txt)
