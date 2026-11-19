"""
Aşama 3c - Etiket özeti
-----------------------
Amaç : Etiketleme ilerlemesini ve sınıf dağılımını sayıyla görmek.
Neden: Hangi sınıf kaç video içeriyor, eksik/emin olunmayan video var mı, sınıf dengesi
       nasıl? Bunlar Aşama 4'te (bölme stratejisi) ve metrik seçiminde belirleyici.

Çalıştırma:
    python src/05_label_summary.py --annotator irem
Çıktı: ekrana yazılır ve outputs/label_summary.txt dosyasına kaydedilir.
"""
import argparse
from pathlib import Path

import pandas as pd

import schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--labeling-dir", default=Path("labeling"), type=Path)
    ap.add_argument("--inventory", default=Path("outputs/inventory.csv"), type=Path)
    ap.add_argument("--out-dir", default=Path("outputs"), type=Path)
    args = ap.parse_args()

    path = args.labeling_dir / f"labels_{args.annotator}.csv"
    if not path.exists():
        raise SystemExit(f"{path} yok. Önce etiketleme aracını kullanın.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    inv = pd.read_csv(args.inventory)
    total = len(inv)

    L = [f"Etiketleyici: {args.annotator} | dosya: {path}"]
    complete = df[df.apply(lambda r: schema.is_complete(r), axis=1)]
    partial = df[~df.index.isin(complete.index)]
    L.append(f"Toplam video: {total} | tamamlanan: {len(complete)} | yarım: {len(partial)} | "
             f"hiç açılmayan: {total - len(df)}")
    if len(partial):
        L.append("Yarım kalanlar: " + ", ".join(partial.video))
    missing = sorted(set(inv.file) - set(df.video))
    if missing:
        L.append(f"Hiç etiketlenmeyen ({len(missing)}): " + ", ".join(missing[:40]) +
                 (" ..." if len(missing) > 40 else ""))
    L.append(f"'Davranış değişiyor' işaretli: {(complete.degisken == '1').sum()} | "
             f"'Emin değilim' işaretli: {(complete.guvensiz == '1').sum()}")

    L.append("\n--- Sınıf dağılımı (yalnızca tamamlanan videolar) ---")
    for f in schema.RADIO_FIELDS:
        vc = complete[f["id"]].value_counts()
        parts = [f"{v}: {vc.get(v, 0)}" for v, _ in f["options"]]
        L.append(f"{f['id']:15s} " + " | ".join(parts))

    L.append("\n--- Bilgi amaçlı: 'görünmüyor' sayıları ---")
    for fid in ["kuyruk_konum", "kuyruk_hareket", "kulak"]:
        L.append(f"{fid}: {(complete[fid] == 'gorunmuyor').sum()} video")
    L.append("\nUyarı: En küçük sınıfta 10'dan az video varsa o sınıfın sonuçları çok belirsiz olur; "
             "raporda belirtilmelidir.")
    text = "\n".join(L)
    print(text)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "label_summary.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
