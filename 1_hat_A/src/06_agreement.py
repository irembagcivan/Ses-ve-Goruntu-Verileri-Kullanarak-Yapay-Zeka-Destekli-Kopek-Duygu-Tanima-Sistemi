"""
Aşama 3d - Etiketleyici uyumu
-----------------------------
Amaç : İki kişinin bağımsız etiketlerinin ne kadar örtüştüğünü ölçmek.
Neden: Etiketler "gerçek" değil, insan algısıdır. İki kişi aynı videoya benzer etiket veriyorsa
       tanımlarımız net demektir. Anlaşılmayan sorular hem şemayı düzeltmemiz hem de
       raporda güvenilirliği göstermemiz için gerekli.

Metrikler:
  * Yüzde uyum   : aynı etiketi verdikleri video oranı. Sezgisel ama şans eseri uyumu saymaz.
  * Cohen's kappa: şans eseri uyumu düzeltir. Yaklaşık yorum (Landis & Koch): <0.20 zayıf,
                   0.21-0.40 orta altı, 0.41-0.60 orta, 0.61-0.80 iyi, >0.80 çok iyi.
  UYARI: Video sayısı azken (ör. 10) kappa çok oynaktır; sayıları kesin sonuç olarak yorumlamayın.
  Bir etiketleyici bir soruda hep aynı seçeneği işaretlediyse kappa tanımsız (nan) çıkabilir.

Çalıştırma:
    python src/06_agreement.py --a irem --b afra
Çıktı: ekrana yazılır; outputs/agreement_<a>_<b>.txt ve ..._disagreements.csv
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

import schema


def cohen_kappa(x, y):
    n = len(x)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / (n * n)
    if pe == 1:
        return float("nan")
    return (po - pe) / (1 - pe)


def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return df.set_index("video")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--labeling-dir", default=Path("labeling"), type=Path)
    ap.add_argument("--out-dir", default=Path("outputs"), type=Path)
    args = ap.parse_args()

    A = load(args.labeling_dir / f"labels_{args.a}.csv")
    B = load(args.labeling_dir / f"labels_{args.b}.csv")
    req = [f["id"] for f in schema.RADIO_FIELDS]
    ok_a = A[(A[req] != "").all(axis=1)].index
    ok_b = B[(B[req] != "").all(axis=1)].index
    common = sorted(set(ok_a) & set(ok_b))

    L = [f"Etiketleyiciler: {args.a} ve {args.b}",
         f"Her ikisinin de tamamladığı ortak video sayısı: {len(common)}"]
    if len(common) < 30:
        L.append("UYARI: 30'dan az ortak video var; kappa değerleri çok güvenilmez.")
    if not common:
        raise SystemExit("\n".join(L) + "\nOrtak tamamlanmış video yok.")

    rows, dis = [], []
    fields = [(f["id"], f["title"]) for f in schema.FIELDS if f["kind"] in ("radio", "check")]
    for fid, title in fields:
        x = [A.loc[v, fid] for v in common]
        y = [B.loc[v, fid] for v in common]
        agree = sum(a == b for a, b in zip(x, y))
        k = cohen_kappa(x, y)
        rows.append({"soru": fid, "uyum_%": round(100 * agree / len(common), 1),
                     "kappa": None if k != k else round(k, 2), "farkli_video": len(common) - agree})
        for v, a, b in zip(common, x, y):
            if a != b:
                dis.append({"video": v, "soru": fid, args.a: a, args.b: b,
                            f"not_{args.a}": A.loc[v, "not"], f"not_{args.b}": B.loc[v, "not"]})
    res = pd.DataFrame(rows)
    L.append("\n--- Soru bazında uyum ---")
    L.append(res.to_string(index=False))

    all_agree = [v for v in common if all(A.loc[v, f] == B.loc[v, f] for f, _ in fields if f in req)]
    L.append(f"\nTüm zorunlu sorularda tamamen aynı etiket veren video: {len(all_agree)} / {len(common)}")

    ddf = pd.DataFrame(dis)
    if len(ddf):
        L.append("\n--- En çok karışan seçenek çiftleri ---")
        for fid, _ in fields:
            sub = ddf[ddf.soru == fid]
            if len(sub):
                pairs = Counter(tuple(sorted((a, b))) for a, b in zip(sub[args.a], sub[args.b]))
                L.append(f"{fid}: " + "; ".join(f"{p[0]} <-> {p[1]} ({n})" for p, n in pairs.most_common(4)))
        L.append("\n--- Farklı etiket verilen videolar (kısa liste) ---")
        for v, g in ddf.groupby("video"):
            L.append(f"{v}: " + ", ".join(f"{r.soru} ({r[args.a]} vs {r[args.b]})" for _, r in g.iterrows()))
    text = "\n".join(L)
    print(text)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"agreement_{args.a}_{args.b}"
    (args.out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    ddf.to_csv(args.out_dir / f"{stem}_disagreements.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
