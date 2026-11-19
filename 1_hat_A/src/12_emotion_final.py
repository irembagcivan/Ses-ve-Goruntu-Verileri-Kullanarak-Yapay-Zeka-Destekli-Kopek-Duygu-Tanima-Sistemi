"""
Aşama 3f - Duygu etiketleri: uyum ölçümü ve nihai (insan) etiket seti
--------------------------------------------------------------------
Girdi : labeling/labels_duygu_<a>.csv, labeling/labels_duygu_<b>.csv   (04_label_tool.py --stage duygu)
Adım 1: python src/12_emotion_final.py --a irem --b afra
          -> uyum (yüzde, Cohen kappa, karışıklık) outputs/duygu_uyum.txt
          -> labeling/uyusmazlik_duygu.xlsx  ('final' sütununu doldurun: i / a / sakin / mutlu / kizgin / uzgun / belirsiz)
Adım 2: python src/12_emotion_final.py --a irem --b afra --merge
          -> labeling/duygu_final.csv  (video, duygu_insan [Relaxed/Happy/Angry/Sad ya da boş=belirsiz], durum)
          -> outputs/duygu_final_ozet.txt  (sınıf dağılımı)
Aynı etiketi verdikleri videolar otomatik nihaidir. 'belirsiz' nihai etiket olursa video duygu hedefinden çıkar.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

MAP = {"sakin": "Relaxed", "mutlu": "Happy", "kizgin": "Angry", "uzgun": "Sad", "belirsiz": ""}
OPTS = ["sakin", "mutlu", "kizgin", "uzgun", "belirsiz"]
LAB = Path("labeling")
OUT = Path("outputs")


def load(p):
    d = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig").set_index("video")
    return d[d.duygu != ""]


def sheet_xlsx(df, path, ia, ib):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook(); ws = wb.active; ws.title = "Karar"
    ws.append(list(df.columns))
    for r in df.itertuples(index=False):
        ws.append(list(r))
    for c in ws[1]:
        c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="DDDDDD")
    fcol = list(df.columns).index("final") + 1
    dv = DataValidation(type="list", formula1='"' + ",".join(["i", "a"] + OPTS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    for i in range(2, len(df) + 2):
        ws.cell(row=i, column=fcol).fill = PatternFill("solid", fgColor="FFF2A8")
        dv.add(ws.cell(row=i, column=fcol))
    for j, n in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(j)].width = {"video": 16, "final": 16}.get(n, 22 if "not" not in n else 40)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    ia, ib = args.a, args.b
    A = load(LAB / f"labels_duygu_{ia}.csv"); B = load(LAB / f"labels_duygu_{ib}.csv")
    common = sorted(set(A.index) & set(B.index))
    print(f"{ia}: {len(A)} video, {ib}: {len(B)} video, ortak: {len(common)}")
    sheet = LAB / "uyusmazlik_duygu.xlsx"

    if not args.merge:
        a, b = A.loc[common, "duygu"], B.loc[common, "duygu"]
        lines = [f"Duygu etiketi uyumu ({ia} - {ib}), ortak video: {len(common)}",
                 f"Yüzde uyum: {100 * (a == b).mean():.1f}",
                 f"Cohen kappa (belirsiz dahil, {len(OPTS)} seçenek): {cohen_kappa_score(a, b):.3f}"]
        keep = [v for v in common if A.loc[v, "duygu"] != "belirsiz" and B.loc[v, "duygu"] != "belirsiz"]
        if len(keep) > 1:
            lines.append(f"Cohen kappa (ikisi de 'belirsiz' dememiş {len(keep)} video): "
                         f"{cohen_kappa_score(A.loc[keep, 'duygu'], B.loc[keep, 'duygu']):.3f}")
        lines.append(f"'belirsiz' sayısı: {ia}={(A.duygu == 'belirsiz').sum()}, {ib}={(B.duygu == 'belirsiz').sum()}")
        cm = pd.crosstab(a.rename(ia), b.rename(ib)).reindex(index=OPTS, columns=OPTS, fill_value=0)
        lines += ["", f"Karışıklık (satır={ia}, sütun={ib}):", cm.to_string()]
        lines += ["", f"Dağılım {ia}: {A.duygu.value_counts().to_dict()}", f"Dağılım {ib}: {B.duygu.value_counts().to_dict()}"]
        if "guven" in A.columns:
            lines.append(f"Güven ({ia}): {A.guven.value_counts().to_dict()}  ({ib}): {B.guven.value_counts().to_dict()}")
        txt = "\n".join(lines)
        OUT.mkdir(exist_ok=True); (OUT / "duygu_uyum.txt").write_text(txt, encoding="utf-8"); print(txt)
        dis = [v for v in common if A.loc[v, "duygu"] != B.loc[v, "duygu"]]
        prev = {}
        if sheet.exists():
            old = pd.read_excel(sheet, sheet_name="Karar", dtype=str).fillna("")
            prev = dict(zip(old.video, old.final))
        df = pd.DataFrame({"video": dis, ia: A.loc[dis, "duygu"].values, ib: B.loc[dis, "duygu"].values,
                           "final": [prev.get(v, "") for v in dis],
                           f"{ia}_not": A.loc[dis, "not"].values if "not" in A else "",
                           f"{ib}_not": B.loc[dis, "not"].values if "not" in B else ""})
        sheet_xlsx(df, sheet, ia, ib)
        print(f"\nUyuşmazlık: {len(dis)} video -> {sheet}")
        return

    dec = {}
    if sheet.exists():
        d = pd.read_excel(sheet, sheet_name="Karar", dtype=str).fillna("")
        for r in d.itertuples(index=False):
            m = dict(zip(d.columns, r)); f = m["final"].strip().lower()
            if f == "i": f = m[ia]
            elif f == "a": f = m[ib]
            if f:
                if f not in OPTS:
                    raise SystemExit(f"Geçersiz final değeri: {m['video']} -> {f}")
                dec[m["video"]] = f
    rows, pend = [], []
    for v in common:
        a, b = A.loc[v, "duygu"], B.loc[v, "duygu"]
        if a == b: lab, st = a, "uyumlu"
        elif v in dec: lab, st = dec[v], "ortak_karar"
        else: lab, st = "", "karar_bekliyor"; pend.append(v)
        rows.append(dict(video=v, duygu_insan=MAP.get(lab, ""), durum=st if lab != "belirsiz" else st + "_belirsiz"))
    F = pd.DataFrame(rows)
    F.to_csv(LAB / "duygu_final.csv", index=False, encoding="utf-8-sig")
    lines = [f"Nihai duygu etiketi: {len(F)} video; karar bekleyen: {len(pend)}",
             "Durum: " + str(F.durum.value_counts().to_dict()),
             "Dağılım (boş=belirsiz/bekleyen): " + str(F.duygu_insan.replace('', '(boş)').value_counts().to_dict())]
    txt = "\n".join(lines); (OUT / "duygu_final_ozet.txt").write_text(txt, encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
