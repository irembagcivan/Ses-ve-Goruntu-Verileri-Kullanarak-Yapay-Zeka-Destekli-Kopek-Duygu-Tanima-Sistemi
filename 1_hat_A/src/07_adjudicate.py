"""
Aşama 3e - Uyuşmazlıkları birlikte çözüp NİHAİ etiket setini üretmek
--------------------------------------------------------------------
İki adım:

  1) Tablo üret (iki etiket dosyası değişince tekrar çalıştırılabilir):
       python src/07_adjudicate.py --a irem --b afra
     -> labeling/uyusmazlik.xlsx  (Excel/Numbers ile açın; 'Karar' sayfası)
        'final' sütununda her hücrede açılır liste vardır: i (irem), a (afra) ya da seçenek adı.
        Karar verilecek sorular: ses, durus, hareket, kuyruk_hareket (grup 1) ve kopek_sayisi (grup 2).
        'kulak' ve 'kuyruk_konum' tartışılmaz: yalnızca iki kişinin AYNI dediği videolar kullanılır
        (uyuşmazlıkta o hedef için video dışarıda kalır, "konsensüs-only").
        Tabloyu yeniden üretmek mevcut 'final' girişlerinizi korur (aynı video+soru için).

  2) Birleştir:
       python src/07_adjudicate.py --a irem --b afra --merge
     -> labeling/labels_final.csv  + outputs/final_label_summary.txt

Otomatik kurallar (tartışma gerektirmez):
  * İki kişi aynı etiketi verdiyse o etiket nihaidir.
  * 'gorunurluk': iki değerden KÖTÜ olanı alınır (iyi < kismi < kotu). Bu soru model hedefi değil,
    filtre olarak kullanılacak; eşik farkı yüzünden birebir uyum beklenmez.
  * 'degisken' ve 'guvensiz': herhangi biri 1 dediyse 1.
  * Tek kişinin tamamladığı videolarda o kişinin etiketi geçici olarak alınır ve 'tek_kisi' ile işaretlenir.
"""
import argparse
from pathlib import Path

import pandas as pd

import schema

CORE = ["ses", "durus", "hareket", "kuyruk_hareket"]
ADJ = CORE + ["kopek_sayisi"]            # birlikte karar verilecek sorular
SECONDARY = ["kuyruk_konum", "kulak"]     # sadece konsensüs
PRIORITY = ADJ + SECONDARY
SEVERITY = {"iyi": 0, "kismi": 1, "kotu": 2}


def load(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig").set_index("video")


def write_xlsx(sheet, path, common, A, B, ndiff, ia, ib):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "Karar"
    ws.append(list(sheet.columns))
    for r in sheet.itertuples(index=False):
        ws.append(list(r))
    head = PatternFill("solid", fgColor="DDDDDD")
    yellow = PatternFill("solid", fgColor="FFF2A8")
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = head
    fcol = list(sheet.columns).index("final") + 1
    allowed = {f["id"]: schema.allowed_values(f) for f in schema.RADIO_FIELDS}
    for i, q in enumerate(sheet.soru, start=2):
        ws.cell(row=i, column=fcol).fill = yellow
        dv = DataValidation(type="list", formula1='"' + ",".join(["i", "a"] + list(allowed[q])) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(ws.cell(row=i, column=fcol))
    widths = {"grup": 6, "video": 16, "fark_sayisi": 11, "soru": 16, ia: 18, ib: 18, "final": 20}
    for j, name in enumerate(sheet.columns, start=1):
        ws.column_dimensions[get_column_letter(j)].width = widths.get(name, 40)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions

    ws2 = wb.create_sheet("Izleme gerekmeyen")
    ws2.append(["video", "neden"])
    for v in common:
        if ndiff[v] == 0:
            ws2.append([v, "iki kişi tüm sorularda aynı"])
        elif not any(A.loc[v, q] != B.loc[v, q] for q in ADJ):
            diffs = [q for q in SECONDARY if A.loc[v, q] != B.loc[v, q]]
            ws2.append([v, "karar sorularında aynı; sadece " + ", ".join(diffs) + " farklı (konsensüs-only)"])
    for c in ws2[1]:
        c.font = Font(bold=True)
        c.fill = head
    ws2.column_dimensions["A"].width = 18
    ws2.column_dimensions["B"].width = 70
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--labeling-dir", default=Path("labeling"), type=Path)
    ap.add_argument("--out-dir", default=Path("outputs"), type=Path)
    args = ap.parse_args()

    A = load(args.labeling_dir / f"labels_{args.a}.csv")
    B = load(args.labeling_dir / f"labels_{args.b}.csv")
    req = [f["id"] for f in schema.RADIO_FIELDS]
    ok_a = set(A[(A[req] != "").all(axis=1)].index)
    ok_b = set(B[(B[req] != "").all(axis=1)].index)
    common = sorted(ok_a & ok_b)
    only_one = sorted((ok_a | ok_b) - set(common))
    none = sorted(set(A.index) | set(B.index) - ok_a - ok_b)
    none = [v for v in none if v not in ok_a and v not in ok_b]

    sheet_path = args.labeling_dir / "uyusmazlik.xlsx"
    old_csv = args.labeling_dir / "uyusmazlik.csv"
    ia, ib = args.a, args.b

    def read_prev():
        prev = {}
        for pth in (sheet_path, old_csv):
            if pth.exists():
                df = (pd.read_excel(pth, sheet_name="Karar", dtype=str).fillna("") if pth.suffix == ".xlsx"
                      else pd.read_csv(pth, dtype=str, keep_default_na=False, encoding="utf-8-sig"))
                for r in df.itertuples(index=False):
                    d = dict(zip(df.columns, r))
                    if str(d.get("final", "")).strip():
                        prev[(d["video"], d["soru"])] = str(d["final"]).strip()
        return prev

    # ---------- 1) tablo ----------
    ndiff = {v: sum(A.loc[v, q] != B.loc[v, q] for q in PRIORITY) for v in common}
    rows = []
    for v in common:
        for q in ADJ:
            if A.loc[v, q] != B.loc[v, q]:
                rows.append(dict(grup=1 if q in CORE else 2, video=v, fark_sayisi=ndiff[v], soru=q,
                                 **{ia: A.loc[v, q], ib: B.loc[v, q]}, final="",
                                 **{f"{ia}_not": A.loc[v, "not"], f"{ib}_not": B.loc[v, "not"]}))
    cols = ["grup", "video", "fark_sayisi", "soru", ia, ib, "final", f"{ia}_not", f"{ib}_not"]
    sheet = pd.DataFrame(rows, columns=cols)

    if not args.merge:
        prev = read_prev()
        sheet["final"] = [prev.get((v, q), "") for v, q in zip(sheet.video, sheet.soru)]
        sheet = sheet.sort_values(["grup", "fark_sayisi", "video"], ascending=[True, False, True]).reset_index(drop=True)
        write_xlsx(sheet, sheet_path, common, A, B, ndiff, ia, ib)
        n1 = sheet[sheet.grup == 1]
        n2 = sheet[sheet.grup == 2]
        print(f"Grup 1 (ses/duruş/hareket/kuyruk hareketi): {len(n1)} satır, {n1.video.nunique()} video")
        print(f"Grup 2 (köpek sayısı)                    : {len(n2)} satır, {n2.video.nunique()} video")
        print(f"Tablo: {sheet_path}")
        if only_one:
            print("UYARI - sadece bir kişinin tamamladığı videolar:", only_one)
        return

    # ---------- 2) birleştir ----------
    if not sheet_path.exists() and not old_csv.exists():
        raise SystemExit("Önce tabloyu üretin (--merge olmadan çalıştırın) ve 'final' sütununu doldurun.")
    dec = pd.DataFrame([dict(video=v, soru=q, **{ia: "", ib: ""}, final=f) for (v, q), f in read_prev().items()])
    if dec.empty:
        dec = pd.DataFrame(columns=["video", "soru", "final", ia, ib])
    # a/i kısayolları için irem/afra değerleri
    dec[ia] = [A.loc[v, q] if v in A.index else "" for v, q in zip(dec.video, dec.soru)]
    dec[ib] = [B.loc[v, q] if v in B.index else "" for v, q in zip(dec.video, dec.soru)]
    allowed = {f["id"]: schema.allowed_values(f) for f in schema.RADIO_FIELDS}
    decision = {}
    bad = []
    for r in dec.itertuples(index=False):
        d = dict(zip(dec.columns, r))
        fin = d["final"].strip()
        if not fin:
            continue
        v, q = d["video"], d["soru"]
        if fin.lower() == "i":
            fin = d[args.a]
        elif fin.lower() == "a":
            fin = d[args.b]
        if fin not in allowed[q]:
            bad.append((v, q, fin))
            continue
        decision[(v, q)] = fin
    if bad:
        print("GEÇERSİZ 'final' değerleri (düzeltin):")
        for b in bad:
            print("  ", b)
        raise SystemExit(1)

    out, pending = [], []
    for v in sorted(set(A.index) | set(B.index)):
        row = {"video": v, "durum": ""}
        if v in common:
            for q in req:
                a, b = A.loc[v, q], B.loc[v, q]
                if a == b:
                    row[q] = a
                elif q == "gorunurluk":
                    row[q] = a if SEVERITY[a] >= SEVERITY[b] else b
                elif (v, q) in decision:
                    row[q] = decision[(v, q)]
                elif q in SECONDARY:
                    row[q] = ""            # konsensüs-only: bu hedef için video dışarıda
                else:
                    row[q] = ""
                    pending.append((v, q))
            row["durum"] = "tam" if not any(p[0] == v for p in pending) else "karar_bekliyor"
            row["konsensus_disi"] = ",".join(q for q in SECONDARY if row[q] == "")
        elif v in ok_a or v in ok_b:
            src = A if v in ok_a else B
            for q in req:
                row[q] = src.loc[v, q]
            row["durum"] = "tek_kisi"
        else:
            for q in req:
                row[q] = ""
            row["durum"] = "etiketsiz"
        row.setdefault("konsensus_disi", "")
        for q in ("degisken", "guvensiz"):
            vals = [X.loc[v, q] for X in (A, B) if v in X.index and X.loc[v, q] != ""]
            row[q] = "1" if "1" in vals else ("0" if vals else "")
        row["not"] = " | ".join(x for x in (
            f"{args.a}: {A.loc[v,'not']}" if v in A.index and A.loc[v, "not"] else "",
            f"{args.b}: {B.loc[v,'not']}" if v in B.index and B.loc[v, "not"] else "") if x)
        out.append(row)

    cols = ["video", "durum"] + req + ["konsensus_disi", "degisken", "guvensiz", "not"]
    final = pd.DataFrame(out)[cols]
    final.to_csv(args.labeling_dir / "labels_final.csv", index=False, encoding="utf-8-sig")

    lines = [f"Nihai etiket seti: {len(final)} video", str(final.durum.value_counts().to_dict()),
             f"Karar bekleyen (video,soru): {len(pending)}"]
    done = final[final.durum.isin(["tam", "tek_kisi"])]
    for q in req:
        lines.append(f"\n[{q}] " + str(done[q].value_counts().to_dict()))
    args.out_dir.mkdir(exist_ok=True)
    (args.out_dir / "final_label_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nYazıldı: labeling/labels_final.csv")


if __name__ == "__main__":
    main()
