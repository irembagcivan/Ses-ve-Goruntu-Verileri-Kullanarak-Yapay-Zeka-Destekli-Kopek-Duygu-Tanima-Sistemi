import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from ultralytics import YOLO

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJE = Path(__file__).parent
GIRIS = PROJE / "hazir_veri_seti" / "goruntuler"
CIKIS = PROJE / "hazir_veri_seti" / "goruntuler_kirpik"
CSV = PROJE / "hazir_veri_seti" / "kirpma_kayitlari.csv"

GUVEN_ESIGI = 0.03     # çok düşük: köpek olduğunu biliyoruz
PAD_ORANI = 0.10
CIKTI_BOYUT = 224
MIN_ALAN_ORANI = 0.05  # kareyi %5'inden küçük kutular büyük ihtimalle yanlış

model = YOLO("yolov8x.pt")  # ilk seferde ~130 MB iner


def tuvale_yerlestir(parca):
    kh, kw = parca.shape[:2]
    olcek = CIKTI_BOYUT / max(kh, kw)
    parca = cv2.resize(parca, (max(1, int(kw * olcek)), max(1, int(kh * olcek))))
    tuval = np.full((CIKTI_BOYUT, CIKTI_BOYUT, 3), 114, dtype=np.uint8)
    oy = (CIKTI_BOYUT - parca.shape[0]) // 2
    ox = (CIKTI_BOYUT - parca.shape[1]) // 2
    tuval[oy:oy + parca.shape[0], ox:ox + parca.shape[1]] = parca
    return tuval


def kirp(goruntu, kutu):
    h, w = goruntu.shape[:2]
    x1, y1, x2, y2 = kutu
    pw, ph = (x2 - x1) * PAD_ORANI, (y2 - y1) * PAD_ORANI
    x1, y1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    x2, y2 = min(w, int(x2 + pw)), min(h, int(y2 + ph))
    parca = goruntu[y1:y2, x1:x2]
    return None if parca.size == 0 else tuvale_yerlestir(parca)


df = pd.read_csv(CSV)
hedefler = df.index[df["durum"] == "tam_goruntu"].tolist()
print(f"İkinci geçiş: {len(hedefler)} görüntü", flush=True)

for idx in tqdm(hedefler, desc="Tekrar deneniyor"):
    goreli = Path(df.at[idx, "dosya"])
    goruntu = cv2.imread(str(GIRIS / goreli))
    if goruntu is None:
        continue
    h, w = goruntu.shape[:2]

    # Sınıf filtresi yok; sadece insan (0) hariç tüm kutular
    sonuc = model.predict(goruntu, conf=GUVEN_ESIGI, imgsz=1280,
                          device="cpu", verbose=False)[0]
    if len(sonuc.boxes) == 0:
        continue

    kutular = sonuc.boxes.xyxy.cpu().numpy()
    guvenler = sonuc.boxes.conf.cpu().numpy()
    siniflar = sonuc.boxes.cls.cpu().numpy().astype(int)
    alanlar = (kutular[:, 2] - kutular[:, 0]) * (kutular[:, 3] - kutular[:, 1])

    gecerli = (siniflar != 0) & (alanlar / (h * w) >= MIN_ALAN_ORANI)
    if not gecerli.any():
        continue

    skor = np.where(gecerli, alanlar * guvenler, -1)
    i = int(skor.argmax())
    kirpik = kirp(goruntu, kutular[i])
    if kirpik is None:
        continue

    cv2.imwrite(str(CIKIS / goreli), kirpik)
    df.at[idx, "durum"] = "kirpildi_2gecis"
    df.at[idx, "tespit_sinifi"] = model.names[siniflar[i]]
    df.at[idx, "guven"] = round(float(guvenler[i]), 3)
    df.at[idx, "kutu_alan_orani"] = round(float(alanlar[i]) / (h * w), 3)

df.to_csv(CSV, index=False)
print("\n", df["durum"].value_counts(), sep="")
print("Hâlâ tam_goruntu kalanlar elle kontrol edilmeli (CSV'de 'durum' sütunu).")