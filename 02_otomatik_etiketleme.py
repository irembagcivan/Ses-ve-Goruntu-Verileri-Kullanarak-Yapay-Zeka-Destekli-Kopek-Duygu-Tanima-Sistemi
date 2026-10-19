import os
# Mac M1/M2 Conda ortamında Segmentation Fault (bellek çakışması) hatasını önleyen ayarlar:
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ÖNCE PyTorch import edilmelidir (OpenCV ile çakışmaması için)
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchaudio

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

import matplotlib
matplotlib.use("Agg")  # Mac ekran kartı arayüz çakışmasını önler
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.set_num_threads(2)

# Video okuma (OpenCV) ile Metal GPU çakışmasını önlemek için bu aşamada M2 CPU kullanıyoruz
cihaz = torch.device("cpu")
print("1. Kütüphaneler güvenli modda yüklendi. Model hazırlanıyor...", flush=True)

VERI_KLASORU = Path(__file__).parent
ORNEK_KARE_SAYISI = 6   # Her videodan eşit aralıklı alınacak kare sayısı
KUME_SAYISI = 4         # 4 Duygu Sınıfı: Happy, Sad, Angry, Relaxed

# 2. GÖRSEL ÖZNİTELİK ÇIKARICI (Hızlı ve hafif ResNet18 modeli)
agirliklar = models.ResNet18_Weights.DEFAULT
resnet_modeli = models.resnet18(weights=agirliklar)
# Son sınıflandırma katmanını çıkarıp 512 boyutlu öznitelik vektörü alıyoruz
resnet_modeli = nn.Sequential(*list(resnet_modeli.children())[:-1])
resnet_modeli = resnet_modeli.to(cihaz)
resnet_modeli.eval()

print("2. ResNet18 modeli başarıyla yüklendi!", flush=True)

goruntu_donusumu = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def gorsel_oznitelik_cikar(video_yolu):
    """Videodan eşit aralıklı kareler alıp ResNet18 ile 512 boyutlu öznitelik çıkarır."""
    okuyucu = cv2.VideoCapture(str(video_yolu))
    toplam_kare = int(okuyucu.get(cv2.CAP_PROP_FRAME_COUNT))

    if toplam_kare <= 0:
        okuyucu.release()
        return np.zeros(512), None

    kare_indeksleri = np.linspace(0, max(0, toplam_kare - 1), ORNEK_KARE_SAYISI, dtype=int)
    kare_tensorleri = []
    orta_kare_goruntusu = None

    for idx in kare_indeksleri:
        okuyucu.set(cv2.CAP_PROP_POS_FRAMES, idx)
        basarili, kare = okuyucu.read()
        if basarili and kare is not None:
            kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
            if orta_kare_goruntusu is None and idx >= toplam_kare // 3:
                orta_kare_goruntusu = cv2.resize(kare_rgb, (224, 224))
            kare_tensorleri.append(goruntu_donusumu(kare_rgb))

    okuyucu.release()

    if not kare_tensorleri:
        return np.zeros(512), None

    batch = torch.stack(kare_tensorleri).to(cihaz)
    with torch.no_grad():
        oznitelikler = resnet_modeli(batch).squeeze(-1).squeeze(-1)
        if oznitelikler.ndim == 1:
            ortalama_oznitelik = oznitelikler.cpu().numpy()
        else:
            ortalama_oznitelik = oznitelikler.mean(dim=0).cpu().numpy()

    return ortalama_oznitelik, orta_kare_goruntusu


def ses_ozniteligi_cikar(video_yolu):
    """Torchaudio ile videodan ses okuyup MFCC ve enerji özniteliklerini çıkarır."""
    try:
        dalga_formu, frekans = torchaudio.load(str(video_yolu))
        # Stereo ise tek kanala (mono) düşür
        if dalga_formu.shape[0] > 1:
            dalga_formu = torch.mean(dalga_formu, dim=0, keepdim=True)

        # İlk 10 saniyeyi al
        maks_ornek = frekans * 10
        if dalga_formu.shape[1] > maks_ornek:
            dalga_formu = dalga_formu[:, :maks_ornek]

        if dalga_formu.shape[1] < 1600:
            return np.zeros(27)

        mfcc_donusturucu = torchaudio.transforms.MFCC(
            sample_rate=frekans,
            n_mfcc=13,
            melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 40}
        )
        mfcc = mfcc_donusturucu(dalga_formu).squeeze(0).numpy()
        mfcc_ort = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        rms_enerji = float(torch.sqrt(torch.mean(dalga_formu ** 2)).item())

        return np.concatenate([mfcc_ort, mfcc_std, [rms_enerji]])
    except Exception:
        return np.zeros(27)


# 3. TÜM VİDEOLARI TARAMA VE ÖZNİTELİK BİRLEŞTİRME
video_listesi = sorted([
    f for f in VERI_KLASORU.iterdir()
    if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in [".mov", ".mp4"]
])

print(f"3. Toplam {len(video_listesi)} video bulundu. Öznitelikler çıkarılıyor...\n", flush=True)

birlesik_oznitelikler = []
dosya_isimleri = []
onizleme_resimleri = []
ses_basarili_sayisi = 0

for video in tqdm(video_listesi, desc="Videolar İşleniyor"):
    gorsel_vektor, ornek_resim = gorsel_oznitelik_cikar(video)
    ses_vektoru = ses_ozniteligi_cikar(video)

    if np.any(ses_vektoru):
        ses_basarili_sayisi += 1

    toplam_vektor = np.concatenate([gorsel_vektor, ses_vektoru])
    birlesik_oznitelikler.append(toplam_vektor)
    dosya_isimleri.append(video.name)
    onizleme_resimleri.append(ornek_resim)

print(f"\nSes kanalı okunan video sayısı: {ses_basarili_sayisi} / {len(video_listesi)}", flush=True)

# 4. BOYUT İNDİRGEME (PCA) VE KÜMELEME (K-MEANS)
X = np.nan_to_num(np.array(birlesik_oznitelikler))
olcekleyici = StandardScaler()
X_olcekli = olcekleyici.fit_transform(X)

pca = PCA(n_components=min(15, len(video_listesi) - 1), random_state=42)
X_pca = pca.fit_transform(X_olcekli)

kmeans = KMeans(n_clusters=KUME_SAYISI, random_state=42, n_init=20)
kume_etiketleri = kmeans.fit_predict(X_pca)

skor = silhouette_score(X_pca, kume_etiketleri)
print(f"Kümeleme Tamamlandı! Silhouette Kalite Skoru: {skor:.3f}\n", flush=True)

# 5. SONUÇLARI TABLOYA KAYDETME
sonuc_tablosu = pd.DataFrame({
    "dosya_adi": dosya_isimleri,
    "kume_no": kume_etiketleri,
    "duygu_etiketi": [f"Kume_{k}" for k in kume_etiketleri],
})

print("--- KÜMELERE GÖRE VİDEO DAĞILIMI ---")
print(sonuc_tablosu["duygu_etiketi"].value_counts())

csv_yolu = VERI_KLASORU / "otomatik_etiketler.csv"
sonuc_tablosu.to_csv(csv_yolu, index=False)
print(f"\nEtiketler '{csv_yolu.name}' dosyasına kaydedildi.")

# 6. HER KÜME İÇİN ÖNİZLEME KOLAJI OLUŞTURMA
fig, eksenler = plt.subplots(KUME_SAYISI, 6, figsize=(15, 10))
fig.suptitle("Kümelere Ait Örnek Köpek Kareleri (Hangi Küme Hangi Duygu?)", fontsize=16)

for kume_id in range(KUME_SAYISI):
    kume_indeksleri = np.where(kume_etiketleri == kume_id)[0]
    secilenler = kume_indeksleri[:6]

    for sutun in range(6):
        ax = eksenler[kume_id, sutun]
        ax.axis("off")
        if sutun < len(secilenler):
            vid_idx = secilenler[sutun]
            if onizleme_resimleri[vid_idx] is not None:
                ax.imshow(onizleme_resimleri[vid_idx])
                ax.set_title(f"Küme {kume_id}\n{dosya_isimleri[vid_idx]}", fontsize=9)

plt.tight_layout()
resim_yolu = VERI_KLASORU / "kume_onizleme_tablosu.jpg"
plt.savefig(resim_yolu, dpi=150)
print(f"Görsel özet '{resim_yolu.name}' olarak kaydedildi!")