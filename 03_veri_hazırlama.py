import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import subprocess
import tempfile
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 1. KLASÖR VE PARAMETRE AYARLARI
ANA_KLASOR = Path(__file__).parent
ETIKET_DOSYASI = ANA_KLASOR / "otomatik_etiketler.csv"
CIKTI_KLASORU = ANA_KLASOR / "hazir_veri_seti"

# Her videoyu 4 zaman penceresine bölerek veri sayısını 127'den 508'e çıkarıyoruz
PENCERE_SAYISI = 4

# kume_onizleme_tablosu.jpg analizimize göre güncellenmiş duygu eşleştirmesi:
KUME_DUYGU_HARITASI = {
    0: "Relaxed",  # Sırtüstü yatan / uzanan rahat köpekler
    1: "Angry",    # Sokakta ayakta gergin / tetikte duran köpekler
    2: "Sad",      # Barınakta kapalı bölmede duran köpekler
    3: "Happy"     # İnsanlarla ve diğer köpeklerle sosyalleşen köpekler
}

# Çıktı klasörlerini oluştur
(CIKTI_KLASORU / "goruntuler").mkdir(parents=True, exist_ok=True)
(CIKTI_KLASORU / "spektrogramlar").mkdir(parents=True, exist_ok=True)

ffmpeg_yolu = imageio_ffmpeg.get_ffmpeg_exe()
etiket_tablosu = pd.read_csv(ETIKET_DOSYASI)

# 2. DATA LEAKAGE ÖNLEME: VİDEO BAZINDA EĞİTİM (%80) VE TEST (%20) AYRIMI
egitim_videolari, test_videolari = train_test_split(
    etiket_tablosu,
    test_size=0.20,
    stratify=etiket_tablosu["kume_no"],
    random_state=42
)

egitim_seti_isimleri = set(egitim_videolari["dosya_adi"])
print(f"Eğitim Videoları: {len(egitim_videolari)} | Test Videoları: {len(test_videolari)}")
print(f"Her videodan {PENCERE_SAYISI} örnek çıkarılarak toplam {len(etiket_tablosu)*PENCERE_SAYISI} veri üretilecek...\n", flush=True)


def videodan_ses_cikar(video_yolu):
    """imageio-ffmpeg kullanarak .mov videosundan 16kHz mono ses sinyali çıkarır."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as gecici_wav:
        wav_yolu = gecici_wav.name

    try:
        komut = [
            ffmpeg_yolu, "-y", "-i", str(video_yolu),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            wav_yolu
        ]
        subprocess.run(komut, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        frekans, ses_verisi = wavfile.read(wav_yolu)
        ses_verisi = ses_verisi.astype(np.float32)
        maks_deger = np.max(np.abs(ses_verisi))
        if maks_deger > 0:
            ses_verisi = ses_verisi / maks_deger
        return ses_verisi, frekans
    except Exception:
        return np.zeros(16000 * 4, dtype=np.float32), 16000
    finally:
        if os.path.exists(wav_yolu):
            os.remove(wav_yolu)


def spektrogram_uret(ses_parcasi, frekans=16000):
    """Ses parçasını 224x224 boyutunda 3 kanallı Log-Spektrogram görüntüsüne çevirir."""
    if len(ses_parcasi) < 1600:
        ses_parcasi = np.pad(ses_parcasi, (0, 16000 - len(ses_parcasi)))

    _, _, Sxx = spectrogram(ses_parcasi, fs=frekans, nperseg=512, noverlap=256)
    log_spek = np.log10(Sxx + 1e-8)

    min_d, maks_d = log_spek.min(), log_spek.max()
    if maks_d - min_d > 1e-6:
        norm_spek = (log_spek - min_d) / (maks_d - min_d) * 255.0
    else:
        norm_spek = np.zeros_like(log_spek)

    spek_resim = cv2.resize(norm_spek.astype(np.uint8), (224, 224))
    spek_renkli = cv2.applyColorMap(spek_resim, cv2.COLORMAP_VIRIDIS)
    return spek_renkli


# 3. TÜM VİDEOLARI PENCERELERE BÖLÜP KAYDETME
veri_kayitlari = []
ses_okunan_video = 0

for _, satir in tqdm(etiket_tablosu.iterrows(), total=len(etiket_tablosu), desc="Veri Seti Hazırlanıyor"):
    dosya_adi = satir["dosya_adi"]
    kume_no = int(satir["kume_no"])
    duygu_adi = KUME_DUYGU_HARITASI[kume_no]
    grup = "train" if dosya_adi in egitim_seti_isimleri else "test"

    video_yolu = ANA_KLASOR / dosya_adi
    video_koku = Path(dosya_adi).stem

    # A) Sesi çıkar
    ses_sinyali, frekans = videodan_ses_cikar(video_yolu)
    if np.max(np.abs(ses_sinyali)) > 0:
        ses_okunan_video += 1

    # B) Görüntü karelerini ve ses pencerelerini çıkar
    okuyucu = cv2.VideoCapture(str(video_yolu))
    toplam_kare = int(okuyucu.get(cv2.CAP_PROP_FRAME_COUNT))

    if toplam_kare <= 0:
        okuyucu.release()
        continue

    kare_indeksleri = np.linspace(toplam_kare * 0.1, toplam_kare * 0.9, PENCERE_SAYISI, dtype=int)
    ses_parca_uzunlugu = max(16000, len(ses_sinyali) // PENCERE_SAYISI)

    for pencere_id, kare_idx in enumerate(kare_indeksleri):
        okuyucu.set(cv2.CAP_PROP_POS_FRAMES, kare_idx)
        basarili, kare = okuyucu.read()
        if not basarili or kare is None:
            continue

        # 1. Görüntüyü 224x224 olarak kaydet
        kare_224 = cv2.resize(kare, (224, 224))
        goruntu_adi = f"{video_koku}_p{pencere_id}.jpg"
        goruntu_yolu = CIKTI_KLASORU / "goruntuler" / goruntu_adi
        cv2.imwrite(str(goruntu_yolu), kare_224)

        # 2. İlgili zaman penceresinin ses spektrogramını kaydet
        baslangic = pencere_id * ses_parca_uzunlugu
        bitis = baslangic + ses_parca_uzunlugu
        ses_parcasi = ses_sinyali[baslangic:bitis]
        spek_goruntusu = spektrogram_uret(ses_parcasi, frekans)

        spek_adi = f"{video_koku}_p{pencere_id}.jpg"
        spek_yolu = CIKTI_KLASORU / "spektrogramlar" / spek_adi
        cv2.imwrite(str(spek_yolu), spek_goruntusu)

        veri_kayitlari.append({
            "ornek_id": f"{video_koku}_p{pencere_id}",
            "ana_video": dosya_adi,
            "goruntu_dosyasi": f"goruntuler/{goruntu_adi}",
            "spektrogram_dosyasi": f"spektrogramlar/{spek_adi}",
            "sinif_id": kume_no,
            "duygu_adi": duygu_adi,
            "veri_grubu": grup
        })

    okuyucu.release()

# 4. HAZIR VERİ SETİ TABLOSUNU KAYDET
hazir_df = pd.DataFrame(veri_kayitlari)
hazir_df.to_csv(CIKTI_KLASORU / "veri_seti_indeksi.csv", index=False)

print(f"\nHarika! Ses kanalı başarıyla ayrıştırılan video sayısı: {ses_okunan_video} / {len(etiket_tablosu)}")
print(f"Toplam üretilen örnek sayısı (Görüntü + Spektrogram çifti): {len(hazir_df)}")
print("\n--- EĞİTİM VE TEST DAĞILIMI ---")
print(pd.crosstab(hazir_df["duygu_adi"], hazir_df["veri_grubu"]))
print(f"\nTüm dosyalar '{CIKTI_KLASORU.name}' klasörüne kaydedildi!")