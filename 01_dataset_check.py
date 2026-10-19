from pathlib import Path
import cv2
import pandas as pd

# Veri seti klasörünün yolu (Kod videolara aynı klasördeyse "." olarak kalabilir)
VERI_SETI_KLASORU = Path("/Users/irembagcivan/Desktop/A Real World Dataset of Canine Emotions Using Vide/Ses ve Goruntu Verileri Kullanarak Yapay Zeka Destekli Kopek Duygu Tanima Sistemi/Internship dataset/hazir_veri_seti")

video_dosyalari = []
diger_dosyalar = []

# Klasördeki tüm dosyaları tarayalım
for dosya in sorted(VERI_SETI_KLASORU.iterdir()):
  # Gizli sistem dosyalarını (.DS_Store vb.) atla
  if dosya.is_file() and not dosya.name.startswith("."):
    if dosya.suffix.lower() in [".mov", ".mp4", ".avi"]:
      video_dosyalari.append(dosya)
    else:
      diger_dosyalar.append(dosya)

print(f"Toplam Video Sayısı: {len(video_dosyalari)}")
print(
    "Klasördeki Diğer Dosyalar (Etiket/CSV var mı?):"
    f" {[d.name for d in diger_dosyalar]}\n"
)

# Videoların teknik özelliklerini kaydedeceğimiz listeler
kayitlar = []
bozuk_videolar = []

for video_yolu in video_dosyalari:
  video_okuyucu = cv2.VideoCapture(str(video_yolu))

  # Video açılamıyorsa bozuk listesine ekle
  if not video_okuyucu.isOpened():
    bozuk_videolar.append(video_yolu.name)
    continue

  fps = video_okuyucu.get(cv2.CAP_PROP_FPS)
  toplam_kare = int(video_okuyucu.get(cv2.CAP_PROP_FRAME_COUNT))
  genislik = int(video_okuyucu.get(cv2.CAP_PROP_FRAME_WIDTH))
  yukseklik = int(video_okuyucu.get(cv2.CAP_PROP_FRAME_HEIGHT))

  sure_saniye = toplam_kare / fps if fps > 0 else 0
  boyut_mb = video_yolu.stat().st_size / (1024 * 1024)

  kayitlar.append({
      "dosya_adi": video_yolu.name,
      "sure_saniye": round(sure_saniye, 2),
      "fps": round(fps, 1),
      "toplam_kare": toplam_kare,
      "cozunurluk": f"{genislik}x{yukseklik}",
      "boyut_mb": round(boyut_mb, 2),
  })

  video_okuyucu.release()

# Sonuçları Pandas tablosuna dönüştürelim
tablo = pd.DataFrame(kayitlar)

print("--- VİDEO SÜRE VE BOYUT ÖZETİ ---")
print(
    tablo[["sure_saniye", "fps", "toplam_kare", "boyut_mb"]].describe().round(2)
)

if bozuk_videolar:
  print(f"\nAçılmayan / Bozuk Videolar: {bozuk_videolar}")
else:
  print("\nTüm videolar sorunsuz açılıyor!")

# Detaylı listeyi CSV dosyası olarak kaydet
tablo.to_csv("video_ozet_bilgileri.csv", index=False)
print("\nDetaylı liste 'video_ozet_bilgileri.csv' adıyla kaydedildi.")