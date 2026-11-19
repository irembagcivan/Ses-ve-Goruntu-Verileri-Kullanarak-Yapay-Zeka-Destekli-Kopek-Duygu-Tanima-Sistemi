# Görev 10B — Kontrollü pose incelemesi

Bu araç mevcut pose JSONL ve Görev 10A kalite raporlarını kullanır. Yeni detection,
pose inference veya model yükleme yapmaz. Confidence, anatomik doğruluk veya
accuracy değildir. Gerçek Mac videoları bu bulut ortamında bulunmadığından burada
yalnız sentetik testler yapılmıştır.

## 1. Üç videoluk manifest

Depo kökünde `review_manifest.json` isimli bir yerel dosya oluşturun. `VIDEO_ID_1`
ve diğer yer tutucuları gerçek batch klasörleriyle değiştirin. Aşağıdaki kalite
raporlarının önce Görev 10A ile üretilmiş olması gerekir. `video` alanı yalnız
`--render` kullanıldığında gereklidir.

```json
{
  "videos": [
    {
      "pose": "results/batch/VIDEO_ID_1/attempt-000001/pose.jsonl",
      "quality": "results/quality/video1.json",
      "frames": "results/quality/video1.frames.jsonl",
      "video": "/path/to/Internship dataset/original1.MOV"
    },
    {
      "pose": "results/batch/VIDEO_ID_2/attempt-000001/pose.jsonl",
      "quality": "results/quality/video2.json",
      "frames": "results/quality/video2.frames.jsonl",
      "video": "/path/to/Internship dataset/original2.mov"
    },
    {
      "pose": "results/batch/VIDEO_ID_3/attempt-000001/pose.jsonl",
      "quality": "results/quality/video3.json",
      "frames": "results/quality/video3.frames.jsonl",
      "video": "/path/to/Internship dataset/original3.mp4"
    }
  ]
}
```

Göreli yollar manifest dosyasının bulunduğu klasöre göre çözülür. Manifesti
kullanıcı yolları içerebileceği için Git'e eklemeyin. Rapor videoları manifest
sırasıyla `video1`, `video2`, `video3` olarak tanımlar; mutlak yolları kopyalamaz.
Manifest sırasını değiştirmek bu inceleme kimliklerini değiştirir. Bu kimlikler
tracking kimliği değildir. Araç en fazla 10 video kabul eder; 127 videoyu otomatik
olarak taramaz.

## 2. Rapor ve boş manuel inceleme tablosu

Mevcut Mac Python ortamınızda, depo kökünde:

```bash
python -m src.quality.run_review \
  --manifest review_manifest.json \
  --output-dir results/review10b \
  --per-category 2 \
  --low-confidence 0.5 \
  --visibility-change-min 8
```

Bu komut video çözmez; model çalıştırmaz. Şunları oluşturur:

- `comparison.json`: video başına detection coverage, sıfır/tek/çoklu detection
  oranları, detection ve keypoint confidence histogramları, dört görünümün nokta
  sayısı ortalamaları, temporal bayrak toplamları ve seçilen kareler.
- `manual_review.csv`: seçilen kare başına bir satır. Alanlar:
  `video_id`, `frame_index`, `category`, `detection_count`,
  `mean_keypoint_confidence`, `visible_keypoints_0_3`, `visible_keypoints_0_5`,
  `manual_dog_visible`, `manual_anatomy_correct`, `notes`.

Son üç alan her zaman **boş** oluşturulur. Bunları PNG/orijinal video incelemesinden
sonra siz doldurun; belirsiz değerlendirmeleri notes alanında belirtin. Çok köpekli
karelerde tablo ortalaması ve görünür nokta sayısı o karedeki tüm detection'lara
aittir; bireyler arasında zamansal eşleştirme yapılmaz. Aynı kare birden fazla
kategoriye girerse tek satırda kategoriler noktalı virgülle birleştirilir.

## 3. Aynı karede dört görselleştirme

Yeni bir çıktı klasörü seçerek:

```bash
python -m src.quality.run_review \
  --manifest review_manifest.json \
  --output-dir results/review10b_visual \
  --per-category 2 \
  --low-confidence 0.5 \
  --visibility-change-min 8 \
  --render
```

Her seçilmiş kare için `video1/`, `video2/`, `video3/` alt klasörlerinde:

1. `raw39-0.3.png`: orijinal 39 noktalık şema, confidence >=0.3.
2. `raw39-0.5.png`: aynı şema, confidence >=0.5.
3. `conservative-0.3.png`: mevcut 9 aday noktalık profil, confidence >=0.3.
4. `conservative-0.5.png`: aynı profil, confidence >=0.5.
5. `comparison.png`: aynı karenin dört görünümünü yan yana 2×2 karşılaştırır.
6. `points.csv`: orijinal indeks/isim, geçerli görüntü koordinatları/skor ve dört
   görünümün maskeleri. Geçersiz/eksik değerler boş; geçerli koordinatlar float
   olarak saklanır. Bu dosya ham pose.jsonl yerine geçmez.

Dosyalar `frame-00000001-...` biçimindedir. Çizilen noktalarda keypoint adı ve
confidence yazılır; başlıkta detection confidence, detection sayısı ve görünür
nokta sayısı yer alır. Çok sayıda yakın noktada yazılar örtüşebilir; points.csv
okunabilir sayısal karşılığı sunar. Görüntüde raster çizim için yuvarlama yapılır;
kaynak pose koordinatları değişmez. Orijinal 39 indeks ve sıra korunur.

Hiç skeleton bağlantısı çizilmez. `raw39` görünümü, antler dahil modelin çok türlü
şemasını teşhis için gösterir; bu etiketler köpek anatomisi veya doğrulanmış eklem
konumları olarak sunulmaz. Conservative maske mevcut modülden isme göre alınır.

## Ölçümler ve seçim politikası

- Detection coverage = en az bir detection bulunan geçerli kayıt / tüm geçerli
  kare kayıtları. Gerçekte köpek bulunma oranı veya detection recall değildir.
- Confidence histogramları [0,1] aralığında 0.1 genişlikli kutulardır; yalnız
  geçerli sayısal skorlar dahil edilir, son kutu 1.0'ı içerir. Histogram örnek
  sayısı ayrıca yazılır. Keypoint skorları görüntü dışı noktaları da içerebilir;
  görünürlük yalnız sonlu, sınır içinde, geçerlilik işaretli noktaları içerir.
- Dört görünümün görünür nokta ortalamaları detection gözlemi başınadır.
  Hiç detection yoksa ortalama null'dır. Detection var ama güvenilir nokta yoksa
  görünür nokta sayısı sıfırdır. Eksik confidence ortalaması null kalır; düşük
  confidence kategorisine otomatik sokulmaz.
- No-dog: en erken K kare; false negative olarak etiketlenmez.
- Low-confidence: ortalama geçerli keypoint confidence <0.5 olan karelerden en
  düşük K ortalama (`--low-confidence` ile değiştirilebilir).
- Visibility-change: kalite raporundaki gained+lost toplamı >=8 olanlardan en
  yüksek K değer (`--visibility-change-min` ile değiştirilebilir).
- Coordinate-jump: kalite raporunda en az bir sıçrama bayrağı bulunan karelerden
  en yüksek bayrak sayısına sahip K örnek.
- Eşitlikler frame_index ile çözülür. K `--per-category` ile belirlenir
  (varsayılan 2; en fazla 20). En çok 4×K ayrı kare seçilir. Seçim deterministiktir;
  istatistiksel olarak tarafsız video örneklemesi değildir. Her kategori bulunmak
  zorunda değildir; görünürlük değişimi/jump confidence doğruluğuna eşit değildir.

Temporal sayılar mevcut kalite raporunun eşiklerine bağlıdır. Araç farklı videoların
ayarlarını ayrıca listeler ve `temporal_settings_comparable` alanıyla ayar eşitliğini
belirtir; havuzlanmış anatomik kalite skoru üretmez. Aynı ayarlar tek başına video
koşullarının eşdeğer olduğunu göstermez. Ground truth kullanılmaz.

## Güvenlik ve sınırlamalar

Çıktı klasörü önceden varsa işlem reddedilir. Kaynaklar ve mevcut batch/motion
çıktıları silinmez. Kalite raporunun pose SHA-256 değeri, ayrıntı dosyasının kayıt
sayısı, satır/kare/detection eşleşmesi kontrol edilir; giriş hash'leri rapora
kaydedilir. Görev 10A raporu ayrıntı dosyasını kriptografik olarak imzalamaz;
`quality` ile aynı çalışmada oluşan doğru `.frames.jsonl` dosyasını vermelisiniz.
Orijinal video pose verisine hash ile bağlı olmadığından doğru video seçimi sizin
sorumluluğunuzdadır; PNG üretiminde çözünürlük uyumu ayrıca kontrol edilir.

PNG için mevcut OpenCV gerekir; yeni bağımlılık kurulmaz. Kareler sıralı çözülür,
yalnız seçilen poz kayıtları ve bir karenin dört görüntüsü RAM'de tutulur. Source
video bu işleme göre yeniden encode edilmez; yalnız PNG kopyalar çıkarılır.
PNG hatasında karşılaştırma raporu ve manuel tablo yine üretilir, ilgili videoda
`failed_or_unavailable` kaydedilir, CLI 2 döner. Kısmi PNG'ler korunabilir. Temel
girdi hatasında yeni klasörde failure.json kalır; eski dosyalara dokunulmaz.

## Manuel incelemeden sonra verilebilecek kararlar

1. No-dog karelerinde köpek gerçekten görünüyorsa detection kaçırmaları için
   ayrı, küçük ve etiketli değerlendirme hazırlamak; henüz recall iddia etmemek.
2. 0.3'te görünen ek noktalar anatomik olarak makulse yalnız görselleştirme
   eşiği denemek; yanlışlarsa confidence düşürmenin kaliteyi düzeltmediğini kaydetmek.
3. Raw39 iyi ama conservative seyrekse model ile çizim maskesini ayrı değerlendirmek.
4. Sıçramalar hatalı tahminle ilişkiliyse crop/detection değişimi ve temporal
   filtrelemeyi sonraki kontrollü deneylerde incelemek. Gerçek hızlı hareketi hata
   saymamak; tracking veya smoothing'i bu görevde otomatik eklememek.

Herhangi bir ayar/model değişikliğinden önce manuel tablo ve aynı karelerin dört
panelini birlikte değerlendirin. Bu görev model kalitesini otomatik iyileştirmez.
