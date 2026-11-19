> Proje belgeleri: Başlangıç sayfasındaki Yöntem ve sonuçlar / Sunum bağlantılarından PDF, DOCX, PPTX, TXT veya MD ekleyebilir, açabilir ve silebilirsin. Dosyalar yerel analiz kayıtlarının altında saklanır; paylaşım paketine veya GitHub’a otomatik eklenmez. Hazır belgelerde Listeden kaldır, kaynak dosyayı değiştirmeden koleksiyondan çıkarır.

> Bağlantı güncellemesi: START_ANALYSIS servisi başlatıcı penceresinden bağımsız çalıştırır. Bilgisayar yeniden başladığında başlatıcıyı tekrar açın. Video ekleme ekranı güncel oturumu alır, kopan analiz kontrolünü sürdürür ve son kaydı aynı tarayıcıda hatırlar. `Yeniden bağlan` düğmesi mevcuttur. Yeni vücut tahmini ve dört duygu yorumu için `docs/four_emotions_bridge.md` dosyasına bakın.

# Arkadaşlar için güncel sürüm · son etiketler v13 / değerlendirme v14

Yeni [v14 denetimi](docs/review_and_day_validation.md), aynı konsensüs dosyasını
ve 103 videoda üç kamera gününü ayıran yeniden doğrulamayı açıklar. Kural türevi
duygu dosyası eğitim hedefi değildir. Kappa için iki bağımsız ham etiket formu
ve doğrulanmış oturumlar için insan onaylı eşleme hâlâ gerekir.


Son sürüm 127 videoyu altı ayrı hedefle işler: hareket, ses, duruş, kuyruk
konumu, kuyruk hareketi, kulak. 93 güvenli eğitim videosu vardır. Korunan 25
test dosyasından 23 tanesi yeni kalite etiketleriyle ölçülebilir. Güvensiz ve
uzlaşılmayan hedefler maskelenir. Eski beş hedefli sonuçlar güncel sonuç değildir.

Güncel deney açıklaması: `docs/labels_final_update.md`. Yeni kontrol noktaları
`results/demo_v13_release/` içindedir. Aktif sürüm `configs/demo/active_release.json`
ile seçilir. Geçmiş analiz kayıtları otomatik yeniden etiketlenmez; yeni sonuç
için videoyu yeniden yükleyin.


## V15 ses duygu ve karar birleşimi

Yeni video analizinde tek önerilen karar ve ayrıca ses duygu paneli bulunur.
Ses modeli dış insan algısı etiketleriyle valence/arousal öğrenir; yerel kural
CSV'sini kullanmaz. Yerel köpeklerde duygu doğruluğu bilinmiyor. Ham dış veri
ZIP'te yok; çalıştırma için gerekli başlıklar ve kaynak lisansı var.
Ayrıntı: docs/literature_and_affect_v15.md.

## Açılacak dosyalar

Ana klasörde **BASLA.html** ile başlayın. **DEGERLENDIRME.html** raporu, **KURULUM.html**
kurulum rehberini açar. Yeni video analizi için **START_ANALYSIS.command** (Mac)
veya **START_ANALYSIS.bat** (Windows) çalıştırın; Terminal açık kalsın.
Güncel sayfa: http://127.0.0.1:8768/start . 8766 eski uygulama olabilir.
Galeri üstündeki panelden dosya seçilebilir. Ses yoksa görüntü modeli sonuç üretir.

## Hangi klasör güncel?

Bu bilgisayarda güncel kaynak:
`/Users/niluferoden/Documents/Codex/canine-multimodal-emotion`

`Documents/canine-multimodal-emotion` ayrı, eski bir Git kopyasıdır. Yeni geliştirmeler
ve paylaşım paketi `Documents/Codex` altındaki projeden üretilir. Eski kopyanın
`.venv` ortamı hâlen kullanılabilir; eski kodla yeni kodu karıştırmayın.

GitHub çalışma dalı: `codex/multimodal-demo-20261108`.
https://github.com/nilufer-oden/canine-multimodal-emotion/tree/codex/multimodal-demo-20261108
Depo özeldir; GitHub bağlantısını açacak arkadaşların depoya erişimi olmalıdır.
Alternatif olarak `canine_tam_proje_paketi.zip` dosyasını paylaşın.

## Tam paketle ilk kurulum

ZIP'i bütünüyle bir klasöre çıkarın. ZIP içinden program çalıştırmayın.
Python 3.11 kullanımı mevcut ortamda test edilmiştir. Sanal ortam herkese kendi
bilgisayarında kurulmalıdır; Mac sanal ortamını Windows'a kopyalamayın.

Mac / Linux:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-share.txt
python -m src.demo.launch --check-only
python -m src.demo.launch
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-share.txt
.\.venv\Scripts\python.exe -m src.demo.launch --check-only
.\.venv\Scripts\python.exe -m src.demo.launch
```

Kurulumdan sonra Mac'te `START_ANALYSIS.command`, Windows'ta
`START_ANALYSIS.bat` kullanılabilir. Tarayıcı otomatik açılır. Açılmazsa:
http://127.0.0.1:8768

`requirements-share.txt`, şu anda çalışan temel paket sürümlerini sabitler.
Tüm geçişli bağımlılıkları içeren platformlar arası bir kilit dosyası değildir.
Yeni Windows/Linux kurulumu ayrıca doğrulanmalıdır; mevcut gerçek çıkarım testleri
bu Mac'te yapılmıştır. Root `requirements.txt` daha genel geliştirme aralıklarıdır;
paylaşım uygulaması için `requirements-share.txt` dosyasını tek başına kullanın.

## Neler dahil?

Tam paket kodu, eğitilmiş FusionNet/CNN/GRU/BiGRU ağlarını, normalizasyonları,
YOLO/RTMPose/YAMNet ve YOLOv8s-seg çalışma ağırlıklarını içerir. Ham kaynak veri seti,
kişisel yüklenen videolar ve 127 önizlemeli büyük hazır galeri pakete eklenmez.
Yeni video analizi bu galeriden bağımsız çalışır. Model ağırlıkları GitHub'a
konmaz; GitHub'dan yalnız kod indiren kişi tam paketteki `models` ve ilgili
`results/demo_v13_release` model dosyalarını da edinmelidir.

## Yeni videoda segmentasyon

Video yükle ve analizi başlat. Köpek pozu, ses olayları ve sekiz davranış modeli
karşılaştırmasının yanında renkli segmentasyon videosu ve siyah-beyaz maske görünür.
Her örnek için PNG maskeleri ve zaman damgalı JSON indirilebilir. Maskeler dikdörtgen
kutulardan türetilmez; ön-eğitimli segmentasyon ağı piksel düzeyinde tahmin eder.

İlk 60 saniye analiz edilir. Maske videosu en fazla 8 kare/sn ve uzun kenarı
960 piksel olarak işlenir; indirilen maskeler bu görüntü boyutuna karşılık gelir.
Kare içindeki köpek numaraları kalıcı birey kimliği değildir. Gerçek maske etiketleri
olmadığından IoU/Dice başarısı ölçülmemiştir. Segmentasyon mevcut davranış ağlarını
bu maskelerle yeniden eğitmez; önceki F1 skorları değişmez.
