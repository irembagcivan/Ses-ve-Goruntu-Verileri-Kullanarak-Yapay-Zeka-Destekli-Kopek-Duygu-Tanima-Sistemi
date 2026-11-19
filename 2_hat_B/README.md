> **V16 güncellemesi (15 Kasım 2026): Dört duygu katmanı artık kural türevli etiketlerle değil, **insan duygu etiketleriyle** eğitilir.** Etiketler iki bağımsız etiketleyiciden ve uzlaşma sonrası oluşturuldu (Cohen kappa 0,553; 125 etiketli video: Relaxed 50, Happy 32, Sad 26, Angry 17; 2 belirsiz video dışarıda). Dosya: `results/v14_input_review/emotion_human_labels.csv`. Kayıt günü gruplu (3 grup, leave-one-group-out) değerlendirmede macro-F1 yaklaşık 0,40–0,44 (tesadüf yaklaşık 0,25; tüm C değerleri raporlandı): `python -m src.evaluation.human_emotion_day_eval`, sonuç `results/human_emotion_eval/summary.md`. Bu bağımsız bir test değildir; üç kayıt grubu vardır ve etiketler kayıt günüyle kısmen karışıktır (ilk günde Angry yok). Görüntü ve ses özellikleri aynı kalmıştır. Aşağıda ve eski belgelerde geçen “kural türevi duygu” ifadeleri V16 öncesi tarihsel durumu anlatır.

> Belge yönetimi: Başlangıçta Proje belgeleri → Yöntem ve sonuçlar / Sunum. Dosya ekleme, açma/indirme ve silme desteklenir; hazır örnekler listeden kaldırılabilir. Eklenen kişisel dosyalar yerelde kalır ve GitHub/paylaşım paketine otomatik dahil edilmez.

> Son bağlantı ve şema güncellemesi: Başlatıcı servisi ayrı süreçte çalıştırır; arayüz yeniden bağlanır ve oturumu yeniler. Ekran görüntüsündeki `vucut` alanı için üç sınıflı yardımcı model eklendi (`configs/demo/body_state.json`); dört duygu katmanı v3 bu puanları da kullanır. Kaynak eski etiket kopyasıdır; son Excelin birebir doğrulanması tamamlanmadı.

> Güncel ek: Yeni video analizinde kuyruk/kulak/duruş/hareket ve YAMNet ses olaylarından **Mutlu, Üzgün, Öfkeli, Rahat** için insan duygu etiketleriyle (V16) eğitilmiş dört duygu tahmin katmanı vardır. Gün gruplu doğruluğu zayıftır (macro-F1 ≈ 0,40–0,44); puanlar kalibre değildir. Ayrıntılar: [dört duygu katmanı](docs/four_emotions_bridge.md).

# Köpek videolarından poz ve ses tabanlı davranış analizi

Projeyi devralacak kişi veya yapay zekâ için: [Ayrıntılı devir dosyası](AIYE_ILET.md).

Son insan etiketleriyle altı hedef tahmin eden yerel araştırma prototipi:
**hareket, ses, duruş, kuyruk konumu, kuyruk hareketi ve kulak**.
Etiketler duygu sınıfları değildir. Eski kural türevi duygu dosyası artık kullanılmaz; dört duygu katmanı insan etiketleriyle eğitilir (V16 notu).

## Son araştırma ve duygu güncellemesi · V15

[Model seçimi ve gerçek deney raporu](docs/literature_and_affect_v15.md): aynı gün
gruplu davranış değerlendirmesinde karar birleşimi macro-F1 0,370 (önceki klasik
birleşik 0,346). Uygulama için eğitim CV'si %75 görüntü/%25 ses kararını seçti.

Dış insan algısı etiketli EmotionalCanines seslerinden ayrı valence/arousal modeli
eğitildi. Ana ekranda tek tahmini duygu yönü ve uyarılma düzeyi, destek ve davranış
kanıtı gösterilir. Bu, yerel 127 videoda doğrulanmış veya görsel+sesle uçtan uca
eğitilmiş duygu modeli değildir. Dört duygu katmanı (V16) yerel 127 videonun insan duygu etiketleriyle eğitilir; gün gruplu başarısı zayıftır.

## Güncel durum

- Kaynak listesi ve donmuş özellik önbelleği 127 videoyu kapsar; isim ve SHA256
  eşleşmesi kontrol edilmiştir.
- `labels_final kopyası.csv`, v13 eğitimindeki son etiketlerle hücre bazında aynıdır.
- 11 güvensiz kayıt bütün hedeflerde dışlanır. Boş, görünmeyen veya uzlaşılmayan
  alanlar yalnız ilgili hedefte maskelenir; eski etiketle doldurulmaz.
- YOLOv8n köpek tespiti, köpeğe özel RTMPose-S pozları ve YAMNet ses temsilleri
  çalışır. Kodlayıcılar donmuş; sinir ağı başlıkları ve klasik sınıflandırıcılar eğitilmiştir.
- FusionNet, CNN1D, GRU, BiGRU, LSTM–LSTM–GRU ve klasik LR/RF karşılaştırmaları var.
- Yeni video yükleme, H264/AAC oynatma, ses olayları ve gerçek YOLOv8s-seg
  piksel maskeleri çalışır. Maskeler henüz davranış eğitiminin girdisi değildir.

## Değerlendirme: iki ayrı kayıt

**V13 tarihsel dosya bölmesi:** 93 eğitim videosu, eğitimden korunmuş 25 eski test
kaydı. Yeni kalite bayraklarıyla 23 test videosunda ölçüm yapılır; destek hedefe
bağlıdır. Bu test geliştirme boyunca incelendi; bağımsız yeni test değildir.

**V14 gün gruplu yeniden doğrulama:** dosya adları veya disk tarihleri yerine
videonun gömülü kamera çekim tarihi incelenir. Gün grupları dış katlarda tamamen
ayrılır; epoch ve model seçimi yalnız iç eğitim günlerinde yapılır. Metadata günü,
onaylanmış çekim oturumu veya hayvan kimliği sayılmaz. Tarihi belirsiz kayıtlar
bu değerlendirmeye alınmaz. Sonuç ve kesin kapsam:
[eleştiri ve veri denetimi](docs/review_and_day_validation.md).

Bu iki protokol farklı örnekleri kapsar. Skor farkını doğrudan yöntem kazancı diye
sunmayın. Hedef tanımı değişen eski beş görevli deneyler de altı görevli sonuçlarla
aynı ölçekte karşılaştırılamaz. Kanıtlanmış genel üstünlük iddiası yoktur.

## Çalıştırma ve paylaşım

Ana klasörde **BASLA.html** başlangıç sayfasını açın. Değerlendirme raporu
**DEGERLENDIRME.html**, kurulum rehberi **KURULUM.html** olarak tarayıcıda okunur.
Yeni video analizi için **START_ANALYSIS.command** (Mac) veya
**START_ANALYSIS.bat** (Windows) çalışmalı. Güncel adres:
http://127.0.0.1:8768/start . 8766 önceki oturuma ait olabilir.
Galeri içindeki video yükleme paneli sunucuyla açıldığında aynı sayfada çalışır;
diskten açılan galeri çalışan analiz uygulamasına yönlendirir.
Ses kanalı olmayan videoda görüntü sinir ağı kullanılabilir.


Python 3.11 kullanın. Mevcut Mac ortamında gerçek çıkarım doğrulanmıştır;
yeni Windows/Linux kurulumu ayrıca doğrulanmalıdır.

```sh
python -m pip install -r requirements-share.txt
python -m src.demo.launch --check-only
python -m src.demo.launch
```

Mac: `START_ANALYSIS.command`. Windows: `START_ANALYSIS.bat`.
Tarayıcı: http://127.0.0.1:8768/ . Galeri: http://127.0.0.1:8768/gallery .

[Kurulum ve paylaşım rehberi](PAYLASIM_KURULUM.md) ·
[Son etiketlerle eğitim sonuçları](docs/labels_final_update.md) ·
[Eleştiri, dosyalar ve gün gruplu değerlendirme](docs/review_and_day_validation.md) ·
[Sunum](docs/hoca_sunumu.html) · [Konuşma notları](docs/sunum_konusma_notlari.md).

Aktif kontrol noktası ve galeri: `configs/demo/active_release.json`.
Ağırlıklar, kaynak videolar ve büyük galeri Git dışında tutulur. Tam paylaşım
paketi çalışma ağırlıklarını içerir; kaynak veri setini ve kişisel yüklemeleri içermez.

## Yeniden üretim

```sh
python -m src.deep_learning.final_labels --output results/demo_v13_repeat
python -m src.evaluation.recording_days --output results/day_metadata_repeat
python -m src.evaluation.day_validation --days results/day_metadata_repeat/recording_days.csv --output results/day_validation_repeat
python -m pytest -q
node tests/final_labels_ui_smoke.cjs
```

Komutları depo kökünde çalıştırın. Özellikler ve etiket girdileri ayrıca gerekir;
paket içi yollar için paylaşım rehberine bakın. Tarih denetimi orijinal videoları
okur; gün gruplu eğitim, kaydedilmiş özellik ve metadata ile tekrar çalışabilir.

## Açık bilimsel sınırlar

İki etiketleyicinin ayrı bağımsız formları bu dosyalarda yoktur; konsensüs
CSV'sinden Cohen kappa hesaplanamaz. `konsensus_disi` alanı kappa değildir.
Köpek kimlikleri ve doğrulanmış oturum eşlemesi yoktur. En büyük kutu insanın
odak köpeğiyle uyuşmayabilir. Sesler karışıktır; odak köpeğe ses kaynağı ataması yoktur.
Kısa pencere doğruluğu ve segmentasyon IoU/Dice ölçülmemiştir. Hırlama ve bazı
poz sınıflarında örnek çok azdır. Duygu doğrulaması tamamlanmış değildir.

Kaynak veri: https://data.mendeley.com/datasets/5g5hg9hrkv/2 . Kullanımda veri ve
model lisansları korunmalıdır; bkz. `docs/third_party/`.

Eski uygulama ayrıntıları ve tarihli görev kayıtları
[arşivlenmiş README](docs/history/README_pre_v14.md) içinde korunmuştur.
Güncel durum için bu sayfayı ve v13/v14 raporlarını kullanın.
