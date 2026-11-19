# Arkadaş değerlendirmesi, yeni dosyalar ve gün gruplu doğrulama · v14

14 Kasım 2026. Bu belge v13 etiket güncellemesinin ardından değerlendirme protokolünü güçlendirir. Etiket dosyası değişmediği için v13 canlı demo kontrol noktaları korunmuştur; yeni gün gruplu skorlar ayrı değerlendirmedir.

## Arkadaşının söylediklerini değerlendirme

| Konu | Değerlendirme / yapılan işlem |
|---|---|
| Hazır tespit, poz, ses ve fusion hattını kullanmak | Doğru öneri. 127 kaydın özellikleri ve kaynak hash sırası doğrulandı; yeniden piksel/ses çıkarımı gerekmedi. |
| Yalnız 97 etiketli video var | Önceki sürüm için doğruydu. Son konsensüs CSV artık 127 videoyu kapsıyor. 11 güvensiz kayıt dışlanınca v13 eğitiminde 93 kayıt var. |
| SHA256 kimlik/oturum sızıntısını önlemez | Doğru. SHA yalnız birebir içerik kopyalarını yakalar. Kamera günleriyle yeni dış-kat gruplaması eklendi; köpek kimliği sızıntısını tamamen önlediğimiz iddia edilmez. |
| 25 test dosyası bağımsız değil | Doğru ve hâlâ geçerli. Aynı videoları yeniden bölmek onları yeni bağımsız test yapmaz. |
| Kural türevi duygu döngüsel | Doğru. İnsan davranış etiketlerinden türetilmiş duygu, bağımsız duygu gerçekliği değildir; bu dosya eğitim veya duygu başarısı hedefi yapılmadı. |
| README çelişkili | Doğru. Yalnız üstüne güncellik notu eklemek yetersiz kalmıştı. README yeniden yazıldı; eski görev kayıtları açıkça tarihsel arşive taşındı. |
| Düşük sonuçları üstünlük gibi sunmamak | Doğru. Yeni gün gruplu sonuçlar ve çoğunluk referansı birlikte gösteriliyor. Kanıtlanmış genel üstünlük iddiası yok. |
| Kappa eksik | Doğru. İki kişinin ayrı, bağımsız, uzlaşma öncesi etiketleri dosyada yok; konsensüsten veya kural etiketinden kappa üretilemez. |

## Gelen iki dosyanın içeriği

**labels_final kopyası.csv:** 127 satır, altı hedef, konsensüs/kalite bayrakları. Hücre bazında v13 eğitiminde kullandığımız `labels_final.csv` ile aynı. Oturum, kayıt günü ve iki ayrı etiketleyici sütunları yok. Yeniden eski beş hedefli etiketlere dönülmedi.

**duygu_kural_turevli.csv:** 127 kaydın tamamı `KURAL TÜREVLİ - insan duygu etiketi değil` kaynağıyla işaretli. 37 Relaxed, 33 Happy, 23 Angry, 22 Sad ve 12 boş duygu var. `vucut/aktivite/kuyruk` eski şemada. Bu dosyayı v2 insan gerçeği diye birleştirmek etiket kaynağını bozar.

Aynı isimli davranış alanları bile birebir aynı değil: güvenli kayıtlarda karşılaştırılabilir 116 duruş etiketinin 8’i, 77 kulak etiketinin 14’ü farklı. Örneğin IMG_5370.MOV son insan dosyasında ayakta, kural dosyasında oturuyor. Bunlar bir etiketleyici uyumu/kappa hesabı değil, dosya sürümleri arasındaki farklılıklardır. İnsan konsensüs CSV eğitim kaynağı olarak korundu.

## Kayıt tarihi denetimi

127 video incelendi. 114 kayıtta gömülü kamera creationdate bilgisi, 9 kayıtta yalnız container creation_time, 4 kayıtta tarih yok. 17 videoda kamera ve container günü farklı. Bu nedenle container tarihini doğrudan kayıt günü saymadık; dosya adı ve disk mtime/ctime kullanılmadı.

Örnek: IMG_5592.mov kamera bilgisinde 22 Haziran, container bilgisinde 1 Temmuz gösteriyor. Kaydedilen metadata yalnız tarih alanlarıdır; konum bilgileri rapora taşınmadı.

Ana değerlendirmeye yalnız kamera creationdate bulunan ve güvensiz olmayan 103 video alındı. 13 yalnız-container/eksik tarihli kayıt ile 11 güvensiz kayıt bu değerlendirme dışında; galeride kalırlar. Gün dağılımı:

| Kamera günü | Güvenli video |
|---|---:|
| 2026-06-18 | 60 |
| 2026-06-20 | 4 |
| 2026-06-22 | 39 |

Bu günler gömülü metadata kanıtıdır; çekim sahibi tarafından doğrulanmış oturum veya köpek kimliği değildir. Aynı köpek birden fazla gün bulunabilir. Güçlü sızıntı önleme için insan onaylı oturum/kimlik eşlemesi hâlâ gerekir.

## Yeni deney nasıl yapıldı?

Dış katlarda bir günün bütün videoları dışarıda tutuldu (3 leave-one-day-out katı). Her dış katın kalan iki günüyle iç leave-one-day-out seçim yapıldı. Sinir ağlarının 20/40/60/80 epoch seçimi ve klasik LR/RF seçimi yalnız iç katlardan yapılır; dış gün metriklerine bakılmaz. Ölçekleme, sınıf ağırlıkları ve imputer yalnız o katın eğitim kayıtlarıyla öğrenilir. Bütün yöntemler aynı 103 video üzerinde dış-kat tahmini üretir.

Kodlayıcılar ve 1 Hz/ilk 60 sn temsilleri değiştirilmedi. Tohum 42. Ham etiket, kalite, tarih, insan notu, konsensüs, köpek sayısı veya kural duygu etiketi özellik vektörüne katılmadı. Tarih yalnız grup bölmesinde kullanıldı.

Üç günden biri yalnız 4 güvenli videodur; bazı iç eğitim katları yalnız bu dört videodan oluşur. Bu durum model seçimini çok oynak yapar. Sınıf eksiklikleri ve her katın eğitim/test destekleri raporda açıkça saklanır; sınıfı olmayan eğitim katı için başarı beklenmez. Sabit sınıf listesiyle macro-F1 kullanıldı.

## Gün gruplu sonuçlar

| Model | Birleştirilmiş dış-kat macro-F1 | Günlere eşit ağırlıklı macro-F1 |
|---|---:|---:|
| neural_visual | 0.344 | 0.284 |
| neural_audio | 0.296 | 0.232 |
| neural_fusion | 0.330 | 0.275 |
| sequence_cnn | 0.313 | 0.268 |
| sequence_gru | 0.308 | 0.239 |
| sequence_bigru | 0.279 | 0.250 |
| sequence_lstm_lstm_gru | 0.261 | 0.244 |
| classical_visual | 0.344 | 0.297 |
| classical_audio | 0.283 | 0.254 |
| classical_fusion | 0.346 | 0.318 |
| majority_baseline | 0.215 | 0.245 |

İlk sütun dışarıda bırakılmış bütün videoların tahminleri birleştirildikten sonraki altı hedef macro-F1 ortalamasıdır. İkinci sütun önce her günün skorunu hesaplar, ardından günlere eşit ağırlık verir. 4 videolu güne büyük günlerle eşit ağırlık vermenin etkisi görünür tutuldu.

Bu sonuçlar farklı günlere geçişin zor olduğunu gösterir; veri sızıntısının varlığını tek başına kanıtlamaz. V13 dosya bölmesi 23 test videosu, bu değerlendirme 103 dış-kat tahmini içerir; sayısal fark yalnız bölme değişiminin nedensel etkisi veya yöntem kazancı diye sunulamaz. Yalnız üç grup olduğundan güvenilir bir genel üstünlük/istatistiksel anlamlılık iddiası kurulmadı.

**Bağımsız yeni test hâlâ yok.** Bu dosyalar önceki geliştirmede kullanıldı. Doğru isim, geçmiş veri üzerinde gün gruplu yeniden doğrulamadır. Nihai bağımsız test için daha önce hiç incelenmemiş yeni köpek/oturum kayıtları gerekir.

## Kappa ve oturum eşlemesi için eksik girdiler

Kappa için iki kişinin video bazında, uzlaşma öncesi ayrı ham etiket formları gerekir. Aynı konsensüs dosyasının iki kopyasını karşılaştırmak uygun değildir. Kod hazır:

```sh
python -m src.evaluation.annotation_agreement --left etiketleyici_a.csv --right etiketleyici_b.csv --output results/kappa.json
```

Her ham form `video` ve altı hedef sütununu içermeli; `konsensus_disi/kural/kaynak` tabloları reddedilir. Eksik/görünmeyen hücreler hedef bazında çiftlerden çıkarılır; çift sayısı ve ham uyum ayrıca raporlanır. Tek kategori gibi kappa tanımsızlığı sıfırla doldurulmaz. Mevcut proje için kappa hesaplandı denmedi.

Oturum taslağı: `results/recording_day_audit_v14/recording_days.csv`. `confirmed_session_id` ve `annotator_confirmed` alanları bilinçli olarak boş/false bırakıldı. Bunları çekim bilgisi olan kişi doğrulamalı; mevcut tarih grupları onaylanmış oturum gibi adlandırılmadı.

## Çıktılar ve tekrar

- Makine okunur tam dış/İç kat sonuçları: `results/day_validation_v14/report.json`.
- Kodla birlikte taşınan rapor: `docs/day_validation_results_v14.json`.
- Metadata ve boş oturum-onay alanları: `results/recording_day_audit_v14/recording_days.csv`.
- Galeri yeni gün tablosunu eski V13 dosya bölmesinden ayrı gösterir. Video yanındaki çıkarımlar V13 demo modelleridir; gün gruplu dış-kat tahminleri diye sunulmaz.
- Canlı modeller yeniden aynı etiket dosyasıyla gereksiz değiştirilmedi. Yeni kat modelleri yalnız değerlendirme içindir; sonradan dış gün skoruyla en iyi ağırlık seçilip uygulamaya geçirilmedi.

```sh
python -m src.evaluation.day_validation --output results/day_validation_repeat
python -m pytest -q
node tests/final_labels_ui_smoke.cjs
```

Ham videolar yeniden özellik çıkarmak için gerekmedi. Metadata denetimi ilk kez orijinal videolardan alındı; sonraki tekrarlar CSV ve özellik snapshot ile yapılır. İki bağımsız etiket formu ve onaylı oturum eşlemesi geldiğinde kappa/oturum değerlendirmesi ayrıca tamamlanabilir.

## Sürüm doğrulaması

380 Python testi geçti. Gün grupları dış ve iç katlarda ayrık; 103 videonun her biri tam bir dış kat tahmininde yer alıyor. Kaydedilmiş dış-kat sınıf tahminlerinden 11 modelin bütün birleştirilmiş macro-F1 değerleri yeniden hesaplanıp doğrulandı. Galeri durum denetimi gün tablosunu ve 127 video/30 eski etiketsiz/25 tarihsel test filtresini kontrol etti.
