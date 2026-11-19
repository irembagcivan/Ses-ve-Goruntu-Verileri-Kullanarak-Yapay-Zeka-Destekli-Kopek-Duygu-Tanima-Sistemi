> **V16 güncellemesi (15 Kasım 2026): Dört duygu katmanı artık kural türevli etiketlerle değil, **insan duygu etiketleriyle** eğitilir.** Etiketler iki bağımsız etiketleyiciden ve uzlaşma sonrası oluşturuldu (Cohen kappa 0,553; 125 etiketli video: Relaxed 50, Happy 32, Sad 26, Angry 17; 2 belirsiz video dışarıda). Dosya: `results/v14_input_review/emotion_human_labels.csv`. Kayıt günü gruplu (3 grup, leave-one-group-out) değerlendirmede macro-F1 yaklaşık 0,40–0,44 (tesadüf yaklaşık 0,25; tüm C değerleri raporlandı): `python -m src.evaluation.human_emotion_day_eval`, sonuç `results/human_emotion_eval/summary.md`. Bu bağımsız bir test değildir; üç kayıt grubu vardır ve etiketler kayıt günüyle kısmen karışıktır (ilk günde Angry yok). Görüntü ve ses özellikleri aynı kalmıştır. Aşağıda ve eski belgelerde geçen “kural türevi duygu” ifadeleri V16 öncesi tarihsel durumu anlatır.

> Belge yönetimi: Başlangıçta Proje belgeleri → Yöntem ve sonuçlar / Sunum. Dosya ekleme, açma/indirme ve silme desteklenir; hazır örnekler listeden kaldırılabilir. Eklenen kişisel dosyalar yerelde kalır ve GitHub/paylaşım paketine otomatik dahil edilmez.

> Son bağlantı ve şema güncellemesi: Başlatıcı servisi ayrı süreçte çalıştırır; arayüz yeniden bağlanır ve oturumu yeniler. Ekran görüntüsündeki `vucut` alanı için üç sınıflı yardımcı model eklendi (`configs/demo/body_state.json`); dört duygu katmanı v3 bu puanları da kullanır. Kaynak eski etiket kopyasıdır; son Excelin birebir doğrulanması tamamlanmadı.

> Güncel ek: Dört duygu tahmin katmanı (V16) insan duygu etiketleriyle eğitilir; ayrıntı yukarıdaki V16 notunda ve [dört duygu katmanı](docs/four_emotions_bridge.md) belgesinde.

# Başka bir yapay zekâya proje devri

**Proje:** canine-multimodal-emotion  
**Hazırlanma:** 14 Kasım 2026  
**Etiket sürümü:** labels_final-v13  
**Değerlendirme sürümü:** nested_camera_day-v14  
**Yeni araştırma/duygu sürümü:** literature_affect-v15; ayrıntı: docs/literature_and_affect_v15.md
**Güncel uygulama:** http://127.0.0.1:8768/start

Bu belge proje kaynak kodu, özellik önbelleği ve raporlarla birlikte okunmalıdır.
Burada yazılanlar tamamlanan işler ile açık eksikleri ayırır. Doğrulanmamış
bir özelliği tamamlanmış veya bir skoru bağımsız test başarısı olarak sunmayın.

## 0. Son eklenen V15: bu bölüm önceki aşamaların üzerine geçer

Artık yerel altı davranış modelinin yanında **dış insan algısı etiketleriyle eğitilmiş
ayrı ses duygu modeli** vardır. EmotionalCanines'ın 1.400 sesi pinned LFS hash'leriyle
alındı; ortak train/test sesleri ve çelişkili/tekrarlı eğitim kayıtları çıkarılarak
1.195 eğitim kaydı kullanıldı. Resmi 200 test satırı (199 benzersiz ses) korundu.
YAMNet ortalama+std temsilleri üzerinde eğitim CV'si RF'yi seçti: valence doğruluğu
%43, arousal %52.5, iki hedef macro-F1 ortalaması 0.477; iki boyut birlikte doğru %23.5. MLP alternatifi 0.469.
Bu, Husky/Shiba insan algısı etiketlerinde ses sınıflandırmasıdır. Yerel 127 videoda
artık iki kişinin insan duygu etiketi var (V16, kappa 0,553); gün gruplu duygu macro-F1 yaklaşık 0,40–0,44. Eski kural türevi
duygu CSV'si kullanılmıyor. Görüntü davranış kanıtıdır; senkron duygu etiketleriyle
eğitilmiş tam çok modlu duygu modeli henüz yoktur.

Ayrıca karar birleşimi aynı 103 videolu gün protokolünde 0.370 macro-F1 verdi;
önceki klasik birleşik 0.346 idi. Canlı karar %75 görüntü/%25 ses olarak yalnız
93 eğitim videosunun CV'sinden seçildi. Ana panel tek tahmini durum ve desteği
gösterir; ayrıntılı modeller karşılaştırma bölümünde. Ses yoksa veya uygun havlama
yoksa kesin duygu uydurulmaz. İlgili kod src/affect/, sonuçlar results/affect_v15_clean/
ve results/late_fusion_v15/ içindedir. Kaynak: docs/literature_and_affect_v15.md.

Aşağıdaki V13/V14 açıklamaları tarihsel deneyleri ayrıntılandırır. “Duygu etiketi yok”
ifadeleri yerel 127 videoyu anlatır; yeni dış etiketli ses modelinin yokluğu anlamına gelmez.

## 1. Yeni yapay zekâya doğrudan verilecek görev

Aşağıdaki metni, bu belge ve paketle birlikte kullanabilirsiniz:

> Bu, mevcut ve çalışan bir araştırma projesinin devamıdır. Önce AIYE_ILET.md,
> README.md, configs/demo/active_release.json, docs/review_and_day_validation.md,
> docs/labels_final_update.md ve bunların dayandığı JSON raporlarını oku. Sonra
> kodla belgelerin uyuştuğunu denetle. Sıfırdan başlama; doğrulanmış önbellekleri
> ve mevcut çalışan uygulamayı kullan. Asıl araştırma amacı köpeğin görüntü ve
> sesinden duyguya yaklaşmak; fakat mevcut bağımsız insan etiketleri yalnız altı
> davranış/ses hedefini doğruluyor. Kuraldan türetilen duygu etiketini gerçek duygu
> ground truth'u gibi kullanma. Eski beş hedefli sonuçları yeni şemayla karıştırma.
> Üç kamera günündeki iç içe gruplu değerlendirmeyi incele; bunun onaylanmış
> oturum/kimlik ayrımı veya bağımsız yeni test olmadığını koru. Önce gerçek açıkları
> önem sırasıyla açıkla, sonra mümkün olan düzeltmeleri mevcut kod üzerinde yap.
> Yeni deneyde özellik, veri kapsamı, bölme ve model değişikliklerini ayrı kaydet.
> Model/epoch seçimini dış test skoruna göre yapma; sahte sonuç veya etiket üretme.
> Sonunda değiştirdiğin dosyaları, gerçek deneyleri, çalıştırma komutlarını,
> kontrolleri ve hâlâ gerekli insan girdilerini bildir. Büyük ham videoları,
> yüklemeleri, ağırlıkları ve veri dosyalarını Git'e ekleme.

## 2. Amaç ve akademik çerçeve

Uzun vadeli amaç, köpeklerin görüntü ve sesinden duygusal durum hakkında bilgi
üreten çok modlu bir sistemdir. Şu anda ölçülebilen görev **insan etiketli gözlenebilir
davranış ve ses tahmini**dir. Mutlu, korkmuş, öfkeli gibi bağımsız doğrulanmış
insan duygu etiketleri bulunmuyor. Duygu sınıflandırmasının tamamlandığı iddia edilmez.

Sistem hazır kodlayıcılarla özellik düzeyinde transfer öğrenmesi yapar. YOLO,
RTMPose ve YAMNet donmuştur; sınıflandırıcılar ve sinir ağı başlıkları eğitildi.
Piksel ve dalga biçiminden uçtan uca ince ayar yapılmış değildir.

Değerli mevcut katkılar: köpeğe özel poz çıkarımı, ses temsili, altı görevli model,
tek/çok modlu karşılaştırmalar, zamansal mimari karşılaştırmaları, etiket kalite
maskelemesi, gün gruplu yeniden doğrulama ve çalışan yerel analiz arayüzü.

## 3. Hangi kopya ve GitHub dalı güncel?

Bu Mac'teki güncel ana klasör:

`/Users/niluferoden/Documents/Codex/canine-multimodal-emotion`

Ayrı eski kod kopyası:

`/Users/niluferoden/Documents/canine-multimodal-emotion`

İkinci klasörün kodu güncel kaynak değildir. Yalnız mevcut Python ortamı ilk
klasördeki komutları çalıştırmak için kullanılmıştır:

`/Users/niluferoden/Documents/canine-multimodal-emotion/.venv/bin/python`

GitHub özel depo: https://github.com/nilufer-oden/canine-multimodal-emotion  
Güncel çalışma dalı:
https://github.com/nilufer-oden/canine-multimodal-emotion/tree/codex/multimodal-demo-20261108  
Taslak PR: https://github.com/nilufer-oden/canine-multimodal-emotion/pull/11

**main dalı güncel sürüm sanılmamalı.** PR henüz birleştirilmedi. Yerel dal adı
`codex/emotion-demo`; GitHub çalışma dalının adı farklıdır. Yerel ve uzak commit
kimlikleri de farklı olabilir; yayımlanan kaynak ağaçlarının eşitliği kontrol edildi.
Son işlevsel sürümün, bu belge eklenmeden önce doğrulanan yerel commit'i
`d717d945e88c2f56796043ae12574cfc81b0fbe4`, kaynak ağacı
`8dd05967a86f3deabb395df766197975db21efef` idi. Paket kendi güncel kaynak
kimliğini RELEASE.json içinde taşır; güncel durumda onu esas alın.

## 4. Kaynak video ve etiketler

Yerelde 127 MOV video var. Kaynak klasör:
`/Users/niluferoden/Desktop/Internship dataset kopyası`

Veri kaynağı: https://data.mendeley.com/datasets/5g5hg9hrkv/2 .
Veri ve ön-eğitimli model lisansları kullanım/paylaşım sırasında ayrıca korunmalı.
Kaynak videolar özel paylaşım paketine ve GitHub'a eklenmedi.

Son bağımsız davranış etiketleme dosyası `labels_final.csv`; sonradan gönderilen
`labels_final kopyası.csv` ile hücre bazında aynı olduğu doğrulandı. 127 video
isimle birebir eşleşiyor. İki kişiyle konsensüs etiketlemesi yapılmış, ancak iki
kişinin ayrı uzlaşma öncesi ham formları mevcut pakette yok.

Sütunlar:

```text
video, durum, kopek_sayisi, gorunurluk, kuyruk_konum, kuyruk_hareket,
kulak, durus, hareket, ses, konsensus_disi, degisken, guvensiz, not
```

Eğitimde kullanılan metin kopyası:
`results/labels_v13/source_labels_extracted.csv`.
Bu dosya, kullanıcının yüklediği CSV'nin okunabilen metin içeriğinden oluşturuldu.
Raporlanan etiket SHA256 bu işlenen kopyaya aittir; yüklenen orijinalin bayt hash'i
olarak sunulmaz. Dosya içeriğinin hücre düzeyinde uyumu ayrıca doğrulandı.

### Altı hedef ve sabit kategori sırası

| Hedef | Sınıflar |
|---|---|
| `hareket` / Hareket | `hareketsiz`, `kosuyor_oynuyor`, `yuruyor` |
| `ses` / Ses | `havlama`, `hirlama`, `inleme`, `yok` |
| `durus` / Duruş | `ayakta`, `oturuyor`, `sirtustu`, `yatiyor` |
| `kuyruk_konum` / Kuyruk konumu | `asagi_bacak_arasi`, `notr`, `yukari` |
| `kuyruk_hareket` / Kuyruk hareketi | `sallamiyor`, `salliyor` |
| `kulak` / Kulak | `dik_one`, `geride_yassi`, `notr` |

Eski `aktivite/vucut/durus/kuyruk/kulak` şeması güncel değildir. Yeni dosyada
`vucut` bulunmaz; otomatik geri doldurulmadı. Hareket ile ses, kuyruk konumu ile
kuyruk hareketi ayrı hedeflerdir. Havlama hareket sınıfına dönüştürülmedi.

### Kalite politikası

- `guvensiz=1`: bütün hedefleri eğitim ve değerlendirme dışında; video galeride kalır.
- Boş, `gorunmuyor` veya ilgili `konsensus_disi` hedef: yalnız o hücre -1 ile maskelenir.
- Bir kulak etiketi eksik diye videonun hareket etiketi atılmaz.
- `degisken=1`: video boyunca davranış değişiyor; zaman damgalı gerçek etiket sağlamaz.
- Kalite, not, tarih, köpek sayısı, konsensüs ve insan etiketi model girdisi değildir.
- 11 güvensiz ve 36 değişken video var. Eski 30 etiketsizin 27'si güvenli eğitime katıldı.

### Kural türevi duygu dosyası (V16 öncesi, tarihsel)

Bu bölüm V16 öncesini anlatır. Artık dört duygu katmanı bu dosyayla değil, `emotion_human_labels.csv` ile eğitilir.

`duygu_kural_turevli.csv` dosyasında 127 kayıt bulunuyor. Her satırın kaynağı
“KURAL TÜREVLİ - insan duygu etiketi değil”. Dağılım: 37 Relaxed, 33 Happy,
23 Angry, 22 Sad, 12 boş. Eski beş hedefli şema ve `vucut` alanı kullanılıyor.

Bu dosya modelin gerçek duygu etiketi değildir. Eğitim veya duygu başarı hedefi
yapılmadı. Kurala ait aynı davranış girdileriyle o kural çıktısını tahmin edip
“duyguyu öğrendi” demek döngüseldir. İki dosyayı körlemesine birleştirmeyin:
güvenli kayıtlarda karşılaştırılabilir 116 duruş etiketinin 8'i ve 77 kulak etiketinin
14'ü farklıdır. Bunlar iki bağımsız etiketleyicinin kappa hesabı değildir.

## 5. Özellikler ve önbellek

Özellikler bütün 127 videoyu kapsar. Video kimliği ve SHA256 sırası denetlendi;
son etiketlerle yeniden eğitim için ham video/poz çıkarımı tekrarlanmadı.

- Görüntü: 471 boyutlu normalize poz/hareket özetleri.
- Ses: ortalama 1024 YAMNet gömmesi + 29 olay/erişilebilirlik özelliği = 1053 boyut.
- Zamansal dizi: 127 × 60 × 1141, 1 Hz, ilk en fazla 60 saniye.
- Dizi görsel bölümü: 39 noktanın normalize x/y koordinatları ve 39 geçerlilik maskesi = 117.
- Dizi ses bölümü: saniyeye hizalanmış 1024 YAMNet gömmesi.
- Gerçek uzunluklar ayrıca saklı; padding eğitim istatistiklerine ve zaman havuzlamasına katılmaz.

Başlıca önbellekler:

| Yerel yol | İçerik |
|---|---|
| `results/demo_v9/feature_snapshot.npz` | İlk 127 videoluk statik özellikler, eski etiketler ve kaynak kimlikleri |
| `results/demo_v11/sequence_snapshot.npz` | 127 video zaman dizileri ve gerçek uzunluklar; eski etiketler güncel hedef değildir |
| `results/demo_v13_release/feature_snapshot.npz` | Statik özellikler + diziler + son altı hedef + kaynak kimlikleri + ayrım indeksleri |
| `results/demo_v1/features/` | İlk videoların özellikleri ve zaman damgalı ham pose JSONL kayıtları |
| `results/demo_v4/new_features/` | Önceden etiketsiz 30 videonun özellikleri/pose kayıtları |
| `results/demo_v4/manifest.csv` | Orijinal 127 kimlik, kaynak hash'i ve tarihsel split |
| `results/demo_v13_release/manifest.csv` | Son etiketler, kalite ve güncel v13 split |

**Paket yolları:** Eski statik snapshot `artifacts/feature_snapshot.npz`, eski
zamansal snapshot `sequence_artifacts/sequence_snapshot.npz` altında da vardır.
Her iki snapshot'ın eski etiketleri yeniden eğitimde insan gerçeği diye kullanılmaz;
son CSV'den yeni altı hedef oluşturulur. Güncel birleşik snapshot asıl sonuç
klasöründeki yolunu korur.

## 6. Görüntü modelleri: eğitim ve gösterim aynı şey değil

Davranış eğitiminin önbellek özellikleri **YOLOv8n** köpek kutularından ve
**DeepLabCut SuperAnimal-Quadruped RTMPose-S** çıktılarından gelir. Model 39 nokta
üretir. Örnekleme yaklaşık 2 fps, ilk 60 saniye; özelliklerde her karede en büyük
köpek kutusu seçilebilir. Bu seçim insanın etiketlediği odak köpeğiyle veya önceki
karedeki bireyle aynı olmak zorunda değildir.

Galerideki daha temiz önizlemeler ayrı bir gösterim hattıyla YOLOv8s ve YOLOv8m
kurtarma, güven eşikleri ve optik akış kullanır. Daha güzel görünen galeri,
eğitim özelliklerinin de otomatik yenilendiği anlamına gelmez. Bu ayrımı koruyun.

39 noktanın resmî sırası doğrulandı; resmî skeleton bağlantı listesi boş.
Projedeki bağlantı çizgileri muhafazakâr özel gösterim rehberidir; doğrulanmış
anatomik yer doğruluğu veya keypoint benchmark'ı değildir. Çoklu köpek kutuları ve
poz örnekleri gösterilir; doğrulanmış sürekli birey kimliği/takibi yoktur.

### Segmentasyon

Yeni video yüklemesinde **YOLOv8s-seg** gerçek piksel maskeleri üretir. Kutu
şeklini maske diye sunmaz. Kare başına her örneğin ayrı ikili PNG maskesi,
zaman/kutu/skor/alan JSON'u ve segmentasyon videosu çıkarılır. Önizleme maskesi
örnek maskelerinin birleşimi olabilir. Numaralar kare içidir; kalıcı dog_id değildir.

İlk 60 saniyeye kadar, en fazla 8 fps, uzun kenar en fazla 960 piksel hedeflenir.
İnsan maske gerçeği yok: IoU/Dice ölçülmedi. **Davranış modelleri bu maskelerden
özellik çıkararak yeniden eğitilmedi.** Segmentasyon eklemek F1'i artırdı denmez.
Eski galeri önizlemelerinin tümüne yeni maskeler eklenmedi.

## 7. Ses hattı

YAMNet, 16 kHz mono sesin log-mel temsilleriyle çalışır; 521 ses olayı ve 1024
boyutlu gömme sağlar. Resmî HDF5 ağırlıklarını okuyan projeye ait, yalnız çıkarım
yapan PyTorch uyarlaması vardır. TensorFlow uygulamasıyla sayısal eşdeğerlik
ölçülmedi; resmî PyTorch sürümü olarak tanıtılmamalı.

Ses zaman çizelgesinde Dog, Bark, Growling, Whimper (dog), Howl, Speech, Music
adayları gösterilir. Eşik gösterim tercihidir, kalibre doğruluk yüzdesi değildir.
Birden çok olay aynı anda aday olabilir. Eğitim özellikleri ilk 60 saniyeyi kullanır.
Yeni video raporu da ilk 60 saniyeyle sınırlıdır; eski galeri ses raporları daha uzun
kayıtları kapsayabilir. Galeri önizlemesi çoğunlukla yalnız 6/12 saniyedir.

Ses kaynağı ayrıştırması ve tek köpeğe ses ataması yoktur. İnsan etiketleri odak
köpeğe aitken arka planda başka köpek havlaması olabilir; bu fark gerçek bir hatadır.

## 8. Eğitilen modeller

### FusionNet ve modalite ablasyonları

Statik görsel ve ses girdilerinin ayrı öğrenilebilir dalları vardır:
Linear(64) → LayerNorm → GELU → Dropout(0.35) → Linear(32) → GELU.
Birleşimde iki 32 boyutlu dal birleştirilir; ortak Linear(32)/GELU/Dropout ve
altı ayrı sınıflandırma başlığı kullanılır. Görüntü, ses ve birleşik ağlar ayrı eğitilir.

AdamW: lr=0.001, weight_decay=0.01, gradient clip=1, seed=42. Kayıp, görünmeyen/
eksik hedefleri maskeleyen sınıf ağırlıklı çapraz entropilerin görev ortalamasıdır.
Sınıf ağırlıkları, imputer ve normalizasyon yalnız ilgili eğitim katında öğrenilir.

### Zamansal modeller

Aynı 1 Hz poz+YAMNet dizilerinde CNN1D, GRU, BiGRU ve LSTM→LSTM→GRU var.
Görsel/ses girdileri 32'şer boyuta projekte edilip birleştirilir. GRU gizli boyutu 32;
BiGRU iki yönde 32'dir. Hibritte bütün tekrar eden katmanlar 32 birimdir.
Padding için gerçek uzunluklar kullanılır. BiGRU çevrimdışı gelecekteki bağlamı da
kullanır; nedensel canlı model değildir.

CNN1D iki kernel-3 zaman konvolüsyonu ve maskeli ortalama havuzlama kullanır.
Statik MLP ile dizi mimarileri hem mimari hem temsil bakımından değişir; bunu
saf mimari ablasyonu diye sunmayın. Dizi modelleri kendi aralarında aynı girdiyi kullanır.

### Klasik referanslar

Görsel, ses ve birleşik özelliklerde LR/RF karşılaştırıldı. LR: C=1, balanced,
max_iter=2000. RF: 150 ağaç, min_samples_leaf=2, balanced, random_state=42.
İmputer/scaler kat içinde öğrenilir. Çoğunluk tahmini ayrıca gösterilir.

### İncelenen makaleden alınan fikir

Multi-QuadEmoNet, DOI https://doi.org/10.3389/fvets.2026.1799281 . İki LSTM ardından
GRU fikri denendi. Bizim uygulamamız MFCC girdili orijinal çalışmanın birebir
tekrarı değildir; poz+YAMNet, farklı katman boyutları ve davranış hedefleri kullanır.
Makaledeki yaklaşık %95 iki sınıflı ses doğruluğu bizim altı görev macro-F1
ortalamasıyla doğrudan kıyaslanamaz. Bkz. docs/paper_review_20261114.md.

## 9. V13 tarihsel dosya bölmesi

Eski 25 test dosyası eğitim dışında tutuldu. Bunların ikisi yeni dosyada güvensiz;
23 videoda ölçüm var. 93 güvenli eğitim videosu kullanıldı. Test üyeliği ile
etiketi geçerli test desteği farklıdır. Üç eğitim-içi GroupKFold katında 20/40/60/80
epoch ve klasik LR/RF seçimi yapıldı. Gruplama SHA256 olduğundan aynı hayvanın
farklı kayıtlarını ayırmaz. Test geçmiş geliştirmelerde tekrar tekrar incelendi.

| Hedef | Eğitim etiketi | Ölçülebilen test etiketi |
|---|---:|---:|
| Hareket | 93 | 23 |
| Ses | 93 | 23 |
| Duruş | 93 | 23 |
| Kuyruk konumu | 70 | 17 |
| Kuyruk hareketi | 90 | 23 |
| Kulak | 63 | 14 |

| Model | Eğitim içi CV macro-F1 | V13 test macro-F1 |
|---|---:|---:|
| `neural_visual` | 0.458 | 0.449 |
| `neural_audio` | 0.463 | 0.368 |
| `neural_fusion` | 0.490 | 0.442 |
| `sequence_cnn` | 0.482 | 0.410 |
| `sequence_gru` | 0.393 | 0.402 |
| `sequence_bigru` | 0.423 | 0.364 |
| `sequence_lstm_lstm_gru` | 0.372 | 0.406 |
| `classical_visual` | 0.467 | 0.514 |
| `classical_audio` | 0.464 | 0.379 |
| `classical_fusion` | 0.508 | 0.478 |
| `majority_baseline` | — | 0.240 |

Eğitim CV seçimi `classical_fusion` oldu. Testte görüntü referansı daha yüksek
çıktı diye seçimi değiştirmedik. Canlı uygulama mevcutsa klasik birleşik modeli
öne çıkarır; bütün diğer modeller karşılaştırılır.

Bu skor altı hedefin sabit kategori listeleri üzerinden macro-F1 ortalamasıdır.
Testte örneği bulunmayan sınıflar da sıfır katkıyla hesaba katılır. Eğitimde hırlama
1, oturma 2, sırtüstü 2 kayıttır. “Doğruluk %44” diye çevrilmemelidir.
Eski beş hedefli 0.32–0.47 sonuçlarla doğrudan artış hesabı yapılmaz.

## 10. V14 gün gruplu yeniden doğrulama

Videoların gömülü metadata'sı orijinal dosyalardan okundu. Dosya adı veya disk
mtime/ctime çekim günü gibi kullanılmadı. 114 kayıtta kamera creationdate, 9'unda
yalnız container creation_time, 4'ünde hiç tarih yok. 17 videoda saat dilimi
hizalandıktan sonra da kamera ve container günü farklı. Kamera-yerel gün tercih edilir.
Örneğin IMG_5592 kamera tarihi 22 Haziran, container tarihi 1 Temmuz.

Yalnız kamera creationdate olan ve güvensiz olmayan **103** video ana gruplu
değerlendirmede kullanıldı. 13 belirsiz/container-only ve 11 güvensiz kayıt dışarıda;
galeriden silinmedi. Güvenli gün dağılımı: 18 Haziran 60, 20 Haziran 4, 22 Haziran 39.

Dış kat: bir günün tüm videoları dışarıda. İç kat: kalan iki gün üzerinde
leave-one-day-out ile epoch/LR-RF seçimi. Ölçekleme ve sınıf ağırlıkları yalnız
katın eğitim verisinde. Dış gün skoruyla seçim yapılmadı. Her uygun video bir kere
dışarıda kaldı. Kaynak hash'i ve gün gruplarının ayrıklığı denetlenir.

| Model | Birleştirilmiş dış-kat macro-F1 | Günlere eşit ağırlıklı macro-F1 |
|---|---:|---:|
| `neural_visual` | 0.344 | 0.284 |
| `neural_audio` | 0.296 | 0.232 |
| `neural_fusion` | 0.330 | 0.275 |
| `sequence_cnn` | 0.313 | 0.268 |
| `sequence_gru` | 0.308 | 0.239 |
| `sequence_bigru` | 0.279 | 0.250 |
| `sequence_lstm_lstm_gru` | 0.261 | 0.244 |
| `classical_visual` | 0.344 | 0.297 |
| `classical_audio` | 0.283 | 0.254 |
| `classical_fusion` | 0.346 | 0.318 |
| `majority_baseline` | 0.215 | 0.245 |

İlk skor bütün dış-kat tahminleri birleştirilince hesaplanır. İkinci skor önce
her günü ölçer, sonra günlere eşit ağırlık verir. Dört videolu gün büyük günle eşit
ağırlık aldığı için fark önemlidir. Bazı iç eğitim katlarında yalnız dört video vardır;
model seçimi çok oynaktır. Kat başına sınıf destekleri ham JSON'da yer alır.

**Bu kamera tarihleri onaylanmış oturum veya köpek kimliği değildir.** Aynı köpek
farklı günlerde bulunabilir. Daha önce incelenmiş aynı dosyaları yeniden bölmek
bağımsız yeni test yaratmaz. Doğru ad, geçmiş veri üzerinde gün gruplu yeniden
doğrulamadır. Üç gün genel üstünlük/istatistiksel anlamlılık iddiası için güçlü veri değildir.

V13'ün 23 test videosuyla V14'ün 103 dış-kat tahmini farklı kapsamdır. Skor farkını
yalnız bölmenin nedensel etkisi diye yorumlamayın. Gün gruplu deneyler canlı kontrol
noktalarını değiştirmedi; galeri ayrı tablo gösterir. Video yanındaki tahminler hâlâ
V13 demo modelleridir, V14 out-of-fold tahminleri değildir.

## 11. Kappa ve oturum bilgisi

Kappa hesaplanmadı; mevcut iki CSV iki etiketleyicinin ham kararlarını sağlamıyor.
Konsensüs kopyalarını veya insan davranışı ile kural duygusunu karşılaştırmak kappa
olarak uygun değildir. `konsensus_disi` bayrağından da kappa çıkarılamaz.

Gerekli: iki kişinin aynı video/aynı hedef için bağımsız, uzlaşma öncesi formları.
Kod hazır: `src/evaluation/annotation_agreement.py`. `video` ve altı hedefi ister;
konsensüs/kural tablolarını reddeder. Hedef bazında eşleşen gözlenen çiftleri,
ham uyumu ve ağırlıksız nominal Cohen kappa'yı verir. Tanımsız kappa None'dır.

Oturum taslağı: `results/recording_day_audit_v14/recording_days.csv`.
`confirmed_session_id` boş, `annotator_confirmed` false. İnsan tarafından
onaylanmadan bu alanlar doldurulmuş gibi kullanılmaz. Mevcut gün doğrulayıcı
kamera tarihlerini kullanır; gerçek oturum eşlemesi gelince bölme kodu ayrıca
bu gruplamaya uyarlanıp iç/dış ayrıklık denetlenmelidir.

## 12. Uygulama ve HTML nasıl çalışır?

Ana klasörde:

- `BASLA.html`: bütün girişleri gösteren küçük başlangıç sayfası.
- `DEGERLENDIRME.html`: yöntem eleştirisi ve V14 sonuçlarının okunur raporu.
- `KURULUM.html`: okunur kurulum/paylaşım rehberi.
- `START_ANALYSIS.command`: Mac başlatıcı; `START_ANALYSIS.bat`: Windows başlatıcı.
- `results/demo_v13_release/public/canine_demo_v13.html`: yaklaşık 213 MiB, 127 videolu
  gömülü önizlemeli tek HTML galeri. Kaynak şablonu `src/demo/labels_gallery.html`.

Çalışan yerel adresler:

| Adres | İşlev |
|---|---|
| `http://127.0.0.1:8768/start` | Başlangıç |
| `/analyze` veya `/` | Yeni video seç/yükle/analiz |
| `/gallery` | Hazır galeri ve yükleme iframe paneli |
| `/evaluation` | Değerlendirme HTML |
| `/setup` | Kurulum HTML |
| `/presentation` | Sunum |
| `/api/health` | Sürüm/yetenek |
| `/api/jobs` | Yerel analiz geçmişi |

8766 eski bir uygulama oturumu olabilir. Bu oturumda eski dinleyici durdurulamadı;
yeni uygulama 8768'ye alındı. Bağlantı noktası aktif config içinde yazılıdır.

**Çift tıklanan HTML tek başına derin öğrenme modellerini çalıştırmaz.** Hazır
galeri çevrimdışı açılır; yeni video analizi Python sunucusu ister. Diskten açılan
kaynak upload.html artık JavaScript hatasıyla sessiz kalmaz; kullanıcıyı başlatıcıya
ve çalışan analiz adresine yönlendirir. Sunucuyla açılan galeri kendi içinde yükleme
paneli açar. Kaynak upload.html'yi ana uygulama dosyası sanmayın.

Yeni analiz en fazla 250 MiB ve ilk 60 saniye; MOV/MP4/MKV/WEBM/AVI kabul eder.
Video uzantısı tek başına güvenilmez; gerçekten çözümlenebilir olması kontrol edilir.
Kaynak kopyası UUID iş klasöründe tutulur, orijinal dosya değiştirilmez. H264/AAC
oynatılabilir önizleme üretilir; HTTP Range desteklenir. Tek işçiyle kuyruk işler.

Normal sesli/pozlu yüklemede karşılaştırmalar: FusionNet, tek modlu görüntü ve ses
sinir ağları, üç klasik modalite, CNN1D, GRU, BiGRU ve LSTM–LSTM–GRU: **11** seçenek (yeni önerilen karar birleşimi dahil).
Ses yoksa birleşik modelleri uydurmaz; kullanılabilir görüntü ağı sonuç verir.
Poz yok ama ses varsa kullanılabilir ses ağı çalışabilir. Hiçbir kullanılabilir
modalite yoksa sonuç bulunamadığı belirtilir. Skorlar kalibre olasılık değildir.
Kısa pencereler deneysel FusionNet çıkarımlarıdır; doğru zaman etiketleriyle ölçülmedi.

Sunucu loopback'e bağlıdır; Host/Origin ve oturum token'ı denetlenir. HTML'in diskten
sunucuya doğrudan CORS atlatması eklenmedi. Videolar dış servise gönderilmez;
yerel sunucuya aktarılır. Analiz geçmişi `results/live_analysis/jobs/` altındadır;
kişisel yüklemeler Git'e veya paylaşım paketine alınmaz.

## 13. Kod haritası

| Dosya / klasör | Görevi |
|---|---|
| `src/deep_learning/final_labels.py` | Son şema denetimi, maskeleme, yeniden eğitim, CV ve sonuçlar |
| `src/deep_learning/train.py` | FusionNet, normalizasyon/kayıp yardımcıları; tarihsel beş hedef sabiti de burada |
| `src/deep_learning/predict.py` | Ham videodan neural çıkarım; hedef sırasını metadata'dan okur |
| `src/deep_learning/sequence.py` | CNN/GRU/BiGRU, dizi kurma ve gerçek uzunluk/padding |
| `src/deep_learning/paper_hybrid.py` | Makale esinli küçük LSTM-LSTM-GRU ve eski kontrollü deney |
| `src/evaluation/recording_days.py` | Gömülü tarih denetimi; saat dilimi duyarlı |
| `src/evaluation/day_validation.py` | İç içe gün gruplu yeniden doğrulama |
| `src/evaluation/annotation_agreement.py` | Bağımsız ham formlar gelirse kappa |
| `src/demo/analyze_upload.py` | Gerçek yeni video analiz işçisi |
| `src/demo/upload_server.py` | API, token, kuyruk, güvenli dosya uçları |
| `src/demo/upload.html` | Sunucuya bağlı gerçek yükleme arayüzü |
| `src/demo/labels_gallery.html` | Aktif 127-video galeri şablonu |
| `src/demo/launch.py`, `release.py` | Ortam denetimi, aktif sürüm ve başlatma |
| `src/preprocessing/segmentation.py` | Gerçek köpek maskelerini doğrulayan YOLO adaptörü |
| `src/demo/segmentation.py` | Segmentasyon videosu, maske PNG/JSON/ZIP |
| `src/audio_analysis/yamnet.py` | YAMNet PyTorch çıkarım uyarlaması |
| `src/demo/extract.py` | Donmuş görsel özellik çıkarımı |
| `scripts/build_final_labels_gallery.py` | Hazır galeriyi dışa aktarma |
| `scripts/build_final_labels_defense.py` | Güncel 15 slayt |
| `scripts/export_defense_pdf.py` | Sunum PDF'i ve sayfa PNG kontrolü |
| `scripts/build_start_pages.py` | Kök başlangıç/rapor HTML dışa aktarma |
| `scripts/package_release.py` | Küçük/tam paylaşım arşivi ve checksum |
| `tests/` | Birim/entegrasyon testleri; bunlar model başarısının kanıtı değildir |

`src/demo/index.html` ve pek çok tarihli belge eski deneyler içindir. Güncel
aktif sürüm için config'i esas alın. Eski beş hedefli `TARGETS` sabitini körlemesine
kullanmayın: son model hedef sırası checkpoint metadata'sından okunmalıdır.

## 14. Ortam ve çalıştırma

Test edilen çıkarım ortamı Mac Apple Silicon, Python 3.11.17, CPU. Eski notlardaki
NumPy/OpenCV sürümleri güncel ortam sanılmamalıdır. `requirements-share.txt` güncel
çalışan temel sürümleri sabitler; tam geçişli ve platformlar arası lock dosyası değildir.
Yeni Windows/Linux temiz kurulumu ayrıca doğrulanmadı. Mac .venv Windows'a kopyalanmaz.

Yeni bilgisayarda ZIP'i çıkarın; ZIP içinden program çalıştırmayın:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-share.txt
python -m src.demo.launch --check-only
python -m src.demo.launch
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-share.txt
.\.venv\Scripts\python.exe -m src.demo.launch --check-only
.\.venv\Scripts\python.exe -m src.demo.launch
```

Sunucu çalıştığı Terminal açık kalmalı. Hazır HTML okuyabilmek için kurulum gerekmez.
Ön-eğitimli ve eğitilmiş ağırlıklar GitHub'da yok; yalnız kod indirmek yeni video
analizi için yeterli değildir. Tam ZIP'teki models/ ve results/ kontrol noktaları gerekir.

## 15. Yeniden eğitim / tekrar komutları

### Orijinal çalışma klasöründe V13 tekrar

```sh
python -m src.deep_learning.final_labels --output results/demo_v13_repeat
```

### Paylaşım paketinde V13 tekrar

```sh
python -m src.deep_learning.final_labels --features artifacts/feature_snapshot.npz --sequences sequence_artifacts/sequence_snapshot.npz --metadata artifacts/neural_fusion.json --output results/demo_v13_repeat
```

Eski CSV manifesti ve son CSV metni paket içinde beklenen yollarını korur.
Yeni bir çıktı klasörü adı seçilir; mevcut deneyin üzerine sessizce yazılmaz.

### Kaydedilmiş metadata ve güncel özelliklerle V14 tekrar

```sh
python -m src.evaluation.day_validation --output results/day_validation_repeat
```

Orijinal videolarla tarih denetimini yeniden yapmak gerekiyorsa:

```sh
python -m src.evaluation.recording_days --output results/day_metadata_repeat
```

Farklı bilgisayarda manifestteki `video_path` yolları orijinal Mac'e ait olabilir.
Metadata/ham özellik yeniden çıkarımı için önce kendi gerçek video yollarına
uyarlayın. Hazır özelliklerle yeniden eğitim yolu bu ham video dosyalarını okumaz.

### Ham etiketleyici formları geldiğinde

```sh
python -m src.evaluation.annotation_agreement --left etiketleyici_a.csv --right etiketleyici_b.csv --output results/kappa.json
```

### Denetim ve çıktılar

```sh
python -m pytest -q
node tests/final_labels_ui_smoke.cjs
python scripts/build_start_pages.py
python scripts/build_final_labels_gallery.py
python scripts/build_final_labels_defense.py
python scripts/export_defense_pdf.py
python scripts/package_release.py --full
```

Galeri yeniden üretimi ham önizleme MP4'lerini ve önceki ses raporlarını ister;
bunlar büyük paylaşım ZIP'ine dahil değildir. Yalnız paketle galeri yeniden
üretilemez. Hazır büyük HTML ayrıca verilir; yeni video analizi ondan bağımsızdır.
Sunum yeniden üretimi güncel raporları ve tarihsel sunum şablonunu isteyebilir;
pakette hazır PDF/HTML bulunur. Bütün scriptlerin paketteki her ortamda otomatik
çalıştığını varsaymayın; ihtiyaç duyulan ham yardımcı varlıkları kontrol edin.

## 16. Doğrulanan işler ve somut kanıtlar

- Son kontrolde **381 Python testi** geçti.
- Minimal DOM ile gerçek galeri JavaScript'i çalıştırıldı: 127 video, 30 eski
  etiketsiz, 25 korunmuş tarihsel test filtresi, 11 referans/model ve gün tablosu.
- Galeride yükleme paneli açılması ve diskten açılan upload şablonunun sessizce
  bozulmak yerine başlatıcıya yönlendirmesi denetlendi. Gerçek tarayıcı tıklama
  otomasyonu mevcut değildi; bu kontrolü yapılmış gibi sunmayın.
- Gerçek HTTP dosya yükleme üzerinden sesli kısa klip: 10 model seçeneği,
  altı hedef, segmentasyon ve Range oynatma doğrulandı.
- Ses kanalsız kısa klip: görüntü sinir ağı sonuç verdi, birleşik davranış uydurulmadı.
- Kaydedilmiş yedi neural kontrol noktasının 127×6×7 = 5334 video-hedef tahmini
  yeniden üretildi. Bu, yeni hayvanda başarı ölçümü değil yeniden çıkarım kontrolüdür.
- V14'te 103 videonun her biri bir kez dışarıda kaldı; 11 modelin toplu skorları
  kaydedilmiş sınıf tahminlerinden yeniden hesaplandı.
- Paylaşım ZIP'inde CHECKSUMS.json ve RELEASE.json var; tüm dosya checksum'ları
  önceki paketlemede denetlendi. Yeni paket alındığında tekrar doğrulanabilir.

Kanıt dosyaları:
`results/demo_v13_release/verification.json`,
`results/demo_v13_release/upload_smoke.json`,
`results/ui_upload_fix_verification.json`,
`results/day_validation_v14/report.json`.
Kişisel yükleme geçmişi ve ham yükleme dosyaları ZIP'e girmez; bazı smoke raporları
her paket sürümünde taşınmayabilir. Ana yöntem raporları ve model sonuçları taşınır.

## 17. Açık eksikler ve öncelikli çalışma

| Öncelik | Açık / gerekçe | Uygun sonraki adım |
|---|---|---|
| 1 | Odak köpek ile en büyük kutu uyuşmayabilir | İnsan onaylı odak kutusu/kimlik, bireye ait poz dizisi ve takip doğrulaması |
| 1 | Ses odak köpeğe ait olmayabilir | Ses etiketi kapsamını netleştir; kayıt/odak düzeylerini ayır; bireysel ses gerçeği olmadan atama iddiası kurma |
| 1 | Kappa yok | İki bağımsız uzlaşma öncesi ham etiket formunu al; hedef bazında çift sayısı ve kappa raporla |
| 1 | Onaylanmış oturum/kimlik ve bağımsız yeni test yok | Video–oturum eşlemesini doğrulat, mümkünse yeni kayıtlarla son test oluştur |
| 2 | Üç gün ve nadir sınıflar çok az | Daha fazla bağımsız gün/hayvan ve nadir sınıf örneği; çok tohumlu duyarlılık analizi |
| 2 | Segmentasyon davranış eğitimine katılmadı | Maskeli yeni temsil çıkar, aynı kontrollü protokolle yeniden eğit ve referansla karşılaştır |
| 2 | Poz sıçramaları ve çoklu köpekte karışma | Odak kimlik + güvenilirlik filtreleri; gerekirse insan keypoint doğrulama altkümesi |
| 2 | YAMNet portu sayısal parity ölçülmedi | Aynı girdiyle resmî TensorFlow çıktısıyla eşdeğerlik testi |
| 2 | Anlık davranış ground truth yok | Zaman aralığı bazlı insan etiketleri; pencere modellerinin ayrı değerlendirilmesi |
| 3 | Küçük veride model seçimi oynak | Aynı temsil/girdiyle kontrollü ablasyon, grup düzeyinde belirsizlik; test kazananını seçmeme |
| 3 | Paket kurulumunun diğer platformları doğrulanmadı | Temiz Windows/Linux ortamında dependency ve gerçek video smoke testi |
| Ayrı araştırma | Duygu etiketleri insan etiketli ama özneldir (kappa 0,553), bağımsız uzman etiketi yok | Uygun bağımsız uzman duygu etiketleri ve kapsam tanımı; kuralları insan gerçeğine dönüştürmeme |

Bu sıradaki bazı işler yeni veri/insan girdisi ister. Veri yokken daha karmaşık
bir ağ eklemek onları çözmez. Mevcut sonuçları daha güzel görünmesi için yeniden
etiketlemek veya test skoruna göre video/sınıf seçmek yapılmamalı.

## 18. Gönderilecekler: net liste

### Her yeni yapay zekâya önce

1. **Bu AIYE_ILET.md dosyası.** Büyük kod paketini görmeden bile bağlamı anlar.
2. **canine_tam_proje_paketi.zip** veya tam paketi alamıyorsa **canine_sunum_paketi.zip**.
   Küçük paket kaynak/rapor/özellik/eğitilmiş başlıkları içerir; yeni ham video
   analizi için gerekli ön-eğitimli büyük ağırlıklar yalnız tam pakettedir.
3. Son orijinal **labels_final.csv / labels_final kopyası.csv** dosyasını ayrıca
   eklemek yararlı; işlenen kopya zaten pakettedir. Hangisinin insan konsensüsü
   olduğunu açıkça söyleyin.

### İhtiyaca göre ayrıca

- Görsel arayüzü inceleyecekse **canine_demo_v13.html** (yaklaşık 213 MiB).
  Büyük boyut yüzünden her yapay zekâ kabul etmeyebilir; kod incelemesi için şart değil.
- Hocaya sunumu düzenleyecekse **canine_sunum.pdf** veya `docs/hoca_sunumu.html`.
  Hazır sunum ZIP içinde de bulunur.
- Yeni özellik/detektör/poz/segmentasyon deneyi yapacaksa **127 orijinal video**
  veya üzerinde anlaşılmış gerçek altküme. Ham veri ilk aşamada gerekli değil.
- Kappa için **iki bağımsız ham etiket formu**; oturum bölmesi için **onaylı
  video–oturum–varsa hayvan kimliği eşlemesi**. Mevcut pakette bunlar bulunmuyor.
- `duygu_kural_turevli.csv` yalnız inceleme isteniyorsa, açıkça **kural türevi,
  eğitim gerçeği değil** notuyla. Model eğitimi için zorunlu gönderim değil.

Göndermeye gerek olmayanlar: .venv, önbellek klasörleri, kişisel yüklenen videolar,
token/şifre/.env, eski tekrarlı ZIP'ler ve proje dışı masaüstü dosyaları.
Ham videolar veya model ağırlıkları GitHub'a yüklenmemeli. GitHub özel depoya
arkadaşın/yapay zekânın yetkisi yoksa sadece URL vermek dosyalara erişim sağlamaz.

## 19. Paketi alan yapay zekânın ilk kontrol listesi

- RELEASE.json ve CHECKSUMS.json oku; kök klasör/aktif config ile sürümü doğrula.
- Son etiket CSV'sini video_id ve hash listesine eşleştir; 127 satır, tekrar yok mu?
- Hedef sırası altı görev mi, eski vucut/aktivite/kuyruk başlığına yanlış dönülmüş mü?
- İnsan etiketleri ile kural türevi duygu veya model önerileri karışmış mı?
- V13 ve V14'ü ayrı değerlendir; destek sayılarını, gün gruplarını, kat seçimlerini oku.
- Donmuş snapshot dizilerinin sırası/etiketleri/ölçekleyicileri tutarlı mı?
- Canlı tahminlerin hangi modelden ve hangi sürümden geldiğini kontrol et.
- Galeri, yeni video ve raporların çalışmasını gerçek dosyayla denetle.
- Paket dışı ham video/önizleme varlığı gerektiren scriptleri çalışmadan önce ayır.
- Yapılmış/planlanmış/ölçülmemiş işleri ayrı yaz; bu belgedeki eksik verileri uydurma.

## 20. Kısa akademik anlatım örneği

Bu çalışmada köpek videolarından ön-eğitimli köpeğe özel poz ve ses temsilleri
çıkarılarak altı gözlenebilir davranış/ses hedefi için çok görevli sinir ağları ve
klasik referanslar karşılaştırılmıştır. Konsensüs dışı, görünmeyen ve güvensiz
etiketler uygun düzeyde maskelenmiştir. Önceden incelenmiş dosya bölmesinin
sınırlılıkları nedeniyle üç kamera günü üzerinde iç içe gün gruplu yeniden
doğrulama da yapılmıştır. Veri azlığı, birey/oturum doğrulaması ve bağımsız duygu
etiketlerinin bulunmaması nedeniyle sonuçlar keşifsel davranış tahmini olarak
yorumlanmaktadır; doğrulanmış duygu sınıflandırması veya genel model üstünlüğü
iddiası kurulmamaktadır.
