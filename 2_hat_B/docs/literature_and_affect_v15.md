# Literatüre dayalı model seçimi ve ses duygu modeli · v15

14 Kasım 2026. Yerel altı davranış hedefi ve dış veride insan algısı ile etiketli
iki duygu boyutu ayrı hedeflerdir. Bu sürüm davranıştan duygu kuralı üretmez.

## Araştırma sonucu ve öneri

Tek bir evrensel en iyi model gösterilemiyor; veri, hedef ve hayvan ayrımı değişiyor.
Büyük ağa geçmenin tek başına yüksek başarı garantisi yok. Mevcut küçük veri için
önerimiz: donmuş poz/ses temsilleri, eğitim içinde seçilen karar birleşimi ve ayrı
insan algısı etiketleriyle öğrenilen ses duygu başlığı. Diğer modeller uzman
karşılaştırmasında kalır; ana ekranda bir önerilen davranış ve bir duygu değerlendirmesi var.

İncelenen birincil kaynaklar:

| Çalışma | Yöntem / bulgu | Bizim projeye etkisi |
|---|---|---|
| Abzaliev ve ark., LREC-COLING 2024 | Wav2Vec2 ile havlama bağlamı ve kimlik görevleri; ön-eğitim yararlı | Transfer öğrenmesi anlamlı; bağlam başarısı duygu başarısı değildir |
| Li ve ark., Interspeech 2025, Dog2vec | Köpeğe özel HuBERT temsili; havlama türü/tespit görevleri | Gelecek kodlayıcı adayı; hazır duygu sınıflandırıcısı değil. 1.1 GB/fairseq modeli bu sürümde çalıştırılmadı |
| Garcia-Loya ve ark., Pattern Recognition Letters 2025 | Görsel, atalet ve fizyolojik sinyaller; ExtraTrees ve yüksek F1 | Onların sensörleri bizde yok; bildirilen skoru bizim video/ses başarısı gibi alamayız |
| Dang ve ark., ACM MM 2025, EmotionalCanines | Üç sınıflı valence/arousal; eGeMAPS, Whisper, HuBERT, Wav2Vec2 ve klasik referanslar | Bağımsız insan algısı etiketi kaynağı. Yazarların ayrı köpek test sonuçları kabaca %50–57 düzeyinde; her problemde %95 beklenmemeli |
| Baek ve ark., CREMD 2026 ön baskı | İnsan duygu algısı, bağlam ve etiketleyici uyumu | İnsan algısı ile gerçek iç durum ayrımı; bağlam ve annotator güvenilirliği önemli |
| scikit-learn VotingClassifier belgeleri | Ağırlıklı yumuşak oy, iyi kalibre üyeler için uygun | Üye skorları kalibre değil; karışım skoru doğruluk yüzdesi olarak sunulmaz |

Bu tablo sistematik bir yayın-sıklığı sayımı değildir. Görsel transfer CNN/poz,
ses tarafında MFCC/eGeMAPS ve ön-eğitimli temsiller, klasik sınıflandırıcılar ve
zamansal ağlar kullanılan yöntem aileleridir. Buradaki seçim mevcut veride yapılan
deneye dayanır; literatürde kullanılan adı büyük diye bir model etkinleştirilmedi.

## 1. Davranış modellerini karıştırma deneyi

Özellikler, altı hedef ve aynı 103 güvenli kamera-tarihli video korundu. V14 ile
aynı dış günler (60/4/39) ve aynı iç günler kullanıldı. Önceki iç eğitim seçimleriyle
üyeler oluşturuldu: klasik görüntü, klasik ses, klasik erken birleşim, neural fusion.
9 sabit karışım reçetesinden biri yalnız iç doğrulama ortalamasıyla seçildi.
Dış gün skoruna göre ağırlık seçilmedi. Tek tohum, daha önce incelenmiş veri ve
çok az grup sınırları aynen devam eder.

| Aynı 103 video | Ortalama dış-kat macro-F1 |
|---|---:|
| Önceki klasik birleşik | 0.346 |
| Önceki neural fusion | 0.330 |
| Yeni karar birleşimi | 0.370 |

Dış günlerde seçilen reçeteler:

| Gün | İç eğitimde seçilen reçete | Dış gün macro-F1 |
|---|---|---:|
| 2026-06-18 | visual75_audio25 | 0.343 |
| 2026-06-20 | classical_fusion | 0.295 |
| 2026-06-22 | visual25_audio75 | 0.350 |

Canlı uygulama reçetesi ayrıca, yalnız eski 93 eğitim videosunun üç katlı CV'sinde
seçildi: **%75 klasik görüntü + %25 klasik ses**. Teste bakıp görev bazında farklı
ağırlıklar uydurulmadı. Yeni uygulama bu kararı öne çıkarır. Gerekli modalite yoksa
karışım uydurulmaz; uygun tek modlu ağ kullanılır.

Bu kazanç keşifseldir; istatistiksel olarak kanıtlanmış genel üstünlük demek değildir.
Yeni test yoktur. Karışım ağırlıkları kişiye/hayvana doğrulanmış güven düzeyi değildir.

## 2. Gerçek etiketli yeni ses duygu öğrenmesi

Kaynak, yazarların EmotionalCanines GitHub sürümü:
`6b73c5fce55fc132e97db6a76899f8594a955f59`.
1.400 WAV ve dört etiket CSV'si alındı. Her WAV'ın LFS nesne SHA256'sı ve boyutu
pinned sürümdeki işaretçiyle karşılaştırıldı. Veri CC-BY-NC-SA-4.0 olarak yayımlanmış;
kaynak/attribution lisans dosyası korunur. Ham dış veri Git'e ve paylaşım ZIP'ine eklenmez.

Etiketler doğrudan yazarların insan algısı CSV'lerinden okunur:

- Valence: Negative / Neutral / Positive → Olumsuz / Nötr / Olumlu.
- Arousal: Low / Medium / High → Düşük / Orta / Yüksek uyarılma.

Yerel `duygu_kural_turevli.csv` hiçbir eğitim aşamasında kullanılmadı. (V16: dört duygu katmanı artık yerel insan duygu etiketleriyle eğitilir; bu ses modeli ayrı kalır.)
Bu yeni ses hedefi altı davranış etiketinden türetilmedi. Bununla birlikte kaynak
etiketler insan algısıdır; hayvanın kesin iç öznel durumunun ölçümü değildir.

### Kaynak veri denetiminde bulunan sorunlar

Ham sürümde 1.400 satır ama 1.395 benzersiz ses hash'i var. İki hash eğitim ile
test arasında ortak. Çelişkili eğitim hash etiketleri ve tekrarlı eğitim kaydı da var.
Aynı test sesini eğitime sokmamak için bu kayıtlar karantinaya alındı. Teste dokunarak
daha kolay örnek seçilmedi; resmi 200 test satırı korundu (199 benzersiz ses).
Ayrıca yayımlanan gerçek dosyalarda süre 0.064–30.016 saniye; 13 kayıt 10 saniyeden
uzun. Kağıttaki genel süre tanımını dosya denetimi yerine varsaymadık.

Temiz eğitim: **1195**; resmi test: **200**. Karantinadaki beş eğitim kaydı: `husky_train_00291`, `husky_train_00572`, `husky_train_00577`, `shiba_train_00238`, `shiba_train_00421`.

Yazarlar test köpeklerinin eğitimden ayrıldığını bildiriyor. Yayımlanan CSV'lerde
köpek kimlikleri yok; bu kimlik ayrımını bağımsız doğrulayamadık. İç CV grupları
ses SHA256'sıdır, köpek değildir. Bu yüzden iç CV iyimser olabilir. İki ırkın
birlikte eğitilmesi de yazarların ırk başına deneyinin birebir tekrarı değildir.

### Temsil ve eğitim

Donmuş YAMNet'in 1024 boyutlu gömmelerinin ortalama ve standart sapması birleştirilir
(2048 özellik). LR (C=.1), RF (200 ağaç/min leaf=3/balanced), iki başlıklı küçük
MLP ve LR+MLP eşit oy karşılaştırıldı. MLP: 2048→64/LayerNorm/GELU/Dropout .5→32/
GELU/Dropout .3→iki üç sınıflı başlık. AdamW .001/.01, grad clip 1, seed 42.
Ölçekleme ve sınıf ağırlıkları yalnız eğitim katlarından öğrenildi.
MLP epoch 20/40/80 ve son model, eğitim içindeki 3 stratified-hash CV katında seçildi;
seçim dosyası test skorları hesaplanmadan yazıldı.

| Model | Eğitim CV macro-F1 | Test ort. macro-F1 | Duygu yönü doğruluğu | Uyarılma doğruluğu |
|---|---:|---:|---:|---:|
| neural | 0.451 | 0.469 | 45.5% | 49.0% |
| lr | 0.409 | 0.432 | 38.5% | 48.0% |
| rf | 0.458 | 0.477 | 43.0% | 52.5% |
| soft_vote | 0.420 | 0.453 | 40.5% | 50.0% |


İki boyutun aynı anda doğru olması (joint accuracy):

| Model | Her iki boyut doğru |
|---|---:|
| neural | 23.5% |
| lr | 17.5% |
| rf | 23.5% |
| soft_vote | 19.0% |

Tek bir genel durum etiketi iki tahmini birlikte içerdiğinden bu oran ayrıca önemlidir. İki ayrı doğruluk yüzdesi genel ruh hâlinin aynı oranda doğru olduğu anlamına gelmez.

Eğitim CV seçimi **RF**. Testte de bu başlık en yüksek ortalamayı aldı ama seçim
sebebi test değil eğitim CV'sidir. Puanlar düşük/orta düzeyde; yeni köpeklerde kesin
ruh hâli bulduğumuz iddia edilmez. Yerel 127 videoda duygu etiketleri olmadığından
bu veri üzerinde duygu doğruluğu ölçülmedi. Dış ses testini yerel davranış F1'iyle
veya V14 gün tablosuyla tek tabloda aynı hedefmiş gibi birleştirmeyin.

## 3. Yeni videoda ana sonuç nasıl oluşur?

1. Mevcut köpek/poz, YAMNet ve segmentasyon analizi çalışır.
2. Altı davranış hedefinde eğitim CV seçilmiş %75 görüntü/%25 ses kararı öne çıkar.
3. Ses duygu modeli için Bark≥.30, Bark>Speech, Music<.50 olan bitişik ses
   yamalarından en fazla yaklaşık 4.3 saniyelik bölüm seçilir. Boşluklar birleştirilmez.
4. Bu bölümün YAMNet ortalama/std temsili dış etiketlerle eğitilen RF'ye gider.
5. Üst panel örneğin **“Tahmini: Olumlu · yüksek uyarılma”** gösterir. Destek
   sınırlıysa açıkça yazılır; bütün sınıf skorları görünür. Eşikler gösterim/çekimserlik
   tercihidir, kalibrasyon veya doğruluk yüzdesi değildir.
6. Görüntü davranışları ayrı kanıt olarak gösterilir; görsel etiketler bu ses duygu
   sınıflandırıcısını eğitmedi. Korku/öfke ayrımı eğitilmedi; valence/arousal'dan bu
   sınıflar kesin etiket gibi üretilmez.

Ses yoksa veya belirgin havlama bulunmazsa duygu belirlenemedi denir. Sessizliği
mutlu/nötr diye etiketlemeyiz. Çoklu köpekte ses belli bir bireye atanamaz. Bu başlık
**sesten insan algısına göre duygulanım aktarımıdır**; senkron görsel+ses etiketleriyle
uçtan uca eğitilmiş çok modlu duygu sınıflandırıcısı değildir. Tam çok modlu duygu
hedefi için yerel videolarda bağımsız valence/arousal veya uygun duygu etiketleri gerekir.

## 4. Kaydedilen dosyalar ve komutlar

- `src/affect/prepare.py`: pinned, TLS doğrulamalı indirme ve LFS hash denetimi.
- `src/affect/train.py`: özellik çıkarımı, sızıntılı/çelişkili eğitim kaydı karantinası,
  eğitim CV ve ayrılmış dış veri testi.
- `src/affect/predict.py`: havlama segmenti, bağımsız öğrenilmiş ses duygu tahmini ve genel özet.
- `src/evaluation/late_fusion.py`: iç eğitim seçilmiş karar karışımı ve gün değerlendirmesi.
- `results/affect_v15_clean/`: eğitilmiş başlıklar, scalers, test sonuçları ve veri denetimi.
- `results/late_fusion_v15/`: ağırlık seçimi, iç/dış katlar, dağıtım reçetesi.
- `docs/affect_external_results_v15.json`, `docs/late_fusion_results_v15.json`: kodla paylaşılan sonuçlar.

```sh
python -m src.affect.prepare
python -m src.affect.train --output results/affect_repeat
python -m src.evaluation.late_fusion --output results/fusion_repeat
python -m pytest -q
python -m src.demo.launch
```

Yeni çıktı klasörü kullanılmalı. Hazır dış veri özellikleri ve model dosyaları paylaşım
paketine eklenebilir; ham 1.400 ses eklenmez. Dış veri modeli CC-BY-NC-SA kaynağıyla
birlikte akademik kullanım için belgelenir. Yeni özellik çıkarımı için ham dış veriyi resmi kaynaktan indirmek gerekir.
Hazır özelliklerle tekrar eğitim: `python -m src.affect.train --feature-cache results/affect_v15_clean --output results/affect_repeat`.
Paket kaynak CSV/provenance ve donmuş özellikleri içerir. Yalnız uygulamayı çalıştırmak için indirme gerekmez.

## 5. Birincil kaynaklar

- Abzaliev et al. (2024), https://aclanthology.org/2024.lrec-main.1432/
- Li et al. (2025), https://www.isca-archive.org/interspeech_2025/li25o_interspeech.pdf
- Dog2vec resmi kod/ağırlık yönlendirmesi: https://github.com/fispresent/dog2vec
- Garcia-Loya et al. (2025), https://doi.org/10.1016/j.patrec.2025.06.018
- Dang et al. (2025), https://doi.org/10.1145/3746027.3758286
- Yazarın tam metni: https://kenzhu2000.github.io/papers/acmmm25-dogemo.pdf
- Pinned veri: https://github.com/tmdang1101/EmotionalCanines/tree/6b73c5fce55fc132e97db6a76899f8594a955f59
- CREMD (2026 ön baskı), https://arxiv.org/abs/2602.15349
- Ağırlıklı oy: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
