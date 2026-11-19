# Multi-QuadEmoNet değerlendirmesi ve kontrollü uyarlama

Kaynak: Pearline ve diğerleri (2026), *Multi-QuadEmoNet: cat and dog emotion
classification model from animal vocalization using multi-stage LSTM-GRU paradigm*,
Frontiers in Veterinary Science, DOI: https://doi.org/10.3389/fvets.2026.1799281
İnceleme ve deney: 14 Kasım 2026.

## Kaynaktan alınan fikir ve eleştirel okuma

Çalışma ses tabanlıdır; MFCC ve iki LSTM ardından GRU kullanır. Köpek veri setinde
iki sınıf (sad/angry) için %95 doğruluk bildirir. Bu, bizim beş davranış hedefimizin
ortalama macro-F1 değeriyle karşılaştırılabilir değildir. Segmentasyon yöntemi sağlamaz.

Artırılmış örneklerin kaynak kayıt/hayvan bazında ayrıldığı açıkça belgelenmiyor.
Bu nedenle sızıntı olasılığını dışlayamıyoruz; sızıntı olduğunu kanıtlamış değiliz.
Örnek sayısı ve en iyi epoch anlatımlarında tablolarla uyuşmazlıklar var.
Tablo 13 köpek hücreleri toplam 400, doğru sayısı 376: %94;
bu tablo tek başına %95 sonucunu doğrulamıyor. Veri/ağırlık erişimi yeniden üretim
için ayrıca doğrulanmalı. Veteriner etiket doğrulaması etik beyanında belirtilmiş;
bizim davranış etiketlerimiz bundan dolayı duygu etiketlerine dönüşmez.

## Projeye uygulanan kontrollü deney

Yeni modül: `src/deep_learning/paper_hybrid.py`.
Mevcut 1 Hz / ilk 60 saniye poz + YAMNet dizilerini değiştirmeden GRU, LSTM ve
LSTM→LSTM→GRU karşılaştırıldı. Her tekrar eden katman 32 birimdir.
Bu çalışma MFCC girdili özgün modelin veya yayımlanan sonuçların birebir tekrarı değildir.
Girdi türü, katman genişliği, hedefler ve veri seti farklıdır.

72 eğitim / 25 test ayrımı korundu. Ölçekleme yalnız eğitim katlarında öğrenildi;
padding eğitim istatistiklerine katılmadı. Üç eğitim katında 10/20/40 epoch
karşılaştırıldı; model ve epoch seçimi test skorları hesaplanmadan diske yazıldı.
AdamW, öğrenme oranı 0,001, weight decay 0,01, seed 42, dropout 0,35,
sınıf ağırlıklı beş görev kaybı ve görünmüyor etiketi maskesi korundu.
GRU başlangıç deneyi önceki deneyin CV ve test skorlarını yeniden üretti.

| Model | Eğitim içi CV macro-F1 | Test macro-F1 | Seçilen epoch | Parametre |
|---|---:|---:|---:|---:|
| GRU | 0,4222 | 0,4192 | 40 | 46.706 |
| LSTM | 0,4210 | 0,3989 | 40 | 49.842 |
| LSTM→LSTM→GRU | 0,3109 | 0,3167 | 40 | 64.626 |

Skorlar beş davranış hedefinin macro-F1 ortalamasıdır; doğruluk yüzdesi değildir.
Eğitim içi seçim GRU'dur. Hibritin daha fazla katman kullanması bu veride kazanç
sağlamadı. Bu sonuç, bütün hibrit modellerin başarısız olduğu anlamına gelmez.
Küçük eğitim verisi, optimizasyon ve temsil seçimi olası açıklamalardır;
bu deney bunların hangisinin neden olduğunu ayırmaz.

## Karar ve sınırlar

Hibrit mimari, deney karşılaştırması olarak eklendi; canlı uygulamanın modelleri
değiştirilmedi. Duygu sınıflandırıcısı eğitilmiş gibi sunulmadı. Yeni duygu hedefi
için bağımsız ve uygun uzman etiketlemesi gerekir.

Test kümesi önceki geliştirmelerde incelendi; bu yeni ve bağımsız doğrulama değildir.
Tek seed kullanıldı. Video hash ayrımı hayvan kimliği ayrımı değildir.
Arka planın etkisi ve aynı hayvanın farklı kayıtları kesin olarak dışlanamıyor.
Hibrit fikrini MFCC ile değerlendirmek ayrı bir temsil deneyidir; bu deneyde
hem girdiyi hem mimariyi aynı anda değiştirmeyerek mevcut GRU ile kıyas korundu.

## Yeniden çalıştırma

Depo kökünde, proje Python ortamıyla:

```bash
python -m src.deep_learning.paper_hybrid --output results/paper_hybrid_repeat
python -m pytest tests/test_paper_hybrid.py tests/test_sequence_and_upload.py -q
```

Gerekli mevcut girdiler: `results/demo_v11/sequence_snapshot.npz`,
`results/demo_v11/sequence_gru.json`, `results/demo_v4/manifest.csv`.
Yeni çıktı klasörü adı kullanılmalıdır. Önceden doldurulmuş klasöre yazılmaz.
Makine okunur kayıt: `docs/paper_hybrid_results_20261114.json`.
Yerel ağırlıklar, ölçekleyiciler, CV izleri ve test tahminleri:
`results/paper_hybrid_20261114_controlled/`.

Testler değişken uzunluk, padding etkisizliği, batch sırası, tekrar eden bütün
katmanlara gradyan aktarımı ve kaydedilmiş ağırlıklarla tahmin tekrarını denetler.
