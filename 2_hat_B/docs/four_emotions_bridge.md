# Davranış ve ses olaylarından dört duygu tahmini · v4 (insan etiketleri)

> V16 (15 Kasım 2026): Bu katman artık kural türevli etiketlerle değil, iki bağımsız etiketleyicinin
> insan duygu etiketleriyle eğitilir. Önceki sürümün (v3, kural türevli etiketler) açıklaması tarihsel olarak
> git geçmişinde durur.

Video hattı Happy/Mutlu, Sad/Üzgün, Angry/Öfkeli, Relaxed/Rahat için dört puan üretir. Görüntüde kuyruk
konumu/hareketi, kulak, duruş, hareket ve vücut davranış başlıkları; seste YAMNet Dog/Bark/Growling/Whimper/Howl/Speech/Music
olayları kullanılır (her sınıfın ortalaması, en yüksek üç pencere ortalaması, 0,3 üstü pencere oranı ve ses bulunma bayrağı).

## Etiket kaynağı

`results/v14_input_review/emotion_human_labels.csv`: 127 videodan 125'i etiketli (Relaxed 50, Happy 32, Sad 26, Angry 17);
2 belirsiz video eğitimde kullanılmaz. Etiketler iki bağımsız etiketleyicinin videoyu izleyip ayrı ayrı verdiği duygu
kararlarından ve uzlaşmadan gelir. Uyum: %67,7, Cohen kappa 0,553 (belirsizler hariç 0,576). Bu orta düzeyde bir uyumdur;
etiketler insanın algıladığı duygudur, hayvanın iç durumunun ölçümü değildir. `vucut` sütunu eski davranış etiket kopyasından
gelir ve duygu etiketi değildir.

## Eğitim

Mevcut eğitim bölmesinde en az üç geçerli davranış etiketi olan ve güvensiz olmayan 91 video kullanılır
(Happy 20, Sad 19, Angry 13, Relaxed 39). Görüntü, ses ve birleşik olmak üzere üç LogisticRegression modeli
(C ∈ {0,1; 0,3; 1}, balanced, max_iter=2000); seçim yalnız eğitimde üç katlı StratifiedGroupKFold (grup dosya SHA256).
Bu iç CV değerleri (görüntü 0,69; ses 0,46; birleşik 0,72) test sonucu değildir ve iyimserdir: gruplar kayıt günü değil dosya
özetidir ve girdide insan davranış etiketleri bulunur.

## Gün gruplu değerlendirme (asıl sonuç)

`python -m src.evaluation.human_emotion_day_eval` dondurulmuş özelliklerle (görüntü+poz 471, YAMNet 1053) üç kayıt grubunu
sırayla dışarıda bırakır (A: 18 Haziran 65 video, B: 22 Haziran 43 video, C: diğer/tarihsiz 17 video). Tüm C değerleri raporlanır:

| Özellik | macro-F1 aralığı (C=0,01–1) | %95 GA (en geniş) |
|---|---:|---|
| Görüntü + poz | 0,411 – 0,444 | 0,32 – 0,52 |
| Ses (YAMNet) | 0,286 – 0,340 | 0,21 – 0,41 |
| Birleşik | 0,400 – 0,425 | 0,31 – 0,51 |

Tesadüf seviyesi yaklaşık 0,25. Sonuçlar tesadüfün üstünde ama zayıftır ve güven aralıkları geniştir. Ayrıntı:
`results/human_emotion_eval/summary.md`, `metrics.json`. Bu değerlendirme uygulamadaki köprü modelin birebir
değerlendirmesi değildir; aynı etiketler ve aynı gün bölmesiyle dondurulmuş özelliklerde yapılmış bağımsız bir ölçümdür.

## Sınırlılıklar

- Yalnız 3 kayıt grubu ve 125 video var; bağımsız yeni test yok.
- Etiketler kayıt günüyle kısmen karışıktır (ör. ilk günde Angry yok), model günü de öğrenmiş olabilir.
- Puanlar kalibre değildir; kararsızlık eşikleri doğrulanmış kalibrasyon değildir.
- Eğitimde insan davranış etiketleri, çıkarımda tahmin edilen yumuşak davranışlar kullanılır; bu fark hataya yol açabilir.

## Çıkarım ve belirsizlik

Köpek tespiti örnek karelerin yarısından azındaysa görüntü girişten çıkarılır. Belirgin köpek vokalizasyonu varsa yalnız ses modeli kullanılır.
Konuşma/müzik baskın kayıt veya sessizlik tek başına duygu üretmez. Çoklu köpekte ses bireye atanamaz.
En yüksek puan 0,5 altında veya ilk iki fark 0,15 altında ise **Kararsız · öne çıkan** gösterilir.
Hırlama ile Üzgün/Rahat arasında çatışma varsa ayrıca kararsız raporlanır.

Model `configs/demo/four_emotions.json`, eğitim/çıkarım `src/emotion/behavior_bridge.py`; çıktı `behavior_emotion` (durum `human_label_estimate`).

```sh
python -m src.emotion.behavior_bridge --report results/demo_v13_release/report.json --labels results/v14_input_review/emotion_human_labels.csv
python -m src.evaluation.human_emotion_day_eval
```
