# Son etiketlerle proje güncellemesi · v13

14 Kasım 2026. Kaynak: kullanıcının son `labels_final.csv` dosyası.

Son dosyadaki 127 video kaynak SHA256 listesiyle ve dosya adıyla eşleştirildi. Tamamen etiketsiz video kalmadı; hedef bazında boş veya uzlaşılmayan hücreler hâlâ var. Eski insan etiketleriyle doldurma yapılmadı.

## Şema değişikliği

Eski beş hedefli modelin yerine altı hedef eğitildi: hareket, ses, duruş, kuyruk konumu, kuyruk hareketi, kulak. Vücut etiketi yeni dosyada bulunmadığı için kaldırıldı. Havlamayı hareket sınıfına çevirmedik. Sallanan kuyruğun konumu ayrıca korunur.

## Veri ve kalite

| Durum | Sayı |
|---|---:|
| Toplam video | 127 |
| Eski etiketsiz kayıt | 30 |
| Bunlardan yeni güvenli eğitim kaydı | 27 |
| Güvenli eğitim videosu | 93 |
| Korunan test üyeliği | 25 |
| Yeni etiketlerle ölçülebilen test videosu | 23 |
| Güvensiz video | 11 |
| Zamanla değişken etiketli video | 36 |

11 güvensiz videonun dokuzu eğitim tarafında, ikisi eski test tarafında. Test dosyaları eğitime aktarılmadı. Boş/görünmeyen/uzlaşılmayan hedefler -1 maskesiyle kayıptan ve metrikten çıkarıldı. Güvenli bir videodaki kulak eksikliği hareket etiketini dışlamaz.

| Hedef | Eğitim etiketi | Test etiketi |
|---|---:|---:|
| Hareket | 93 | 23 |
| Ses | 93 | 23 |
| Duruş | 93 | 23 |
| Kuyruk konumu | 70 | 17 |
| Kuyruk hareketi | 90 | 23 |
| Kulak | 63 | 14 |

## Gerçek deney sonuçları

| Model | Eğitim içi CV macro-F1 | Test ortalama macro-F1 |
|---|---:|---:|
| neural_visual | 0.458 | 0.449 |
| neural_audio | 0.463 | 0.368 |
| neural_fusion | 0.490 | 0.442 |
| sequence_cnn | 0.482 | 0.410 |
| sequence_gru | 0.393 | 0.402 |
| sequence_bigru | 0.423 | 0.364 |
| sequence_lstm_lstm_gru | 0.372 | 0.406 |
| classical_visual | 0.467 | 0.514 |
| classical_audio | 0.464 | 0.379 |
| classical_fusion | 0.508 | 0.478 |
| majority_baseline | — | 0.240 |

Eğitim içi CV seçimi: **classical_fusion**. Testte görüntü referansı daha yüksek olsa da test skoruyla seçimi değiştirmedik. Uygulamada CV ile seçilmiş klasik birleşik model öne çıkarılır; sinir ağları ayrı karşılaştırılır.

Skor altı hedefin macro-F1 ortalamasıdır. Sınıf listeleri önceden belirlenmiştir; testte bulunmayan sınıflar da sıfır katkıyla hesaba katılır. Özellikle testte hırlama/sırtüstü örneği olmayabileceği için bu tanım önemlidir. Eski beş hedefli skorlarla doğrudan fark veya iyileşme hesabı yapılamaz.

## Yöntem ve sınırlar

Donmuş YOLOv8n/RTMPose-S ve YAMNet özellikleri yeniden kullanıldı; etiketler, hedef başlıkları ve bütün tahmin modelleri yeniden eğitildi. Veri kaynağı, not, köpek sayısı, konsensüs ve kalite bayrakları model girdisi değildir. Ölçekleyiciler ve sınıf ağırlıkları yalnız eğitim katlarında öğrenildi. 3 katlı GroupKFold ve 20/40/60/80 epoch; seed 42. Klasik LR/RF seçimi de yalnız eğitim CV kullanır. Kaynak hash ayrımı hayvan kimliği ayrımı değildir.

Hırlama eğitimde 1, oturma 2, sırtüstü 2 örnektir. Nadir sınıflar ve küçük destekler başarı iddialarını sınırlar. Çoklu köpekte en büyük kutu, insanın odak köpeğiyle uyuşmayabilir. Notlarda odak köpeğin sessiz ama arka planın havlamalı olduğu kayıtlar var; YAMNet karma sesi dinler ve ses kaynağını bireye atamaz.

Makaledeki LSTM–LSTM–GRU fikri aynı dizi girdileriyle yeniden denendi; MFCC özgün deneyinin birebir tekrarı değildir. İlgili kaynak ve önceki inceleme: `paper_review_20261114.md`. Yeni şemadaki sonuç önceki beş hedefli hibrit deneyinin yerine karıştırılmamalıdır.

Duygu etiketleri yok; yeni dosya duygu modelini doğrulamaz. Değişken video işaretleri zaman damgalı gerçek etiket değildir; kısa pencere doğruluğu ölçülmez. Segmentasyon çalışır fakat bu eğitim maskelerden özellik çıkarmaz; IoU/Dice ölçümü yoktur.

## Uygulama, sunum ve paylaşım

- Aktif sürüm: `configs/demo/active_release.json`.
- 127 videolu tek dosyalık galeri: `results/demo_v13_release/public/canine_demo_v13.html`.
- Yeni video: `python -m src.demo.launch`, http://127.0.0.1:8767/.
- Yeni video çıktısı: altı hedef, 8 model karşılaştırması, ses olayları, poz ve gerçek örnek maskeleri.
- Galeri: 11 model/referans, gerçek etiket, maskeleme, eski etiketsiz filtresi ve karışıklık matrisi.
- Geçmiş analizler eski sürümleriyle kalır. Yeni sonuç için video yeniden yüklenir.
- Sunum ve konuşma notları 12 slaytla yeniden üretildi. Önceki sunum `results/defense_v12_backup/` içinde saklı.
- Tam paylaşım paketi `results/canine_tam_proje_paketi.zip`; büyük galeri ve ham videolar ZIP dışında.

## Doğrulama ve tekrar

374 Python testi geçti. Gerçek HTTP video yüklemesinde 8 model × 6 hedef, segmentasyon ve HTTP Range oynatma doğrulandı. Galeri durumu 127 video / 30 eski etiketsiz / 25 korunan test dosyasıyla kontrol edildi. Kaydedilmiş 7 sinir ağı kontrol noktasının 5.334 tahmini yeniden üretildi. Bu, eğitim verisindeki yeniden üretilebilirliği denetler; yeni köpeklerde doğruluk ölçümü değildir.

Yerelde proje kökünde:

```sh
python -m src.deep_learning.final_labels --output results/demo_v13_repeat
python scripts/build_final_labels_gallery.py --results results/demo_v13_repeat
python -m pytest -q
node tests/final_labels_ui_smoke.cjs
```

Tam paylaşım paketinde eğitim tekrarı için eski donmuş özelliklerin paket içindeki yollarını belirtin:

```sh
python -m src.deep_learning.final_labels --features artifacts/feature_snapshot.npz --sequences sequence_artifacts/sequence_snapshot.npz --metadata artifacts/neural_fusion.json --output results/demo_v13_repeat
```

CSV etiketi, manifest ve kontrol noktaları paket içindedir. Ham videolar yeni özellik çıkarımı dışında tekrar eğitim için gerekli değildir. Kaynak etiket CSV metni okunarak işlenmiştir; rapordaki SHA256 bu işlenen metin dosyasına aittir, yüklenen orijinal dosyanın bayt hash’i olarak sunulmaz.
