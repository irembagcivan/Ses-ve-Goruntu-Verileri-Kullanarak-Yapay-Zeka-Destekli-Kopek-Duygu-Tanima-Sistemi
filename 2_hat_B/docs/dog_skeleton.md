# İsteğe bağlı köpek iskelet görselleştirmesi

`--dog-skeleton`, projenin tanımladığı geriye uyumlu **legacy 2D çizim
rehberini** etkinleştirir. Resmî SuperAnimal skeleton'ı değildir; resmî
`configs/pose/superanimal_quadruped.yaml` dosyasındaki `skeleton: []` değiştirilmez.
Model, checkpoint, inference, koordinatlar ve 39 keypoint'in indeks/sırası
korunur. Bu çizgiler veterinerlik açısından doğrulanmış kemik geometrisi veya
biyomekanik ölçüm modeli olarak yorumlanmamalıdır.

## Mac üzerinde video üretme

Başarıyla kullandığınız mevcut Python ortamında, depo kökünde:

```bash
python -m src.pose_estimation.run_video \
  --input /path/to/your_dog_video.mov \
  --output results/dog_skeleton.mp4 \
  --detector-weights /path/to/yolov8n.pt \
  --pose-weights /path/to/superanimal_quadruped_rtmpose_s.pt \
  --device mps \
  --margin 0.15 \
  --confidence 0.25 \
  --keypoint-confidence 0.5 \
  --dog-skeleton
```

`/path/to/...` yollarını gerçek yerel dosyalarınızla değiştirin. Girdi MP4 de
olabilir. Çıktı dosyası önceden mevcut olmamalıdır. CPU için yalnızca
`--device cpu` kullanın. Model veya paket indirme yapılmaz.

Seçenek verilmezse önceki davranış aynen korunur: resmî şema kullanılır,
39 nokta güven/görüntü sınırlarına göre çizilir ve resmî bağlantılar boş olduğu
için iskelet çizgisi üretilmez. Yeni modda yalnız seçilen 26 nokta çizilir;
seçilmeyenlerin ham tahminleri silinmez veya değiştirilmez.

Backend'in "official skeleton is empty" uyarısı hâlâ resmî model metadata'sı
hakkındadır. `--dog-skeleton` bunu değiştirmez; çizim katmanına ayrı rehber verir.
Mevcut video akışı kare kare çalışır, FPS/çözünürlük aktarımı korunur ve çıktı
**sessizdir**; ses koruma veya codec davranışı bu görevde değiştirilmedi.

## Bağlantı tanımı

Şema `src/pose_estimation/dog_skeleton.py` içindeki
`DOG_VISUALIZATION_EDGES` isim çiftleridir. İndeksler, her ismin mevcut model
şemasındaki gerçek konumu bulunarak çözülür; sabit sayı tahmini yapılmaz.
Orijinal isim yazımları, özellikle `thai`, olduğu gibi korunur.

| Bölge | İsimlere göre çizilen zincirler | Çizim gerekçesi |
|---|---|---|
| Baş | `nose → left_eye → left_earbase → left_earend` | Burun, göz ve aynı taraf kulağın görünen konumları |
| Baş | `nose → right_eye → right_earbase → right_earend` | Sağ tarafta aynı yüz/kulak rehberi |
| Boyun | `neck_base → neck_end` | Modelin isimlendirdiği iki boyun sınırı |
| Sırt | `back_base → back_middle → back_end` | Sırtın başlangıç/orta/son yüzey işaretleri |
| Kuyruk | `tail_base → tail_end` | Kuyruğun iki uç işaretini birleştiren tek doğru |
| Sol ön bacak | `front_left_thai → front_left_knee → front_left_paw` | Aynı uzvun üst/orta/pati işaret zinciri |
| Sağ ön bacak | `front_right_thai → front_right_knee → front_right_paw` | Aynı uzvun üst/orta/pati işaret zinciri |
| Sol arka bacak | `back_left_thai → back_left_knee → back_left_paw` | Aynı uzvun üst/orta/pati işaret zinciri |
| Sağ arka bacak | `back_right_thai → back_right_knee → back_right_paw` | Aynı uzvun üst/orta/pati işaret zinciri |

Legacy profilinde **26 seçilmiş nokta ve 18 bağlantı** vardır. Görev 6
denetimi bu zincirleri doğrulanmış anatomik yapı olarak değerlendirmez. `thai` ve `knee` model
etiketleridir; özellikle ön bacakta bunları kesin omuz/dirsek/carpus karşılıkları
olarak yeniden adlandırmıyoruz. Bağlantılar görünür uzuv zincirinin rehberidir.
Kuyruk çizgisi de gerçek kuyruk kıvrımını veya ara omurları tahmin etmez.

Dört `antler` noktası tamamen dışarıda bırakılır. Ağız/çene, throat, belly ve
body-middle yan işaretleri bu muhafazakâr rehbere dahil edilmez. Boyun, sırt,
kuyruk ve bacaklar arasında konumu açıkça doğrulanmamış omuz/kalça bağlantıları
tahmin edilmez; bu nedenle çizim bazı ayrık parçalardan oluşur. Eksik köprüler
sahte bağlantılarla tamamlanmaz.

## Geçerlilik ve güven filtresi

Bir çizgi ancak iki ucu da:

- Seçilmiş köpek noktalarından biriyse,
- X/Y ve confidence sonluysa,
- `keypoint-confidence ≤ confidence ≤ 1` koşulunu sağlıyorsa,
- `0 ≤ x < width`, `0 ≤ y < height` sınırları içindeyse çizilir.

Bir nokta geçersizse ona bağlı bütün çizgiler bastırılır. Koordinat düzeltme,
interpolasyon, temporal smoothing veya düşük güvenli uç yerine başka nokta
koyma yapılmaz. Yalnız raster çizimi için piksel yuvarlaması vardır; sınırdaki
geçerli kesirli noktanın raster karşılığı görüntü içinde tutulur. `DogPose`
içindeki ham float koordinatlar değiştirilmez.

Güven filtresi modelin yüksek güvenle yaptığı anatomik hataları tespit edemez.
Örtülme, perspektif ve sol/sağ karışmaları yerel videoda görsel olarak
incelenmelidir. Bu görevde yeni uzunluk/eklem açısı kuralları veya tracking
uygulanmadı.

## Doğrulama kapsamı

Otomatik testler: resmî skeleton'ın boş kalması, antler dışlama, isimle indeks
çözümü, anatomik yerel zincirler, değişmeyen varsayılan davranış, düşük güvenli/
NaN/Inf/kare dışı uçların bastırılması, raster sınırları, ham koordinatların
korunması ve video CLI → çizim seçeneği aktarımı.

Testlerde sentetik pose ve sahte video nesneleri kullanılır; gerçek model veya
video indirilmez. Bildirilen 834 karelik yerel başarılı video, önceki modun
sonucudur; yeni bağlantı modu bu görevde gerçek videoda çalıştırılmadı. Hedef
Mac'te yukarıdaki komutla ayrıca incelenmelidir.

## Görev 6 denetimi ve yeni profil

Yukarıdaki 18 bağlantılı şema geriye uyumlu `legacy` çizim rehberidir; geçerli
model isimleri kullanması anatomik olarak doğrulanmış eklem zinciri olduğunu
göstermez. Özellikle `thai`/`knee` karşılıkları ve gövde uçlarının yerleşimi
resmî metadata'da açıklanmaz; eski uzuv yorumları doğrulanmış tanım sayılmamalıdır.
Yeni, azaltılmış `--dog-skeleton-profile conservative` seçeneği yalnız baş/kulak
ve kuyruk rehberini (9 nokta, 7 çizgi) gösterir. `--dog-skeleton` bayrağıyla
birlikte kullanılır; varsayılan eski çizim korunur.

[18 bağlantının isim/indeks denetimi, kararlar ve Mac komutu](dog_skeleton_audit.md).
