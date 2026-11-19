# Görev 6 — Anatomik denetim ve muhafazakâr çizim profili

## Kanıt sınırı

Denetlenen resmî DeepLabCut 3.0.2 yapılandırması:
`configs/pose/superanimal_quadruped.yaml`. Dosya SHA-256'sı
`dac0bce73e5940a9594ca620af2c0359ca86ba1eaa258392537ec94940aa1151`
metadata kaydıyla aynıdır.
[Resmî kaynak](https://github.com/DeepLabCut/DeepLabCut/blob/fa5a8c36100c85300b7d4a534deb38d7bd8b2fa8/deeplabcut/modelzoo/project_configs/superanimal_quadruped.yaml).

Bu dosya **isimleri, sıralamayı ve boş `skeleton: []` listesini** doğrular.
`thai`, `knee`, `neck_base/end`, `back_base/middle/end` için veteriner anatomisi
karşılığı, annotasyon yerleşim tarifi veya doğrulanmış kemik bağlantısı vermez.
Depodaki diğer resmî model/metadata dosyaları da bu eksik tanımı tamamlamaz.
`thai` etiketini tahminle "thigh", omuz veya kalça olarak çevirmiyoruz;
ön bacak `knee` etiketini de belirli bir ekleme yeniden adlandırmıyoruz.

Önceki Görev 5 rehberinin 18 çizgisi **anatomik olarak doğrulanmış skeleton**
olarak sunulmamalıdır. Geçerli nokta adı ve yüksek confidence, doğru anatomik
konum/bağlantı kanıtı değildir. Gerçek videonun görüntüleri bu görevde mevcut
değildir; kullanıcının gözlemlerinin hangi kare/noktada oluştuğu bağımsız
incelenemedi. Yeni profil de gerçek video üzerinde çalıştırılmadı.

## A — Bağlantı denetim tablosu

İndeksler **0 tabanlıdır**, resmî isim listesinden çözülmüştür. Aşağıdaki gerekçeler
eski projenin çizim gerekçeleridir; resmî model bağlantısı iddiası değildir.
"Rehber olarak kalır", yalnız yüzey işaretlerini birleştiren çizgi anlamındadır;
kemik/eklem veya anatomik doğruluk onayı değildir.

| Başlangıç (indeks) | Bitiş (indeks) | Eski çizim gerekçesi | Köpeğe uygunluk sınırı | Potansiyel hata/belirsizlik | Conservative kararı |
|---|---|---|---|---|---|
| nose (0) | left_eye (10) | Burun–sol göz yön rehberi | İsimler yüz işaretlerini tanımlar; kemik değil | Perspektif, sol/sağ karışması, hatalı tahmin | Rehber olarak kalır |
| nose (0) | right_eye (5) | Burun–sağ göz yön rehberi | Aynı sınırlı yüzey gösterimi | Uzak gözün örtülmesi, yüzey dışı izdüşüm | Rehber olarak kalır |
| left_eye (10) | left_earbase (11) | Aynı taraf göz–kulak konumu | Doku/kemik komşuluğu kanıtı değil | Bakış açısı ve yanlış yerleşim uzun çizgi oluşturabilir | Rehber olarak kalır |
| right_eye (5) | right_earbase (6) | Aynı taraf göz–kulak konumu | Aynı sınırlı yüzey gösterimi | Irk/perspektif ve sağ-sol karışması | Rehber olarak kalır |
| left_earbase (11) | left_earend (12) | Sol kulak tabanı–uç rehberi | Etiketler kulak uçlarını belirtir; düz çizgi yaklaşık | Sarkık/kıvrık kulağın şeklini temsil etmez | Rehber olarak kalır |
| right_earbase (6) | right_earend (7) | Sağ kulak tabanı–uç rehberi | Aynı kulak yüzeyi yaklaşımı | Örtülme ve kulak şekli | Rehber olarak kalır |
| neck_base (15) | neck_end (16) | Boynun iki etiketini birleştirme | Anatomik uç yerleşimi metadata'da açıklanmıyor | Baş/gövde tarafı ve doğru anatomik eksen doğrulanamaz | Kaldırılır |
| back_base (19) | back_middle (21) | Sırt başlangıç–orta varsayımı | Yön/yerleşim açıklaması yok | İsimden vertebral sıra çıkarılamaz | Kaldırılır |
| back_middle (21) | back_end (20) | Sırt orta–son varsayımı | Anatomik omurga zinciri doğrulanmadı | Noktalar gövdenin yanlış bölgesindeyse yanıltıcı | Kaldırılır |
| tail_base (22) | tail_end (23) | Kuyruk tabanı–uç rehberi | Kuyruk işaretleri; omur zinciri değil | Kıvrık kuyrukta doğru gerçek şekli izlemez | Rehber olarak kalır |
| front_left_thai (24) | front_left_knee (25) | Sol ön uzuvda varsayılan sıra | `thai`/`knee` anatomik karşılığı açıklanmıyor | Omuz, dirsek veya distal eklem varsayımı yapılamaz | Kaldırılır |
| front_left_knee (25) | front_left_paw (26) | Sol ön uzuv–pati rehberi | `knee` annotasyon yeri belirsiz | Pati adı tek başına doğru eklem zincirini kanıtlamaz | Kaldırılır |
| front_right_thai (27) | front_right_knee (28) | Sağ ön uzuvda varsayılan sıra | Ön uzuv eklem tanımı doğrulanmadı | Anatomik olarak yanlış segment algısı | Kaldırılır |
| front_right_knee (28) | front_right_paw (29) | Sağ ön uzuv–pati rehberi | Başlangıç ekleminin gerçek karşılığı belirsiz | Örtülme ve yanlış segment | Kaldırılır |
| back_left_thai (31) | back_left_knee (33) | Sol arka uzuvda varsayılan sıra | `thai` konumu ve eklem protokolü açıklanmıyor | Kalça/diz/hock eşlemesi tahmin edilemez | Kaldırılır |
| back_left_knee (33) | back_left_paw (30) | Sol arka uzuv–pati rehberi | Diz etiketi ile pati arasındaki topoloji doğrulanmadı | Aradaki eklemler ve kısalma gerçek kemiği temsil etmez | Kaldırılır |
| back_right_thai (32) | back_right_knee (34) | Sağ arka uzuvda varsayılan sıra | Arka uzuv yerleşim protokolü yok | Sağ/sol ve eklem konumu karışabilir | Kaldırılır |
| back_right_knee (34) | back_right_paw (35) | Sağ arka uzuv–pati rehberi | Eklem zinciri metadata ile doğrulanamıyor | Yüksek güvenli yanlış tahmin de çizilebilir | Kaldırılır |

## B — Uygulanan güvenli değişiklik

Yeni `conservative` profili **9 nokta ve 7 çizgi** içerir: baş/kulak rehberi ve
kuyruk tabanı–ucu. Gövde/boyun ve bütün bacak çizgileri, bu alanların anatomik
anlamını metadata'dan doğrulayamadığımız için **yalnız bu profilden çıkarıldı**.
Bu alanların noktaları da conservative çizimde gösterilmez. Tahminler silinmez:
ham 39 nokta ve indeksler aynen korunur.

Yeni bağlantı eklenmedi, noktalar taşınmadı, eklem adları yeniden yorumlanmadı.
Antler noktaları iki profilde de yoktur. Bağlantıları kopuk görünmesin diye
birleştirecek omuz/kalça/gövde köprüleri uydurulmadı.

Geriye uyumluluk:

- `--dog-skeleton` tek başına eski 26 nokta / 18 çizgi davranışını korur
  (`legacy`). Bu, doğrulanmış anatomik skeleton önerisi değildir.
- `--dog-skeleton --dog-skeleton-profile conservative` yeni azaltılmış rehberi
  kullanır; anatomik belirsizliklerin gösterilmesini azaltmak için önerilir.
- Profil seçeneği `--dog-skeleton` olmadan hata verir; sessizce yok sayılmaz.
- Programatik API'de `dog_skeleton=True` eski davranıştır;
  `dog_skeleton='conservative'` yeni profildir. `False` modelin resmî şemasıdır.
- Resmî boş skeleton, model metadata'sı, YOLO/DLC inference ve checkpoint'ler
  değişmez. Video akışında yalnız çizim profili aktarımı güncellenir.

Her çizgi için mevcut iki-uç filtresi uygulanır: sonlu koordinat ve confidence,
confidence eşik aralığı, kare içi X/Y ve seçilmiş nokta olma şartı. Düşük güvenli
uç için çizgi tutulmaz veya önceki kareden taşınmaz.

## C — Görünürlük kaybı ve kalan belirsizlikler

Kareler arasında confidence eşik altına inmesi veya noktanın kare dışına
çıkması çizgiyi bastırır. Bu davranışın güvenlik filtresi kaldırılmadı.
Eşiği körlemesine düşürmek yanlış çizgileri artırabilir. Confidence değeri yüksek
olsa da nokta yanlış yerde olabilir; bu görev bunu anatomik olarak düzeltmez.

Profil azaltmak kopuk parçaları birleştirmez ve model doğruluğunu artırdığı
iddia edilmez. Gerçek anatomik düzeltme için resmî annotasyon tarifleri veya
uzman tarafından incelenmiş örnekler gerekir. Temporal görünürlük iyileştirmesi
bu görevde tracking/smoothing eklenerek yapılmadı. Baş/kulak tarafı ve kuyruk
kıvrımı hâlâ elle incelenmelidir.

## Mac komutu

Başarılı inference ortamınızda, depo kökünde; yolları gerçek dosyalarla değiştirin:

```bash
python -m src.pose_estimation.run_video \
  --input /path/to/your_dog_video.mov \
  --output results/dog_conservative.mp4 \
  --detector-weights /path/to/yolov8n.pt \
  --pose-weights /path/to/superanimal_quadruped_rtmpose_s.pt \
  --device mps \
  --keypoint-confidence 0.5 \
  --dog-skeleton \
  --dog-skeleton-profile conservative
```

Çıktı adı yeni olmalıdır; CPU için `--device cpu`. Ses hâlâ korunmaz.
Bu görevde gerçek MP4 üretilmedi; gerçek video üzerinde yeni profil Mac'te
ayrıca değerlendirilmelidir.

## Test kapsamı

Önceki testler korunur. Yeni QC testleri eksik isim, yinelenen/ters yinelenen
bağlantı, kendine bağlantı, antler dışlama, isimden indeks çözümü, iki profilin
ayrılması, düşük güven/NaN/Inf/kare sınırı filtreleri, değişmeyen ham koordinatlar
ve CLI profil aktarımını denetler. Bunlar anatomik doğruluk kanıtı veya gerçek
model/video testi değildir.
