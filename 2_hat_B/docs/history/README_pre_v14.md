# Tarihsel geliştirme kaydı — güncel durum değildir

Aşağıdaki metin v14 dokümantasyon temizliği öncesi README içeriğidir. Sayılar ve tamamlanma ifadeleri farklı tarihlerden kalmıştır; güncel yöntem veya sonuç olarak alıntılanmamalıdır.

> **Güncel yerel sürüm: labels_final-v13 (14 Kasım 2026).** Son etiketlerle altı hedef yeniden eğitildi; eski beş hedefli açıklamalar tarihsel deneyleri anlatır. Güncel kullanım için [paylaşım rehberini](PAYLASIM_KURULUM.md) ve [son etiket güncellemesini](docs/labels_final_update.md) okuyun.

# Multimodal Canine Emotion Recognition

## Güncel çalışan demo — 8 Kasım 2026

`src.demo` mevcut insan davranış etiketlerinden kuyruk, kulak, duruş, aktivite
ve vücut modellerini eğitir. Görüntü (YOLOv8 + RTMPose), ses (MFCC/spektral
özetler) ve birleşik özellikler için ayrı modeller, ayrılmış test sonuçları,
sesli poz önizlemeleri ve çevrimdışı açılabilen demo ekranı vardır.
Duygu gösterimi davranışlardan örnek kurallarla türetilir; gerçek duygu
etiketleriyle eğitilmiş model değildir.

Komutlar ve sınırlar: [Hızlı demo](docs/quick_demo.md).
V4 galerisinde kaynak klasördeki 127 videonun tamamı bulunur; 72 eğitim, 25
test ve 30 eksik/etiketsiz kayıt ayrılır. YOLOv8s/M fallback ve optik akışla
süzülmüş poz önizlemeleri, sınırlı güvenli bağlantılar ve yerel klasörden
tam uzunlukta orijinal oynatma eklenmiştir. Başarı tablosu eski ayrılmış
25 test videosuna aittir; bu galeri genişlemesi yeni başarı iddiası değildir.
Aşağıdaki eski görev kayıtları kendi tamamlandıkları zamanki durumu anlatır.

YOLOv8, köpeklere özel pose estimation ve ses analiziyle köpek videolarından
**Happy, Sad, Angry, Relaxed** sınıflarını tahmin etmeyi amaçlayan akademik proje.

## Durum ve kapsam

Yapılandırma okuyucusu, arayüzler, YOLOv8 detection, koordinat/crop adaptörleri,
görselleştirme, video ve tek görüntü CLI altyapısı vardır. Seçilen pose modeli
**SuperAnimal-Quadruped RTMPose-S / DeepLabCut PyTorch 3.0.2**'dir. Resmî
39 nokta sırası doğrulandı; resmî skeleton boş. Gerçek ağırlık erişimi engellendiği
için model yükleme ve gerçek köpek görüntüsü çıkarımı henüz doğrulanmadı.
Ses analizi, duygu sınıflandırması ve eğitim uygulanmadı; insan pose modeli
köpek modeli olarak kullanılmaz.

## Planlanan mimari

```text
Video → YOLOv8 Detection/Segmentation → Dog Pose Estimation → Visual Features
Video → Audio Extraction → Audio Features
Visual Features + Audio Features → Multimodal Fusion → Emotion Classification
```

Detection köpek bölgelerini, segmentation köpek maskelerini sağlayacak;
arka planı ezberleme riskini azaltmak için kırpılmış/maskelenmiş görseller ve
pose özellikleri değerlendirilecek. Bu yaklaşımın etkisi henüz ölçülmedi.
YOLOv8 detection ve segmentation ayrı görevlerdir; Ultralytics başka mimariler de
sunduğundan ileride açıkça YOLOv8 ağırlıkları seçilmelidir.
Köpek pose backend'i seçildi; gerçek ağırlık içeriği/lisansı ve anatomik bağlantılar hâlâ açık konulardır.
Fusion ve görsel/ses modellerinin mimarileri de henüz seçilmedi.

## Klasör yapısı

```text
configs/default.yaml             Varsayılan, veriden bağımsız ayarlar
src/config.py                    YAML okuyucu ve temel doğrulama
src/preprocessing/interfaces.py Video, ses, detection, segmentation sözleşmeleri
src/pose_estimation/interfaces.py Köpeğe özel pose sözleşmesi
src/models/interfaces.py         Görsel/ses özellikleri ve fusion sözleşmeleri
src/evaluation/interfaces.py     Değerlendirme sözleşmesi
 tests/test_project_structure.py Yapı, import ve Git ignore kontrolleri
 tests/test_config.py            Yapılandırma kontrolleri
notebooks/                      Keşif çalışmaları
experiments/                    Yerel deney çıktıları (Git dışı)
results/                        Yerel sonuçlar (Git dışı)
requirements.txt                Tam geliştirme bağımlılık aralıkları
pytest.ini                      Test keşif ayarları
```

`src` şimdilik depo kökünden import edilen pakettir; dağıtılabilir paket kurulumu
henüz tanımlanmamıştır. Komutları depo kökünde çalıştırın.

## Yerel Mac kurulumu

Python **3.11 veya 3.12** kullanın. Apple Silicon M3 / macOS Sonoma üzerinde
Rosetta yerine yerel **arm64** Python önerilir (`python3.12 -c
"import platform; print(platform.machine())"` ile kontrol edilebilir).

Yalnızca iskeleti geliştirmek ve test etmek için hafif kurulum:

```bash
cd canine-multimodal-emotion
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'pytest>=8.3,<9.0' 'PyYAML>=6.0.2,<7.0'
python -m pytest
```

Python 3.11 için ilk komutta `python3.11` kullanın. ML geliştirmeye geçildiğinde,
kullanıcının yerel makinesinde tam bağımlılıklar ayrıca kurulabilir:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

Tam kurulum bu görevde çalıştırılmadı. Sürüm aralıkları Python 3.11/3.12 hedefler;
kilit dosyası değildir ve tüm bağımlılık kombinasyonları doğrulanmış değildir.
Mac üzerinde CUDA paketi veya özel CUDA indeksine gerek yoktur. Ultralytics'in
lisans koşullarını ve seçilecek ağırlıkların lisansını kullanım öncesi inceleyin.

### MPS ve 16 GB birleşik bellek

`runtime.device: auto`, gelecekteki runtime için MPS varsa MPS, yoksa CPU
seçme niyetini belirtir; şu an cihaz seçimi uygulaması yoktur. PyTorch kurulduktan
sonra MPS durumunu şu şekilde kontrol edebilirsiniz:

```bash
python -c "import torch; print('built:', torch.backends.mps.is_built()); print('available:', torch.backends.mps.is_available())"
```

Varsayılan batch size 1, worker sayısı 0 ve görüntü boyutu 640'tır. Gelecekte
kareleri akış halinde okuyun; tüm videoyu RAM'e yüklemeyin. MPS işlem desteğini
her model için ayrıca doğrulayın; desteklenmeyen işlemler CPU gerektirebilir.
M3/MPS davranışı bu bulut ortamında test edilmedi.

## Veri seti

Kaynak: https://data.mendeley.com/datasets/5g5hg9hrkv/2

Veri seti ortamda bulunmuyor. Dosya ağacı, etiket biçimi, sınıf eşlemesi,
köpek kimlikleri ve ses kanallarının varlığı henüz incelenmedi; bunlar varsayılmaz.
Lisansı ve erişim koşullarını kontrol ederek veriyi kendiniz indirin ve tercihen
depo dışında, örneğin `~/Datasets/canine-emotion/` altında saklayın.

`configs/default.yaml` kopyasını depo dışında oluşturup `data.root` alanını gerçek
yola ayarlayın. `data.manifest`, dosyalar ve etiketler incelenip açık bir manifest
şeması tasarlanana kadar `null` kalmalıdır. Loader bu yolları okur fakat veri
keşfetmez, etiket üretmez veya dosyaların varlığını doğrulamaz.

**Veri setini GitHub'a yüklemeyin.** `.gitignore`, `data/`, `datasets/`, yaygın
medya dosyalarını, ağırlıkları ve deney çıktılarını dışlar; bu Git için genel
bir dosya boyutu sınırı değildir. Commit öncesi `git status --short` kontrolü yapın.
Yeni medya biçimleri ve depo dışındaki yerel ayarlar ayrıca korunmalıdır.

## Yapılandırma kullanımı

```python
from src.config import load_config

config = load_config()  # Çalışma dizininden bağımsız varsayılan yol
# config = load_config('/absolute/path/local.yaml')  # Tam YAML dosyası
```

Okuyucu güvenli YAML yüklemesini kullanır; kısmi ayar birleştirme yapmaz.
Zorunlu bölümler, sınıfların sırası, cihaz seçenekleri, batch/worker değerleri,
YOLOv8 görev adları ve pose türü doğrulanır. Bu temel doğrulama, model ağırlığının
köpeklere uygun olduğunu kanıtlamaz. Dosya hataları `OSError`, yapılandırma
hataları `ConfigError` olarak bildirilir. İçe aktarma ağırlık indirmez ve ML
kütüphanelerini yüklemez.

## Testler ve geliştirme aşamaları

```bash
python -m pytest -q
```

Testler klasörleri, hafif importları, Git ignore kurallarını, varsayılanları,
özel yapılandırma yüklemeyi ve hatalı yapılandırmaların reddini kontrol eder.
Gerçek video, ses, model doğruluğu, MPS veya eğitim başarısını ölçmez.

Sonraki geliştirme aşamaları:

1. Veri seti dosya/etiket yapısını ve lisansını inceleyip manifest tanımlamak.
2. Video ve ses ön işlemeyi, YOLOv8 detection/segmentation adaptörlerini uygulamak.
3. Köpeklere özel pose backend'i, ağırlıkları ve keypoint şemasını doğrulamak.
4. Görsel/ses özellikleri ve multimodal fusion modellerini uygulamak.
5. Aynı video veya köpeğin bölümler arasında sızmasını engelleyen veri ayrımıyla
   unimodal ve multimodal deneyler yapmak; macro F1, sınıf bazlı metrikler ve
   confusion matrix raporlamak. Arka plan etkisini ablation deneyleriyle incelemek.

Bu görevde sonraki aşamalar başlatılmaz.

## Görev 3 — Köpek pose video altyapısı

Bu bölüm önceki iskelet durumunu günceller: yerel YOLOv8 detection, kontrollü crop
adaptörü, skeleton çizimi ve akış halinde video işleme uygulandı. Görev 4 ile
resmî yapılandırma ve keypoint sırası doğrulanıp yerel checkpoint yükleme yolu
eklendi. Gerçek ağırlık hâlâ bulunmadığı için çalışmış pose videosu iddiası yoktur;
resmî skeleton boş olduğundan anatomik çizgiler üretilmez.
Ayrıntılı API kanıtları ve açma koşulları: [pose_backend.md](docs/pose_backend.md).

### Modüller

- `src/preprocessing/detection.py`: yerel YOLOv8 checkpoint'i, metadata'dan `dog`
  sınıfı seçimi, çoklu kutular ve güven skorları. Eksik ağırlık otomatik indirilmez.
- `src/pose_estimation/geometry.py`: xyxy → xywh, sınırlandırılmış kenar payı,
  kırpım oluşturma ve koordinatların orijinal kareye dönüşü.
- `src/pose_estimation/backend.py`: DLC 3.0.2 yerel checkpoint loader'ı ve
  runner adaptörü; gerçek ağırlıkla henüz test edilmedi. İnsan pose fallback'i yoktur.
- `src/pose_estimation/metadata.py`: resmî YAML'den doğrulanan isim/skeleton
  sözleşmesi; hiçbir köpek bağlantısı uydurulmadı.
- `src/pose_estimation/adapter.py`: köpekleri sırayla işler; batch 1. Kalıcı köpek
  kimliği veya kareler arasında tracking uygulanmaz.
- `src/pose_estimation/visualization.py`: kutu/skor, keypoint ve bağlantı çizimi;
  düşük güvenli, NaN ve kare dışı noktaları filtreler.
- `src/preprocessing/video.py`: MOV/MP4 girdisini kare kare okur, MP4 üretir.
- `src/pose_estimation/device.py`: auto → MPS mevcutsa MPS, yoksa CPU.
- `src/pose_estimation/run_video.py`: CLI ve hataların raporlanması.

### Mac üzerinde komutlar

Mevcut hafif ortamla yalnızca yardım ve ağırlıksız testler çalıştırılabilir:

```bash
source .venv/bin/activate
python -m src.pose_estimation.run_video --help
python -m pytest -q
```

Gerçek kullanım için ayrıca yerel, güvenilir YOLOv8 detection ağırlığı,
SuperAnimal-Quadruped RTMPose-S ağırlığı ve eşleşen resmî DLC PyTorch config gerekir.
PyTorch/Ultralytics/OpenCV ve DeepLabCut PyTorch backend kurulumu hedef Mac'te
doğrulanmalıdır. Görev 4 için ayrı `requirements-pose-macos.txt` eklendi;
DeepLabCut ve PyTorch bulut ortamına kurulmadı. Sonoma/MPS için yukarıdaki sürüm uyarısı
geçerlidir; önce CPU doğrulaması önerilir.

Aşağıdaki komut **gerçek checkpoint gerektirir; uçtan uca çalıştırılmadı**. Yollar gerçek yerel dosyalarla değiştirilmelidir:

```bash
python -m src.pose_estimation.run_video \
  --input data/raw/sample.mov \
  --output results/sample_pose.mp4 \
  --detector-weights /absolute/path/yolov8n.pt \
  --pose-weights /absolute/path/superanimal_quadruped_rtmpose_s.pt \
  --pose-config configs/pose/rtmpose_s.yaml \
  --device cpu \
  --margin 0.15 \
  --confidence 0.25 \
  --keypoint-confidence 0.5
```

MPS doğrulamasında `--device mps` kullanılabilir. Desteklenmeyen MPS işlemlerinde
sessizce cihaz değiştirilmez; hata CPU komutuyla yeniden çalıştırmayı önerir.
`--device auto` yalnızca MPS kullanılabilirliğine göre başlangıç seçimi yapar.

Video akışı tüm videoyu veya tüm tahminleri RAM'de tutmaz. Girdi FPS ve piksel
boyutları writer'a aktarılır; bütün kareler işlenir, `frame_stride` kullanılmaz.
**Çıktıda ses korunmaz.** Değişken FPS, HDR, MOV yön/rotation metadatası ve renk
profili korunması garanti değildir. Codec desteği yerel OpenCV kurulumuna bağlıdır;
yazıcı/okuyucu hataları bildirilir. Decode sırasında EOF ile bozuk kare ayrımı
OpenCV'de kesin değildir. Var olan çıktı dosyaları üzerine yazılmaz; hata halinde
bu çalışmanın kısmi çıktısı silinir.

Test videoları gerçek veri değildir: video akış testleri sahte okuyucu/yazıcı,
detection testleri sahte Ultralytics çıktıları kullanır. Codec, gerçek YOLOv8,
gerçek DLC runner, video çıktısı ve MPS henüz uçtan uca test edilmedi. Bu görevde
duygu sınıflandırması, ses işleme veya model eğitimi uygulanmadı.


## Görev 4 — Tek görüntü ve gerçek model yükleme yolu

CPU öncelikli aday Mac ortamı:

```bash
python3.12 -m venv .venv-pose
source .venv-pose/bin/activate
python -m pip install -r requirements-pose-macos.txt
python -m pip check
```

Python 3.11 de hedeflenir. Bu dosya PyTorch 2.6.0, torchvision 0.21.0,
DeepLabCut 3.0.2 ve NumPy 1.26.4 sürümlerini seçer. Python 3.11/3.12 arm64
bağımlılık çözümlemesi `uv pip compile` ile geçti; **Mac kurulumu veya native
çalışma testi değildir**. Ortak `cv2` namespace'ini sağlayan iki OpenCV
dağıtımı dahil olmak üzere sınırlar ve sürüm kanıtları
[backend dokümanında](docs/pose_backend.md) açıklanmıştır.
Bu ayrı ortam, hafif test ortamınızın üzerine kurulmak zorunda değildir.

Yalnızca resmî model kaydındaki ağırlığı, lisansını inceledikten sonra depo
dışına indirin:

```bash
python -c "from dlclibrary import download_huggingface_model; download_huggingface_model('superanimal_quadruped_rtmpose_s', target_dir='/absolute/path/canine-weights')"
```

Burada bu indirme çalıştırılmadı: resmî Hugging Face adresi 403 verdi. Ağırlık
lisansı ve checkpoint hash'i doğrulanmış değildir. CLI kendiliğinden indirmez;
yalnız güvenilir yerel ağırlıkları kullanın.

```bash
python -m src.pose_estimation.run_image \
  --input data/raw/sample.jpg \
  --output results/sample_pose.jpg \
  --detector-weights /absolute/path/yolov8n.pt \
  --pose-weights /absolute/path/superanimal_quadruped_rtmpose_s.pt \
  --device cpu
```

İsteğe bağlı `--pose-config configs/pose/rtmpose_s.yaml` verilebilir; varsayılan
aynı dosyadır. `--checkpoint-sha256` bağımsız güvenilir kaynaktan doğruladığınız
beklenen hash'i alır. Eksik ağırlık veya uyumsuz config hata verir; çıktı uydurulmaz.
CPU kontrolünden sonra `--device mps` denenebilir; hata halinde CPU önerilir.

Yeni modüller: `src/preprocessing/image.py`, `src/pose_estimation/run_image.py`.
`pose_from_dlc_output`, gerçek DLC çıktı sözleşmesini mevcut `DogPose` sonucuna
çevirir. Keypoint adları/sırası, `configs/pose/superanimal_quadruped.yaml`
dosyasındaki **39 noktanın aynısıdır**; tam liste backend dokümanındadır.
Model çok tür dört ayaklı şeması kullandığından antler noktaları da içerir;
bunların köpeklerde anatomik olarak var olduğu varsayılmaz.

**Resmî `skeleton: []`:** kutular, keypoint'ler ve güven skorları çizilebilir;
resmî bağlantılar olmadığı için anatomik iskelet çizgileri eklenmez. Bu eksik
kısım ayrıca doğrulanmadan tam skeleton desteği iddia edilmez.

Doğrulama kapsamı: önceki testler korunur; dosya/config doğrulama, şema, sentetik
DLC çıktı dönüşümü, RGB/xywh adaptörü, tek model oluşturma, CPU seçimi ve görüntü
hata yolları test edilir. Sentetik birim testleri gerçek model inference'ı değildir.
Gerçek checkpoint yükleme, gerçek köpek fotoğrafı çıkarımı, ağırlık lisansı ve
Apple M3/MPS henüz doğrulanmamıştır. Eğitim veya sonraki görev başlatılmadı.

Görev 4 doğrulama sonucu: **110 test geçti** (önceki 81 + yeni 29), Python 3.12.14.
Sentetik görüntü üzerinde gerçek OpenCV okuma/çizim/JPEG yazma kontrolü geçti;
detector ve pose çıktısı bu testte sahtedir. Hafif test ortamında NumPy ve
OpenCV-headless kuruldu; PyTorch/DeepLabCut kurulmadı. Tam model ve gerçek köpek
görüntüsü üzerinde inference yapılmadı.

## Çalışma ortamını belgelemek

Inference için kullandığınız mevcut Python ortamında, paketleri değiştirmeden:

```bash
python scripts/collect_environment.py --output results/environment_mac.json
```

İsteğe bağlı olarak her yerel model için `--checkpoint /path/to/model.pt`
eklenebilir; argüman tekrarlanabilir. JSON; sürümler, işletim sistemi/mimari,
MPS durumu, Git commit/dirty ve anonim checkpoint SHA-256 kayıtlarını içerir.
Dosya yolları, kullanıcı adları, token'lar ve ortam değişkenleri kaydedilmez.
Bilinmeyen bilgiler `null` olur; önceki Mac gözlemleri otomatik doldurulmaz.
`results/` raporları Git dışıdır ve var olan raporun üzerine yazılmaz.

Ayrıntılar ve yerel Mac doğrulama adımları:
[Deney tekrarlanabilirliği](docs/reproducibility.md).

## Köpek anatomisine göre isteğe bağlı çizim

Video komutuna `--dog-skeleton` eklendiğinde proje tarafından tanımlanan
**26 noktalı / 18 bağlantılı** 2D köpek çizim rehberi kullanılır. Antler noktaları
gösterilmez; modelin 39 ham tahmini ve sırası değişmez. Bu bağlantılar resmî
SuperAnimal skeleton'ı değildir; resmî boş skeleton korunur. Seçenek verilmezse
önceki çizim davranışı devam eder.

Yalnız iki ucu da güvenli, sonlu ve görüntü içinde olan çizgiler çizilir.
[Bağlantılar, sınırlar ve Mac video komutu](docs/dog_skeleton.md).

### Anatomik denetim ve conservative profil

`--dog-skeleton --dog-skeleton-profile conservative`, anlamı metadata'da
ayrıntılı açıklanmayan boyun/gövde/bacak zincirlerini dışlar; 9 nokta ve 7
baş/kulak/kuyruk rehber çizgisi gösterir. Bu yeni bir doğrulanmış anatomik
skeleton değildir. `--dog-skeleton` tek başına eski 26/18 çizimini korur.
Ham model çıktıları değişmez; kopuk parçalar tahminî köprülerle birleştirilmez.

[Ayrıntılı bağlantı denetimi ve kullanım](docs/dog_skeleton_audit.md).

## Video pose verisini JSONL olarak kaydetmek

Video komutuna `--pose-output results/dog_pose.jsonl` ekleyin. Aynı işlemde kare
başına ham detection/39 keypoint verisi yazılır; ayrıca
`results/dog_pose.metadata.json` oluşturulur. Koordinatlar yuvarlanmaz ve çizim
profili/eşiği ham çıktıyı filtrelemez. NaN/Inf değerler `null` olur.

Zaman kaynağı açıkça `frame_index / fps` olarak kaydedilir; gerçek PTS değildir.
Detection indeksi yalnız o kareye aittir; tracking veya kalıcı dog_id eklenmez.
Seçenek kullanılmazsa önceki video davranışı korunur.

[JSONL formatı, metadata, hata politikası ve Mac komutu](docs/video_pose_export.md).

## Offline zamansal hareket analizi

Mevcut video pose JSONL dosyasından, model çalıştırmadan:

```bash
python -m src.analysis.run_motion \
  --input results/dog_pose.jsonl \
  --output results/dog_motion.csv \
  --summary results/dog_motion_summary.json \
  --confidence 0.5
```

Tek köpekli ardışık güvenilir karelerde piksel/saniye ve kutu diyagonaliyle
normalize hareket, baş/gövde/kuyruk grupları ve göreli kuyruk vektörü hesaplanır.
Çok köpekli/eksik kareler birleştirilmez; eksik değerler sıfır sayılmaz.
Sıçramalar işaretlenip özetlerden çıkarılır. Tracking, smoothing, kamera
hareketinin tam düzeltilmesi veya duygu çıkarımı yapılmaz.

`--plot results/dog_motion.png` isteğe bağlıdır; Matplotlib yoksa CSV/JSON yine
üretilir. [Formüller, veri politikası ve sınırlamalar](docs/motion_analysis.md).

## Sıralı video batch işleme

Bir klasördeki MOV/MP4 videoları sıralı işlemek, video başına pose/motion çıktıları
oluşturmak ve doğrulanmış çıktılardan devam etmek için
[`src.batch.run_batch`](docs/batch_processing.md) kullanılabilir. Önce `--limit 3`
ile pilot çalıştırın. Yerel ağırlıklar gerekir; paket veya model indirilmez.
Komutlar, rapor biçimi, resume ve kısmi çıktı politikası için
[batch kullanım belgesine](docs/batch_processing.md) bakın.

### Audio event and behavior demo

YAMNet sound-event candidates now provide a full-recording timeline and three
experimental behavior variants, with the original models retained for comparison.
See [audio workflow](docs/audio_events.md) and
[fixed-split results](docs/audio_behavior_results.md) for commands and limitations.

Three algorithms and six feature groups are now compared with training-only
grouped CV. See the [experiment protocol](docs/behavior_experiments.md) and
[interactive experiment results](docs/experiments_20261108.html). The gallery's
new **Eğitim CV seçimi** mode uses the per-behavior training-selected pipelines.

The [human annotation workspace](docs/annotation_review.md) helps complete the
30 incomplete videos with local playback, JSON backups and a validated import.
Model suggestions remain separate from human labels; original splits stay fixed.

Demo v8 adds an experimental 4-second behavior timeline (2-second hop) using the
frozen training-selected models. See [completion and limitations](docs/demo_status.md)
and the [frozen demo contract](configs/demo/release_v8.json).

### Trainable deep learning release

[CanineFusionNet](docs/deep_learning.md) now learns pose/audio feature encoders,
fusion and five behavior heads. Three neural modality ablations use training-only
CV epoch selection, with classical references retained. The [offline defense](docs/hoca_sunumu.html)
and [speaker notes](docs/sunum_konusma_notlari.md) report both gains and limitations.

### New video upload and temporal architectures

Run `START_ANALYSIS.command` or `python -m src.demo.upload_server`, then open
http://127.0.0.1:8766 for actual file upload and local analysis. Temporal CNN, GRU
and BiGRU experiments and the [remaining-gap audit](docs/audit_and_upload.md)
separate implemented features from unresolved scientific limits.

### Instance segmentation and sharing

New uploads now include real YOLOv8s-seg dog masks, a segmented video and per-instance
PNG/JSON downloads. See [segmentation](docs/segmentation.md) and the
[friend setup guide](PAYLASIM_KURULUM.md). Use `requirements-share.txt` for the
sharing app; the Mac/Windows launchers validate dependencies and model files.
