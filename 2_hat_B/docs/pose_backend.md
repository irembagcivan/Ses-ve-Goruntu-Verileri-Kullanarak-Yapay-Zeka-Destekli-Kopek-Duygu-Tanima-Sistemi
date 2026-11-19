# Görev 4 — SuperAnimal-Quadruped RTMPose-S

## Sonuç ve sınırlar

Model yapılandırması, keypoint sırası ve resmî API sözleşmeleri **DeepLabCut 3.0.2
PyPI wheel'i açılarak** doğrulandı. Paket bağımlılıklarıyla kurulmadı/import
edilmedi. Backend artık yerel checkpoint'i resmî runner ile yüklemeyi uygular;
eski koşulsuz kapatma kaldırıldı. Ancak gerçek checkpoint erişimi ortamda 403
verdi: **gerçek model yüklenmedi ve gerçek görüntü çıkarımı yapılmadı**.

Resmî proje yapılandırmasında `skeleton: []` vardır. Nokta çizimi desteklenir;
resmî bağlantılar olmadığı için anatomik çizgiler uydurulmaz. Bu nedenle görevde
istenen tam iskelet görselleştirmesi henüz sağlanamaz. Alternatif model seçilmedi.

## Model kimliği ve kaynak kanıtları

- Dataset/model ailesi: `superanimal_quadruped`.
- Mimari: `rtmpose_s`; indirme kaydı adı: `superanimal_quadruped_rtmpose_s`.
- [DLClibrary resmî kayıt dosyası](https://github.com/DeepLabCut/DLClibrary/blob/cc8bdd7a51b1f5e0b7fb07b1a0a515308d2e357a/dlclibrary/dlcmodelzoo/modelzoo_urls_pytorch.yaml).
- Kayıttaki ağırlık: `mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped/superanimal_quadruped_rtmpose_s.pt`.
- [Model deposu](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped).
- [Resmî kayıtla eşleşen dosya adresi](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped/resolve/main/superanimal_quadruped_rtmpose_s.pt).
  Bu görevin indirmesiz HEAD kontrolü `Tunnel connection failed: 403 Forbidden`
  verdi. Model kartı da açılamadı. Bu yanıt kaldırılmış dosya veya kullanıcı
  hesabının yetkisiz olduğunu kanıtlamaz; mevcut ortam erişim engelidir.
- [DeepLabCut 3.0.2 PyPI](https://pypi.org/project/deeplabcut/3.0.2/).
  İncelenen wheel SHA-256:
  `dc72f44002e881bd57a7109f14c1a6f09f7575c27f0c2179e60b843d7da21f78`.
- İncelenen resmî Git kaynağı:
  `fa5a8c36100c85300b7d4a534deb38d7bd8b2fa8`; aşağıdaki YAML ve runner oluşturma
  dosyaları indirilen wheel içeriğiyle aynıydı.
- [Model yapılandırması](https://github.com/DeepLabCut/DeepLabCut/blob/fa5a8c36100c85300b7d4a534deb38d7bd8b2fa8/deeplabcut/modelzoo/model_configs/rtmpose_s.yaml):
  SHA-256 `f2b547d694b85c6c3f3a80e3bd08a3549f5d46169937a4861902938f112abb89`.
- [Proje/keypoint yapılandırması](https://github.com/DeepLabCut/DeepLabCut/blob/fa5a8c36100c85300b7d4a534deb38d7bd8b2fa8/deeplabcut/modelzoo/project_configs/superanimal_quadruped.yaml):
  SHA-256 `dac0bce73e5940a9594ca620af2c0359ca86ba1eaa258392537ec94940aa1151`.

Bu iki küçük YAML dosyası `configs/pose/` altında değiştirilmeden saklandı.
Türetilen metadata ve kanıtlar
`configs/superanimal_quadruped_rtmpose_s.metadata.json` içindedir. Ağırlık hash'i
ve ağırlık lisansı doğrulanamadığından alanları `null` bırakıldı. Yapılandırma
hash'i, ağırlık dosyasının güvenilirliğini tek başına kanıtlamaz.

### Dosya biçimi ve lisans

Resmî checkpoint uzantısı `.pt`'dir. DLC 3.0.2 runner kaynak kodunun beklediği
PyTorch snapshot, `snapshot['model']` altında state dict içerir. Bu sözleşme
kaynak kodundan doğrulandı; gerçek dosyanın baytları/içeriği incelenemedi.
Keypoint adlarının checkpoint metadata'sında bulunması zorunlu değildir.

DeepLabCut yazılımı ve birlikte verilen yapılandırmalar LGPL-3.0-or-later;
[lisans ve atıf dosyaları](third_party/README.md) eklendi. Bu lisansın model
**ağırlıklarına** otomatik uygulandığı varsayılmadı. Model kartı erişilene kadar
checkpoint kullanım/yeniden dağıtım lisansı **doğrulanmamış** durumda.

## Doğrulanan 39 nokta ve sırası

Sıra doğrudan resmî `bodyparts` listesidir; isimlerin yazımı değiştirilmedi.
Model kafa çıkışındaki `out_channels: 39` ile tutarlıdır.

| İndeks | Keypoint | İndeks | Keypoint | İndeks | Keypoint |
|---:|---|---:|---|---:|---|
| 0 | nose | 13 | left_antler_base | 26 | front_left_paw |
| 1 | upper_jaw | 14 | left_antler_end | 27 | front_right_thai |
| 2 | lower_jaw | 15 | neck_base | 28 | front_right_knee |
| 3 | mouth_end_right | 16 | neck_end | 29 | front_right_paw |
| 4 | mouth_end_left | 17 | throat_base | 30 | back_left_paw |
| 5 | right_eye | 18 | throat_end | 31 | back_left_thai |
| 6 | right_earbase | 19 | back_base | 32 | back_right_thai |
| 7 | right_earend | 20 | back_end | 33 | back_left_knee |
| 8 | right_antler_base | 21 | back_middle | 34 | back_right_knee |
| 9 | right_antler_end | 22 | tail_base | 35 | back_right_paw |
| 10 | left_eye | 23 | tail_end | 36 | belly_bottom |
| 11 | left_earbase | 24 | front_left_thai | 37 | body_middle_right |
| 12 | left_earend | 25 | front_left_knee | 38 | body_middle_left |

Antler noktaları çok tür şemasının parçasıdır; köpeklerde anatomik olarak mevcut
oldukları iddia edilmez. Nokta sırası değiştirilmez; güven filtresi uygulanır.
Duygu özellikleri için ileride tür/şema bazlı seçim ayrıca tasarlanmalıdır.

**Skeleton:** Resmî proje YAML'inde boş liste doğrulandı. Tam bağlantı şeması
bulundu denemez. `PoseSchema.from_metadata` genel arayüzü bilinmeyen boş şemayı
reddetmeye devam eder; yalnızca resmî yapılandırmadan gelen boş skeleton, açıkça
"yalnız noktalar" olarak kabul edilir ve backend uyarı verir.

## Gerçek backend yolu

Doğrulanan [runner API'si](https://github.com/DeepLabCut/DeepLabCut/blob/fa5a8c36100c85300b7d4a534deb38d7bd8b2fa8/deeplabcut/pose_estimation_pytorch/apis/utils.py):

```python
from deeplabcut.pose_estimation_pytorch.apis.utils import get_pose_inference_runner
runner = get_pose_inference_runner(
    model_config=config, snapshot_path=local_checkpoint,
    batch_size=1, device='cpu', max_individuals=1,
)
```

Uygulama yalnızca doğrulanmış base YAML hash'ini kabul eder, resmî keypoint
metadata'sını ekler, `runner.load_weights_only=True` ayarlar. Runner modeli bir
kez kurar, snapshot yükler, `eval()` çağırır; `inference()` PyTorch inference
modunda çalışır. Güvensiz pickle fallback'i yapılmaz.

Varsayılan YAML `configs/pose/rtmpose_s.yaml`; `--pose-config` ile aynı resmî
YAML'in başka yerdeki kopyası verilebilir. Bu argüman bir proje `config.yaml`
dosyası veya önceki görevdeki `configs/default.yaml` değildir.

`VerifiedRunnerAdapter` OpenCV BGR uint8 kırpımı RGB'ye çevirir. Girdi H×W×3;
DLC ön işlem normalizasyonu ve 256×256 top-down dönüşümünü yapar. Runner'a bir
RGB crop ve `bboxes=[[0,0,width,height]]` COCO xywh context'i verilir.
`bodyparts` çıktısı `(1,39,3)` olarak doğrulanır; son bileşen güven skorudur.
DLC kendi resize/offset dönüşümünü geri alır; adaptör yalnızca dış YOLO crop
ofsetini orijinal kareye ekler. Çift ölçekleme yapılmaz.

Mevcut pose sonucu `DogPose` sınıfıdır; repoda ayrı `PoseResult` yoktur.
`pose_from_dlc_output` bu mevcut arayüze dönüştürür.

### YOLO kutuları ve ikinci detector

DLC'nin top-down runner'ı önceden hesaplanmış kutuları kabul eder. YOLOv8
korundu; ek Faster R-CNN ağırlığı veya ikinci detector yüklenmez.
DLC'nin üst düzey `build_for_superanimal_inference(..., detector_name=None)`
yardımcısı yöntemi bottom-up'a çevirdiğinden bu yol kullanılmaz. Doğrulanan
base yapılandırmanın `method: td` değeri korunur ve kutular runner context'ine
verilir. Her köpek sırayla işlenir, batch 1 ve model yeniden kullanımı korunur.

## MacBook Air M3 / Sonoma 14.6

CPU öncelikli aday ortam `requirements-pose-macos.txt` içindedir:

- PyTorch 2.6.0 / torchvision 0.21.0: eşleşen resmî sürüm çifti.
- İkisinin Python 3.11 ve 3.12 için `macosx_11_0_arm64` wheel'leri PyPI'de doğrulandı.
- DeepLabCut 3.0.2: Python >=3.10, platformdan bağımsız wheel mevcut.
- NumPy 1.26.4: Python 3.11/3.12 hedefi ve eski Albumentations ile daha temkinli seçim.

`uv pip compile --python-platform aarch64-apple-darwin --python-version 3.11`
ve aynı komutun `3.12` sürümüyle aday dosyanın bağımlılık çözümlemesi geçti.
Bunlar `/tmp` altında yapıldı; tam ML ortamı kurulmadı.

Sürüm metadata'sı ve bağımlılık çözümü, native import veya MPS çalışma testi
değildir. Apple M3 üzerinde CPU veya MPS çıkarımı bu görevde yapılmadı.
[PyTorch 2.6 MPS belgesi](https://github.com/pytorch/pytorch/blob/v2.6.0/docs/source/notes/mps.rst)
macOS 12.3+ belirtir; güncel PyTorch sürümlerinin gereksinimleri farklı olabilir.
Sonoma için körlemesine en güncel PyTorch'a yükseltmeyin.

Yerel, ayrı **arm64** ortamda aşağıdaki kurulum adımlarını uygulayın:

```bash
cd canine-multimodal-emotion
python3.12 -m venv .venv-pose
source .venv-pose/bin/activate
python -m pip install -r requirements-pose-macos.txt
python -m pip check
python -c "import torch, torchvision, cv2, deeplabcut; print(torch.__version__, torchvision.__version__, cv2.__version__, deeplabcut.__version__)"
```

Python 3.11 de kullanılabilir. Bu komutlar hedef Mac üzerinde çalıştırılmadı;
kesin çalışmış kurulum olarak sunulmaz. GUI, TensorFlow veya tensorflow-metal
kurulmaz; burada PyTorch backend kullanılır.

DLC `opencv-python-headless`, Ultralytics ise `opencv-python` dağıtımını zorunlu
tutar. Aynı `cv2` namespace'ini sağladıkları için aday listede eşleşen 4.10.0.84
sürümleri sabitlendi. Bu, iki dağıtımın birlikte kurulumunu resmen desteklenmiş
hale getirmez. Mac'te import ve görüntü I/O kontrolü yapılmalı; cv2 sorunu varsa
aynı ortamda dağıtımları rastgele kaldırıp değiştirmek yerine temiz ortamda
bağımlılık çatışması çözülmelidir. `pip check` tek başına bu native çakışmayı
algılamayabilir.

### Ağırlıkları yerelde alma

Model kartındaki lisans koşullarını inceleyin. Yalnızca resmî kayıttaki modeli,
depo dışında bir klasöre indirin; alternatif checkpoint seçmeyin:

```bash
python -c "from dlclibrary import download_huggingface_model; download_huggingface_model('superanimal_quadruped_rtmpose_s', target_dir='/absolute/path/canine-weights')"
```

Bu resmî indirme API'si kaynak kodundan doğrulandı fakat ortam engeli nedeniyle
burada çalıştırılmadı. İndirme başarılıysa model dosyasını ve Hub revision/hash
bilgisini kaydedin. `--checkpoint-sha256` bağımsız güvenilir kaynaktan aldığınız
beklenen hash ile kullanılabilir; yalnızca indirdiğiniz dosyanın hash'ini
hesaplamak dosyanın kökenini bağımsız doğrulamaz. Yerel `.pt` yolu tek başına
resmî kaynak kanıtı değildir; yalnız güvenilir ağırlıkları kullanın.

YOLOv8 detection için de güvenilir yerel ağırlığı sağlayın; CLI model indirmez.

### Tek görüntü

```bash
python -m src.pose_estimation.run_image \
  --input data/raw/sample.jpg \
  --output results/sample_pose.jpg \
  --detector-weights /absolute/path/canine-weights/yolov8n.pt \
  --pose-weights /absolute/path/canine-weights/superanimal_quadruped_rtmpose_s.pt \
  --device cpu
```

Eksik veya boş checkpoint, yanlış config, bozuk görüntü, var olan çıktı,
uyumsuz DLC sürümü ve beklenmeyen çıktı biçimleri hata verir. Sahte sonuç üretilmez.
Geçerli model yüklenirse kutular, geçerli keypoint'ler ve görünür güven skorları
çizilir; resmî skeleton boş olduğundan bağlantı çizilmez. Köpek bulunmazsa
raporda `dogs: 0` belirtilir; keypoint üretilmez.

CPU kontrolü başarılı olduktan sonra isteğe bağlı `--device mps` denenebilir.
`--device auto` yalnızca başlangıçta MPS kullanılabilirliğini kontrol eder.
MPS işlem hatasında CPU seçeneği önerilir; otomatik ve sessiz model yeniden
yükleme yapılmaz. 16 GB için batch 1; tüm video RAM'e alınmaz. Gerçek gecikme ve
RAM kullanımı ölçülmedi.

## Bu görevde doğrulananlar ve doğrulanmayanlar

Doğrulandı: resmî release içeriği ve API imzaları, YAML hash'leri, 39 nokta sırası,
boş skeleton, sentetik çıktı dönüşümü, RGB/xywh adapter sözleşmesi, tek runner
oluşturma, dosya/hata kontrolleri ve mevcut testlerin korunması.

Doğrulanmadı: gerçek ağırlık içeriği/hash/lisansı, gerçek runner yükleme, gerçek
köpek fotoğrafı çıkarımı, Mac native bağımlılıkları/MPS, tam anatomik bağlantılar.
Bu engeller giderilmeden sistem tamamen çalışır olarak sunulmamalı.


### Çalıştırılan kontroller

- Python 3.12.14: **110 test geçti**; önceki 81 test dosyası değiştirilmedi.
- Yeni 29 test: metadata/config/checkpoint denetimleri, çıktı dönüşümü,
  RGB/xywh sözleşmesi, runner yeniden kullanımı, görüntü/CLI hataları ve CPU.
- Sentetik PNG üzerinde gerçek OpenCV decode, overlay ve JPEG encode testi geçti;
  bu testte detector ve pose backend sahtedir. Gerçek hayvan/model testi değildir.
- Hafif test ortamı `pip check` başarılı; tam ML bağımlılıklarının kurulduğu anlamına gelmez.
- NumPy 1.26.4 ve OpenCV-headless 4.10.0.84 test ortamına eklendi.
  PyTorch, torchvision, DeepLabCut ve Ultralytics kurulmadı.
- Ortam kapasitesi: cgroup RAM sınırı 32 GiB, 4 CPU kotası ve başlangıçta yaklaşık
  30 GB boş disk kontrol edildi. Checkpoint erişim engeli nedeniyle büyük ML
  paketlerini kurarak kaynak tüketmeye devam edilmedi.
