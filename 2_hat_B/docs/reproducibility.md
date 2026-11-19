# Çalışma ortamı ve deney tekrarlanabilirliği

`scripts/collect_environment.py`, **çalıştırıldığı Python ortamını** kaydeder.
Yalnız standart kütüphane ile yazılmıştır; paket veya model indirmez, inference
başlatmaz, mevcut bağımlılıkları değiştirmez. Aynı makinedeki başka bir sanal
ortamı veya uzaktaki Mac ortamını keşfetmez.

## Kullanım

Inference için kullandığınız mevcut sanal ortamı etkinleştirin. Depo kökünde:

```bash
python scripts/collect_environment.py --output results/environment_mac.json
```

Checkpoint'leri de kayda bağlamak için tekrarlanabilir `--checkpoint` argümanı:

```bash
python scripts/collect_environment.py \
  --output results/environment_with_checkpoints.json \
  --checkpoint /path/to/yolov8n.pt \
  --checkpoint /path/to/superanimal_quadruped_rtmpose_s.pt
```

Yollar gerçek yerel yollarınızla değiştirilmelidir; bunlar örnek yollardır.
Klasör önceden mevcut olmalıdır. Var olan çıktı dosyasının üzerine yazılmaz.
Yeni bir çıktı adı seçin. Çıktı ile checkpoint yolları aynı olamaz.
`results/` altındaki raporlar mevcut `.gitignore` kuralları gereği Git dışıdır.
Model ağırlıkları kopyalanmaz veya Git'e eklenmez; yalnız baytları okunur.

Checkpoint kimlikleri komut sırasına göre `checkpoint_1`, `checkpoint_2` olur.
Dosya adı/yolu JSON'da yer almaz; hangi sırayı kullandığınızı yerel deney
notunuzda tutun. Hash hesaplaması 1 MiB bloklar halinde yapılır; dosya bütünü
RAM'e yüklenmez. Hash, dosya kimliğini karşılaştırır; dosyanın resmî kaynaktan
geldiğini veya kullanım lisansını tek başına kanıtlamaz. Hesaplama sırasında
checkpoint dosyalarını değiştirmeyin.

## Çıktı içeriği

- `schema_version`: rapor şeması sürümü.
- `python_version`: çalışan Python sürümü.
- `os`: işletim sistemi ailesi ve sayısal sürümü. macOS için ürün sürümü alınır;
  hostname içerebilen tam platform/kernel açıklaması alınmaz.
- `architecture`: bilinen makine mimarisi; örneğin `arm64`.
- `packages`: PyTorch, torchvision, DeepLabCut, Ultralytics, NumPy ve OpenCV
  dağıtımlarının kurulu metadata sürümleri. DeepLabCut/torchvision/Ultralytics/
  NumPy sırf sürüm öğrenmek için import edilmez.
- `runtime.opencv_version`: gerçekten import edilebilen `cv2.__version__`.
  Birden çok OpenCV dağıtımı kuruluysa bunlar ayrı metadata kayıtlarında görünür;
  runtime sürümü ayrıca raporlanır. Bu kayıt dağıtım çakışmasını otomatik çözmez.
- `runtime.mps_available`: `torch.backends.mps.is_available()` sonucu.
  `false`, probe'un çalıştığını fakat MPS'nin kullanılabilir olmadığını;
  `null`, probe'un yapılamadığını belirtir. Bu bir model/MPS işlem testi değildir.
- `git.commit`: betiğin bulunduğu checkout'un commit kimliği.
- `git.dirty`: izlenen değişiklik veya ignore edilmemiş yeni dosya olup olmadığı.
  Ignore edilen model/veri/çıktılar bu alana dahil değildir. `true` ise commit
  tek başına o anki kaynak kodunu tam tanımlamaz.
- `checkpoints`: anonim sıra kimliği, SHA-256 ve durum.

Bilinmeyen, kurulu olmayan, okunamayan veya güvenli biçimde temsil edilemeyen
sürümler `null` olur. Paket metadata'sı bulunması o paketin import veya inference
başarısını kanıtlamaz. MPS ve OpenCV için ayrı durum alanları kullanılır.
Checkpoint durumları `hashed`, `missing`, `not_a_file`, `unreadable` olabilir.
Eksik checkpoint rapora yazılır; betik bu nedenle rapor üretimini durdurmaz.
Başarılı çıkış kodu, bütün bileşenlerin kullanılabilir olduğu anlamına gelmez.

## Hassas bilgi sınırı

JSON'a ortam değişkenleri, API anahtarları, token'lar, Git remote URL'leri,
kullanıcı/hostname, Python executable yolu, çalışma dizini, checkpoint yolları
ve ham exception mesajları eklenmez. Yalnız açıkça seçilmiş alanlar toplanır.
Paket sürümlerindeki `+...` yerel build ekleri kullanıcı/makine bilgisi
içerebildiğinden çıkarılır; bu nedenle yerel build'ler arasında tam ayrım
sağlanmaz. Tanınmayan sürüm metinleri tahmin edilmez.

Commit kimliği ve model hash'leri açıkça istenen tekrarlanabilirlik
parmak izleridir; raporu paylaşırken bunların proje/modeli ilişkilendirebildiğini
hesaba katın. Paket importlarının kendi terminal uyarıları bu JSON'un parçası
değildir. Betik inference kodunun davranışını veya paket ayarlarını değiştirmez.

## Mac'te ayrıca doğrulanacaklar

Daha önce bildirilen Python 3.11.17, torch 2.6.0, torchvision 0.21.0,
DeepLabCut 3.0.2, Ultralytics 8.4.189, NumPy 2.4.6 ve OpenCV 5.0.0 bu betiğe
varsayılan olarak yazılmamıştır. Testlerdeki örnek sürümler gerçek Mac ölçümü
olarak raporlanmaz. Betiği çalışan Mac ortamında çalıştırıp:

1. Gerçek sürüm kayıtlarını bu gözlemlerle karşılaştırın.
2. `runtime.opencv_version` ile OpenCV dağıtım sürümlerini ayrı inceleyin.
3. MPS değerinin `true` ve durumun `checked` olduğunu kontrol edin.
4. Commit/dirty durumunu çalıştırılan kodla ilişkilendirin.
5. Model hash'lerini güvenilir yerel dosyalarınızla karşılaştırın.

Mevcut requirements dosyaları değiştirilmedi. Önceden bildirilen NumPy 2.4.6
ve OpenCV 5.0.0, aday Mac dosyasındaki NumPy 1.26.4 / OpenCV 4.10.0.84 pinleriyle
aynı değildir; genel requirements dosyasındaki OpenCV `<5.0` sınırı da bildirilen
5.0.0 ile uyuşmaz. Bu görev, çalışan ortamı bu pinlere zorlamaz veya paket kurmaz.
Betiğin raporu bir lockfile, model doğruluk testi ya da CPU/MPS karşılaştırma
raporu değildir.

## Testler

```bash
python -m pytest -q
```

Yeni testler sürüm toplama, eksik bileşenler, MPS true/false/unknown,
SHA-256, Git commit/dirty, hassas bilgi filtreleme ve JSON dosya hatalarını
kontrol eder. Bunlar gerçek model veya gerçek Mac inference testi değildir.
Mevcut test dosyaları ve inference/video modülleri bu görevde değiştirilmez.
