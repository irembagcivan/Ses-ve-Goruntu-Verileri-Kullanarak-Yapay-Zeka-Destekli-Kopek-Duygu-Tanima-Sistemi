# Video ham pose verilerinin JSONL olarak kaydedilmesi

Video CLI'sindeki `--pose-output` isteğe bağlıdır. Kullanılmadığında mevcut
inference ve çizim davranışı, dönüş raporu ve dosyalar değişmez. Model veya
requirements değişikliği, eğitim ve tracking yoktur.

## Mac üzerinde kullanım

Mevcut başarılı inference ortamınızda, depo kökünde:

```bash
python -m src.pose_estimation.run_video \
  --input /path/to/your_dog_video.mov \
  --output results/dog_pose_video.mp4 \
  --pose-output results/dog_pose.jsonl \
  --detector-weights /path/to/yolov8n.pt \
  --pose-weights /path/to/superanimal_quadruped_rtmpose_s.pt \
  --device mps \
  --confidence 0.25 \
  --keypoint-confidence 0.5 \
  --dog-skeleton \
  --dog-skeleton-profile conservative
```

Yollar örnektir; mevcut yerel dosyalarınızla değiştirin. Girdi MP4 de olabilir.
CPU için `--device cpu` kullanın. Yeni çıktı adları seçin; MP4, JSONL ve metadata
üzerine yazılmaz. Varsayılan resmî pose YAML'i veya `--pose-config` seçeneği
önceki gibi çalışır. Görselleştirme bayraklarını kaldırmak JSONL şemasını değiştirmez.
Paket veya model indirme gerekmez; ağırlıklar Git'e eklenmez.

Bu komut şu dosyaları üretir:

- `results/dog_pose_video.mp4`: mevcut video writer'ın görselleştirme çıktısı.
- `results/dog_pose.jsonl`: çözümlenip işlenen her kare için bir JSON nesnesi.
- `results/dog_pose.metadata.json`: ayrı çalışma/şema metadata'sı.

`results/` dosyaları mevcut `.gitignore` ile Git dışında kalır.

## Kare formatı

Aşağıdaki gösterim alan şemasıdır, gerçek inference sonucu değildir:

```text
{
  frame_index: 0,
  timestamp_seconds: 0.0,
  timestamp_source: "frame_index_div_fps",
  frame_width: 1920,
  frame_height: 1080,
  dogs: [
    {
      detection_index: 0,
      bbox_xyxy: [x1, y1, x2, y2],
      detection_confidence: score,
      keypoints: [
        {index: 0, name: "nose", x: x, y: y, confidence: score,
         in_frame: true, confidence_valid: true},
        ... orijinal sıradaki toplam 39 kayıt ...
      ]
    }
  ]
}
```

Gerçek JSONL'de her kayıt tek satır geçerli JSON'dur; üstteki örnekteki sembolik
`x`, `score`, `...` ifadeleri gerçek dosyaya yazılmaz. Köpek bulunmayan karede
`dogs: []` yazılır. `frame_index` sıfırdan başlayarak artar; export sıralama
bozukluğunu reddeder. Detection/pose sayısı veya nokta şeması uyuşmazsa hata verir.

`detection_index` yalnız o kareye özgü listedeki konumdur. Bir sonraki karede
aynı indeks aynı köpek anlamına gelmez. **Kalıcı `dog_id` yoktur**; kimlik takibi
sonraki ayrı göreve bırakılmıştır.

### Koordinat ve geçersiz değer politikası

- X/Y, mevcut `DogPoseAdapter` sonucundaki **orijinal kare piksel** koordinatlarıdır;
  ağın 256×256 giriş koordinatları veya çizim için yuvarlanmış pikseller değildir.
- Sonlu ham sayılar yuvarlanmadan, filtrelenmeden ve kırpılmadan saklanır.
- Sayısal olmayan, NaN veya sonsuz değerler ayrı ayrı `null` olur. JSON yazımı
  `allow_nan=False` kullanır; dosyada standart dışı NaN/Infinity sabitleri bulunmaz.
- Sonlu kare dışı/negatif koordinatlar tanı amacıyla korunur; `in_frame: false`
  olarak işaretlenir. Sonlu [0,1] dışı confidence için `confidence_valid: false`
  olur. Bunlar analizde geçerli ölçüm gibi kullanılmamalıdır.
- Düşük confidence noktaları kayıttan çıkarılmaz; çizim eşiği yalnız görselleştirmeyi
  etkiler. Conservative profilinde çizilmeyen gövde/bacak/antler dahil 39 model
  noktasının tamamı aynı sıra ve isimlerle kaydedilir.
- Ham inference nesneleri değiştirilmez; null dönüşümü yalnız JSON temsiline aittir.

## Zaman bilgisi

Mevcut OpenCV akışı FPS'yi `CAP_PROP_FPS` ile alır. Bu uygulamada kaynak
PTS/`CAP_PROP_POS_MSEC` değerlerinin doğruluğunu kanıtlayan bir altyapı yoktur.
Yalnız monoton görünmeleri kaynak zamanı olduklarını kanıtlamaz. Bu nedenle
karışık veya varsayımsal zaman kaynağı yerine bütün karelerde açıkça
**`timestamp_seconds = frame_index / fps`** kullanılır.

Her satır `timestamp_source: frame_index_div_fps`; metadata ise ayrıca
`source_pts_verified: false` içerir. Bunlar duvar saati veya doğrulanmış gerçek
video zaman damgaları değildir. Değişken FPS, kayıp/bozuk kare ve ses eşleştirmesi
senaryolarında gerçek kaynak zamanından sapabilir. İleride doğrulanmış PTS
çıkarımı ayrı geliştirilmelidir. Bu görev video decoder veya sesi değiştirmez;
MP4 çıktısı hâlâ sessiz ve sabit FPS writer mantığındadır.

## Metadata içeriği

- `models`: sabit model ailesi kimlikleri, checkpoint SHA-256 ve hash durumu.
  YOLO varyantı dosya adından tahmin edilmez; `yolov8_detection_local` ve gerçek
  dosya hash'i kullanılır. Pose kimliği `superanimal_quadruped_rtmpose_s`.
- `device`: seçilen cihaz; işlemlerin donanım düzeyinde bağımsız audit'i değildir.
- Detection/çizim confidence eşikleri, crop margin ve görselleştirme profili.
- `video`: okuyucunun bildirdiği FPS, genişlik ve yükseklik.
- `git`: commit ve dirty durumu. Dirty=true ise commit çalışan kaynakların
  tamamını tek başına tanımlamaz.
- `keypoint_schema`: `superanimal_quadruped_39`, sürüm `1`, 39 isim ve sıra.
  Bu **projenin export şema sürümüdür**, resmî SuperAnimal model sürümü iddiası değildir.
- `coordinate_system`, `timestamp_source`, `source_pts_verified`,
  `tracking_enabled: false`, `raw_export_confidence_filter_applied: false`.
- `run_status`: `in_progress`, `complete` veya `failed`.
- `frames_written`: JSONL'ye yazılıp flush edilmiş kare sayısı.

JSONL/metadata dosyaları kullanıcı adı, Git remote/author, mutlak kullanıcı
ve model yolları veya ham exception mesajı içermez. Checksum sadece dosya
kimliğini belirtir; güvenilir kaynak/lisans kanıtı değildir.

CLI gerçek ağırlık yollarından hash hesaplar. Programatik `process_video`
çağrısı checkpoint yollarını bilmiyorsa varsayılan metadata'da hash
`null/status:not_provided` olur; değer uydurulmaz. CLI hashing başarısızlığında
`unreadable` belirtir. Metadata için kullanıcı içeriği taşımayan
`run_metadata(...)` üreticisi kullanılmalıdır.

## Akış, hata ve kaynak yönetimi

JSONL ve video aynı kare döngüsünde üretilir. Çizimden sonra, bir sonraki kare
okunmadan ham sonuç yazılıp flush edilir. Bütün kareler listede tutulmaz;
ek bellek tek karenin köpek/keypoint kaydıyla sınırlıdır.

Normal tamamlamada metadata `complete` olur. İşlem ortasında inference/çizim/
yazma hatası veya kesinti olursa capture, video writer ve export dosyaları
kapatılır. Mevcut davranış gereği kısmi MP4 kaldırılır; JSONL'nin önceki kareleri
inceleme için korunur ve metadata mümkünse `failed` olarak güncellenir.

Disk/işletim sistemi kesintisinde son satır eksik kalabilir; yalnız tam JSON
satırlarını kullanın. Flush, fsync veya güç kesintisine karşı atomik garanti
değildir. Ani süreç öldürmede metadata `in_progress` kalabilir. `complete`, okuma
döngüsünün sona erdiğini ve kayıtların yazıldığını belirtir; OpenCV'nin bozuk
kareyi EOF saymadığını veya çıktı MP4'ün ayrıca decode edildiğini kanıtlamaz.

## Test kapsamı

Sentetik tek/çok/boş köpek kareleri, null dönüşümü, ham kesirli koordinatlar,
39 noktanın sırası, sıralı JSONL/flush, metadata/hash gizliliği, çıktı çakışmaları,
başarı/hata durumunda kaynak kapatma ve eski CLI davranışı kontrol edilir.
Gerçek Mac videosu/model çıkarımı bu görevde çalıştırılmadı.
