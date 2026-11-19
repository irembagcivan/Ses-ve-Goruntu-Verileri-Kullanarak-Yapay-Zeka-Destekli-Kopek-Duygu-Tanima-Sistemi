# Sıralı video batch işleme

Batch CLI mevcut `process_video`, JSONL exporter ve `analyze_file` işlevlerini çağırır;
inference, görselleştirme ve hareket formüllerini değiştirmez. Yeni paket gerekmez.
Gerçek modeller için mevcut Mac inference ortamını etkinleştirin. Ağırlıklar yerel
olmalıdır; otomatik model indirilmez. Henüz gerçek veri üzerinde batch doğrulaması yapılmadı.

## Üç videoluk pilot

Depo kökünde, yolları yerel dosyalarınıza göre değiştirerek:

```bash
python -m src.batch.run_batch \
  --input-dir "/path/to/Internship dataset" \
  --output-dir results/batch \
  --detector-weights /path/to/yolov8n.pt \
  --pose-weights /path/to/superanimal_quadruped_rtmpose_s.pt \
  --device cpu \
  --limit 3
```

Varsayılan resmî pose YAML kullanılır; gerekiyorsa mevcut backend'in kabul ettiği
orijinal YAML için `--pose-config /path/to/rtmpose_s.yaml` verin. `--device mps`
isteğe bağlıdır; mevcut backend'in cihaz doğrulama/hata davranışı korunur.
CPU varsayılandır. Açık CPU/MPS seçiminde sessiz runtime fallback yoktur.
`auto` mevcut cihaz seçicisini kullanır; tekrarlanabilir deneyler için açık cihaz seçin.

Yalnızca doğrudan klasör içindeki normal `.mov`/`.mp4` dosyaları, harf duyarsız
uzantıyla seçilir. Sembolik video bağlantıları ve alt klasörler taranmaz. Sıralama
`(filename.casefold(), filename)` şeklindedir. Diğer uzantılar mevcut pipeline
kabul etmediğinden seçilmez; MOV/MP4 kapsayıcısının codec'i de yerel OpenCV ile
çözülebilmelidir. `--limit 3` sıralı listenin ilk üçünü seçer; resume ile atlananlar
da bu limite dahildir. Limit verilmezse tüm uygun dosyalar seçilir.

## Çıktılar ve mahremiyet

```text
results/batch/
  batch-run-000001/
    000001.json
    report.json
  video-<tam-dosya-adının-SHA256-değeri>/
    attempt-000001/
      started.json
      skeleton.mp4
      pose.jsonl
      pose.metadata.json
      motion.csv
      motion_summary.json
      complete.json
```

Dosya adı uzantısı ve harfleri kimlik hesabına dahildir; aynı köke sahip farklı
videolar çakışmaz. Kimlik bir hayvan/tracking kimliği değildir. Raporlar mutlak
yolları, kaynak adlarını ve ham hata metnini içermez. Pose çıktıları orijinal 39
noktayı korur. Görselleştirme conservative profili kullanır. MP4 sessiz ve sabit
FPS'tir; zaman bilgisi `frame_index / fps`, süre `frames / fps` tahminidir.

Rapor her video için durum (`complete`, `skipped_verified`, `failed`), kare/süre,
çalışma süresi, pose/motion durumu, cihaz (başarılı inference sonrası), hata sınıfı
ve deneme kimliğini kaydeder. Resume süresi doğrulama süresidir. Her video sonrası
ayrı durum JSON'u yazılır; normal bitişte toplu `report.json` oluşur. Hata varsa
CLI 2, aksi halde 0 döndürür. Boş klasör sıfır videolu rapor üretir.

## Resume ve kısmi çıktılar

Aynı komutu tekrar çalıştırın. Resume varsayılan olarak açıktır. Şunlar doğrulanır:

- Kaynak videonun SHA-256 değeri, model/config SHA-256 değerleri, eşikler ve cihaz seçimi.
- Kaynak/config dosyalarının hash'leri, Git durumu ve kurulu dağıtım sürümleri.
- `complete.json` başarı kaydı ve beş temel çıktının SHA-256 bütünlüğü.
- Pose metadata `run_status=complete`, pozitif kare sayısı, motion summary kare
  sayısı ve summary'nin JSONL hash'i.

Yalnızca dosya varlığı yeterli değildir. Eşleşmeyen/kısmi/bozuk çıktılar korunur;
yeni `attempt-000002` klasörü açılır. `--no-resume` de yeni deneme oluşturur,
eski çıktıları silmez. `no_valid_motion` geçerli biçimde tamamlanmış analizi
belirtir; hareket bulunduğu iddiası değildir ve yeniden çalıştırılmaz.

Hatalı videoda diğer videolara geçilir. `failure.json` hata kategorisini kaydeder.
Mevcut video fonksiyonu başarısız MP4'ü temizler; JSONL öneki ve failed metadata
kalabilir. Analiz fonksiyonu kendi kısmi CSV/summary dosyalarını temizler.
Batch bunların bulunduğu deneme klasörünü silmez. İlk model yükleme başarılıysa
modeller tüm videolarda yeniden kullanılır; yükleme başarısızsa sonraki videoda
yeniden denenebilir. Videolar ve kareler RAM'de topluca tutulmaz.

`.batch.lock` aynı çıktı köküne eşzamanlı batch başlatmayı engeller. Normal
hata/bitişte kaldırılır. İşlem zorla öldürülürse kilit kalabilir; çalışan süreç
olmadığını doğruladıktan sonra yalnızca bu boş kilit klasörünü elle kaldırın.
Kesinti durumunda toplu rapor olmayabilir; önceki video durum dosyaları korunur.

## Sınırlamalar

Tracking yoktur. Çok köpekli karelerin hareket analizi mevcut güvenli tek-köpek
politikasını kullanır. Etiket, duygu veya ses analizi yapılmaz. Gerçek decoder
hatalarında pipeline'ın mevcut EOF davranışı korunur: OpenCV erken sonlanmayı
her zaman normal EOF'dan ayıramaz. Tam kaynak kare sayısı garantisi verilmez.
Hash doğrulaması uzun videolarda ek disk okuması gerektirir. Çalışma sırasında
kaynak ve ağırlıkları değiştirmeyin; CLI model hash'lerini başlangıçta alır.
Resume başka Python/paket/Git/kod durumunda ihtiyatlı biçimde yeniden işler.
`auto` cihaz donanım değişimini kesin tanımlamaz; açık `cpu`/`mps` tercih edin.
Süreler makine yüküne bağlıdır. Testler sahte pipeline ve dosyalarla yapılır;
model doğruluğunu, codec uyumluluğunu veya gerçek Mac performansını doğrulamaz.

## Resume teşhis kayıtları

Yeni raporlarda `resume_decision` adayların doğrulama sonucunu ve farklı kimlik
alan gruplarını içerir. Var olan iki attempt'in inference çalıştırmadan
karşılaştırılması ve pose kalite ölçümleri için [teşhis belgesine](pose_quality.md)
bakın. Resume doğrulama kuralları korunur.
