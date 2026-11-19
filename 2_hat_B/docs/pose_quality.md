# Pose kalitesi ve resume teşhisi — Görev 10A

Bu araç yalnızca mevcut JSONL verisini ölçer. Model çalıştırmaz, koordinatları veya
inference confidence eşiklerini değiştirmez. Mac'teki gerçek dosyalar bu bulut
ortamında bulunmadığından burada gerçek videoların kalitesi hakkında karar verilmedi.

## Kalite analizi

Depo kökünde, mevcut Mac ortamınızı etkinleştirin. Yeni çıktı yolları kullanın:

```bash
mkdir -p results/quality
python -m src.quality.run_quality \
  --input "results/batch/VIDEO_ID/attempt-000001/pose.jsonl" \
  --pose-metadata "results/batch/VIDEO_ID/attempt-000001/pose.metadata.json" \
  --motion-summary "results/batch/VIDEO_ID/attempt-000001/motion_summary.json" \
  --output results/quality/video1.json
```

`VIDEO_ID` gerçek `video-...` klasör adıyla değiştirilmelidir. Metadata ve motion
summary isteğe bağlıdır. Rapor bunlardan yalnızca güvenli alanları alır: tamamlanma,
kare sayısı/hash tutarlılığı, görselleştirme profili ve eşiği. `motion.csv` yeniden
hesaplanmaz veya değiştirilmez; hareket özetiyle kalite ölçümleri farklı kavramlardır.

İki yeni dosya oluşur:

- `video1.json`: toplamlar, oranlar, confidence ortalamaları, eşik başına ortalama
  görünür nokta sayısı, her orijinal keypoint için güvenilirlik, ayarlar ve sınırlamalar.
- `video1.frames.jsonl`: her girdi kaydı için kare indeksi, detection sayısı,
  köpek başına confidence/görünürlük/eksik-geçersiz nokta sayıları, örtüşen kutuların
  indeksleri ve IoU değerleri, zamansal değişim türleri. Çoklu köpek karelerinin
  indeksleri bu dosyada `dog_count > 1` kayıtlarında bulunur. Bütün kareler RAM'de tutulmaz.

### Metrik tanımları

- Toplam kare: artan, negatif olmayan frame index ve dogs listesi bulunan kayıtlar.
  Bozuk/tekrarlı kayıtlar ayrıca sayılır; gerçek video kare sayısı olarak uydurulmaz.
- Sıfır/tek/çoklu detection ve sıfır güvenilir nokta oranı: geçerli kayıt sayısına
  bölünür. Sıfır detection kareleri sıfır güvenilir nokta oranına dahildir;
  `detected_zero_reliable_frames` yalnız detection bulunan alt kümeyi sayar.
- Confidence ortalamaları: yalnız sonlu [0,1] skorlar üzerinden hesaplanır.
  Keypoint confidence ortalamasında görüntü dışı ama geçerli skorlu noktalar da
  bulunabilir; görünürlük bundan ayrı ölçülür. Hiç skor yoksa sonuç null'dır.
- Görünürlük: doğru orijinal name/index, sonlu görüntü içi koordinatlar, `in_frame`
  ve `confidence_valid` işaretleri ve confidence >= eşik. Sabit karşılaştırma
  eşikleri 0.3/0.5/0.7'dir. Ana güvenilirlik eşiği `--confidence` (varsayılan 0.5).
- Eksik nokta: 39 şemada hiç bulunmayan, adı/indeksi uyuşmayan veya tekrarlanan
  indeks. Mevcut ama null/sonsuz/görüntü dışı koordinat veya geçersiz skor ayrı
  `invalid_keypoints` olarak sayılır. Eksik nokta oranının paydası 39 × detection
  gözlemi; keypoint güvenilirlik paydası tüm detection gözlemleridir.
- Güvenilirlik: görüntü içi ve eşik üstü gözlem oranıdır; anatomik doğruluk değildir.
- Temporal karşılaştırma: yalnız ardışık indeksli, aynı çözünürlüklü, tek detection'lı
  iki karede yapılır. Kutuların IoU değeri en az 0.3 olmalıdır
  (`--temporal-min-iou`). Çoklu/boş/bozuk kare ve indeks boşluğu üzerinden köprü kurulmaz.
  Bu kontrol tracking değildir; aynı köpek kimliğini garanti etmez.
- `visible_lost/gained`, `below_threshold`, `missing`, `invalid`, `reappeared`
  durumları orijinal indekslerle ayrılır. Reappeared önceki karede güvenilir
  görünmeyip şimdi görünür olanı belirtir; tracking sonrası yeniden kimliklendirme değildir.
- Ani değişim: iki karede güvenilir noktanın Öklid adımı / ortalama bbox köşegeni
  > `--jump-ratio` (0.5). Bu sadece teşhis bayrağıdır; hareketin yanlış olduğu
  anlamına gelmez. Kamera, hızlı gerçek hareket veya crop değişimi de sebep olabilir.
- Aynı karede kutu IoU >= `--overlap-iou` (0.8) ise çift raporlanır. Kutu silinmez;
  iki gerçek köpek de örtüşebilir. IoU testi doğruluk değerlendirmesi değildir.

## İsteğe bağlı PNG teşhisi

Kalite komutuna aşağıdaki seçenekleri ekleyin; yeni rapor/dizin isimleri kullanın:

```bash
python -m src.quality.run_quality \
  --input "results/batch/VIDEO_ID/attempt-000001/pose.jsonl" \
  --output results/quality/video1_visual.json \
  --video "/path/to/Internship dataset/original.MOV" \
  --frames-dir results/quality/video1_png \
  --max-frames 8
```

İlk no-dog/multiple-dogs/few-keypoints/jump örnekleri seçilir (kategori başına ilk
örnek, en çok max-frames; few-keypoints köpek başına <=5 noktadır). Aynı kare için
0.3/0.5/0.7 ve seçilen ana eşikte PNG oluşturulur; frame index, detection confidence,
detection sayısı, güvenilir nokta sayısı ve eşik yazılır. Noktalar tüm 39 şemadan
gösterilir; bu görüntüler mevcut conservative iskelet görüntüsünün aynısı değildir.
Anatomik bağlantı çizilmez, model tahmini çalıştırılmaz. OpenCV yalnız bu seçenekle
gerekir. Kaynak video ardışık çözülür; bütün kareler RAM'e alınmaz. Kaynak dosya
hash'i JSONL ile eşleştirilemediğinden aynı orijinal videoyu vermek kullanıcının
sorumluluğudur; çözünürlük kontrolü tek başına kimlik doğrulaması değildir.

Dizin zaten varsa üzerine yazılmaz. PNG başarısızlığında kalite JSON/JSONL korunur,
CLI 2 döner; oluşturulmuş PNG öneki teşhis için kalabilir. Hiç bayraklı kare yoksa
PNG üretilmez. Orijinal video ve eski çıktılar değiştirilmez.

## Resume neden yeni attempt açmış olabilir?

Kodda resume varsayılan olarak açıktır. Önceki sürümün kimlik kontrolü şunları
karşılaştırır: kaynak video SHA-256, detector/pose/config SHA-256, eşikler/margin,
istenen cihaz, runtime paket/Python bilgisi, Git commit **ve dirty durumu**, tüm
`src`/`configs` kaynak hash'leri. Örneğin yalnız Git commit'i veya dirty durumu
bile değişse yeni attempt beklenir. Farklı `--pose-config` belirtme biçimi,
explicit config yerine varsayılan kullanımı, CPU yerine auto seçimi de kimliği
farklılaştırabilir. Bu olasılıklar gerçek Mac çalışmasının kanıtlanmış sebebi değildir.

Ayrıca complete.json, beş artifact hash'i, metadata complete durumu/kare sayısı,
motion summary kare sayısı ve JSONL hash'i doğrulanır. Aynı videonun farklı isimle
veya başka output root'unda çalıştırılması önceki attempt'in bulunmamasına yol açar.
`--no-resume` açıkça yeniden işler. CLI hatalı çıktı kökünü veya aktif/eski kilidi
reddeder. Bunların hiçbiri bu görevde gevşetilmedi.

Mevcut iki attempt'i inference yapmadan karşılaştırın:

```bash
python -m src.batch.diagnose_resume \
  --attempt "results/batch/VIDEO_ID/attempt-000001" \
  --reference "results/batch/VIDEO_ID/attempt-000002" \
  --output results/quality/video1_resume.json
```

İlk attempt'in çıktıları ikinci attempt'in `started.json` kimliğine karşı
kontrol edilir. `identity_mismatch` sonucunda yalnız farklı sabit alan grupları
(`git`, `runtime`, `code_sha256`, `pose_sha256` vb.) yazılır. Değerler/yollar
kopyalanmaz. Reference verilmezse attempt kendi kayıtlı kimliğine göre doğrulanır.
Geçerli olduğu gösterilen kayıt güncel diskteki model/video dosyalarının aynı
olduğunu kanıtlamaz. İlk başarısız kontrol raporlanır; birden fazla neden bulunabilir.

Yeni batch raporları `resume_decision` alanında enabled, reason ve adayların
kontrol sonuçlarını kaydeder: disabled / no_prior_attempts / verified_complete /
no_eligible_attempt. Adaylarda eksik marker, farklı kimlik, artifact hash farkı,
metadata uyuşmazlığı gibi nedenler ayrılır. Eski raporlar bu bilgiyi taşımadığı
için geçmiş kararı kesin yeniden oluşturmak mümkün olmayabilir.

**Gerekli ek kanıt:** iki attempt'in started.json/complete.json dosyaları,
metadata ve motion summary, her iki batch report.json ve kullanılan komutların
hassas yolları gizlenmiş seçenekleri. Ham video yüklemeden önce bu teşhis
raporlarını incelemek yeterli olabilir. Mevcut çıktıları silmeyin.

Bu yeni teşhis kodu da code_sha256 değerlerini değiştirdiğinden yeni batch
çalıştırılması eski kimliklerle eşleşmeyebilir. Önce salt-okunur diagnosis komutunu
kullanın; otomatik inference başlatmayın.

## Bulgular gelmeden ayar değiştirmeyin

Batch görselleştirmesi mevcut kodda conservative profildir: 39 model noktasının
yalnız **9** tanesi çizim için seçilir; bu alt küme confidence/sınır kontrolünden
sonra daha da küçülebilir. Ham JSONL tüm 39 noktayı korur. Dolayısıyla az nokta
çizilmesi tek başına modelin yalnız az nokta tahmin ettiğini göstermez. Gerçek
metadata ve aynı karedeki 39 noktalı PNG/0.3-0.5-0.7 karşılaştırmasıyla kontrol edin.

Önerilen sıra: veri/metadata bütünlüğü → detection boşlukları ve örtüşmeler → ham
39 nokta ile conservative çizim ayrımı → confidence/invalid/missing nedenleri →
zamansal bayraklı karelerin insan incelemesi → küçük elle anotasyonlu kalite seti.
Ancak bundan sonra crop/eşik/detector ayar deneyi; model değişimi veya temporal
smoothing/tracking ayrı, gerekçelendirilmiş sonraki görevler olmalıdır.
