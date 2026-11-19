> **Tarihsel sürüm belgesi.** Beş hedef ve 72/25/30 sayıları güncel veri durumunu anlatmaz. Güncel altı hedef için [son etiket güncellemesi](labels_final_update.md), değerlendirme protokolü için [v14 denetimi](review_and_day_validation.md) kullanılmalıdır.

# Çalışan davranış ve multimodal demo

## V4: 127 videonun tamamı ve daha sıkı poz gösterimi

Kullanıcının özellikle işaretlediği kalabalık videolar için
`src.demo.crowd_preview` yerel YOLOv8m'yi 960 çözünürlükte doğrudan kullanır.
Bu açık bir görüntüleme ayarıdır; duygu/başarı etiketlerine göre model seçimi
yapılmaz ve test metrikleri değiştirilmez. Seçilen klibin metadata'sında
`explicit_high_recall_crowd_960` modu kaydedilir. Örneğin:

```bash
python -m src.demo.crowd_preview --manifest results/demo_v4/manifest.csv --results results/demo_v4 --detector /path/yolov8m.pt --pose /path/superanimal_quadruped_rtmpose_s.pt --video-id IMG_5532.MOV
```

Galeri artık 72 eğitim, 25 test ve 30 eksik/etiketsiz videoyu içerir.
Etiketler ve önceki model eğitimi değiştirilmez. Başarı tablosu yalnız eski
25 test videosuna aittir; eğitim/etiketsiz videolara ait tahminler ayrı
işaretlenir ve bu metriklere eklenmez.

Görselleştirmede yerel YOLOv8s kullanılır; görüntüde köpek bulunamazsa yerel
YOLOv8m ile 960 çözünürlükte tekrar denenir. Hiç kutu
bulunmayan karelerde 180 derece döndürme ve merkez kırpımı da denenir. Kutular
asıl koordinatlara döner, yinelenen kutular süzülür. Döndürülmüş kutuların pozu
aynı yönde hesaplanıp noktalar orijinal kareye dönüştürülür. Kedi sınıfı köpek
diye yeniden adlandırılmaz. İki ek ağırlığın kaynağı resmî Ultralytics
[v8.3.0 release](https://github.com/ultralytics/assets/releases/tag/v8.3.0)
varlıklarıdır; dosyalar Git'e girmez.

İleri/geri Lucas–Kanade optik akış kontrolü, yeni model noktalarını görüntü
hareketiyle karşılaştırır. Uyuşmayan noktalar, zayıf yeni noktalar ve uzunluğu
ani değişen bağlantılar gizlenir. Yüz, bacak, sırt ve kuyruk bağlantılarının
uzunluğu ayrıca sınırlanır. Görüntü kesmesi, yalnız piksel farkıyla değil renk
dağılımı değişimiyle birlikte kontrol edilir. Bu filtre modelin yanlış eklem
konumunu anatomik olarak düzelttiği anlamına gelmez; güvenilir iskelet yoksa
az nokta veya yalnız kutu gösterilebilir. Özellikle üstten, kısmi veya örtüşen
köpekler hâlâ zor örneklerdir.

V4'te model, gösterilen her üçüncü karede yenilenir: 12 kare/sn önizlemede
yaklaşık 4 model çıkarımı/sn vardır. Aradaki iki karede noktalar sabit
tutulmaz; ileri/geri doğrulanan optik akışla o anki görüntüye taşınır.
Görüntü hareketi doğrulanmazsa bu noktalar gizlenir. Metadata, model ve
görüntüleme FPS'lerini ayrı kaydeder; her karede yeni model çıkarımı yapıldığı
iddia edilmez. Ağır dedektör kurtarma denemeleri 0,35 saniyeden sık yapılmaz;
ölçülmüş kutular bu sürede güvenilir görüntü özellikleriyle takip edilebilir.

```bash
python -m src.demo.gallery inventory --workbook /path/labels.xlsx --videos /path/videos --trained results/demo_v1/trained --output results/demo_v4
python -m src.demo.extract --manifest results/demo_v4/new_manifest.csv --output results/demo_v4/new_features --detector /path/yolov8n.pt --pose /path/superanimal_quadruped_rtmpose_s.pt
python -m src.demo.gallery predict --manifest results/demo_v4/manifest.csv --trained results/demo_v1/trained --old-features results/demo_v1/features/features.csv --new-features results/demo_v4/new_features/features.csv --output results/demo_v4
python -m src.demo.dense_previews --manifest results/demo_v4/manifest.csv --results results/demo_v4 --detector /path/yolov8s.pt --rescue /path/yolov8m.pt --pose /path/superanimal_quadruped_rtmpose_s.pt --stable --clip-directory clips_v4 --fps 12
python -m src.demo.serve --results results/demo_v4 --static --clip-directory clips_v4 --filename canine_demo_v4.html
```

Model tahminlerini besleyen özellikler önceki YOLOv8n/RTMPose ve ses akışından
gelir; S/M algılayıcıları ve sıkı optik akış filtresi önizleme içindir. İnsan
etiketleri ana köpek/video düzeyindedir; diğer köpeklere ayrı duygu etiketi
uydurulmaz. Önizlemeler test videolarında en çok 12, yeni eğitim/etiketsiz kliplerde en çok 6 saniyedir; daha önce hazırlanmış uzun önizlemeler korunabilir. Tam orijinal videolar için
ekrandaki klasör seçiciyle kaynak klasörü seçip “Orijinal videonun tamamı”
görünümüne geçilebilir. Dosyalar tarayıcı içinde yerel Blob URL'leriyle açılır;
bir sunucuya yüklenmez. Orijinaller çizimsizdir. MOV codec desteği tarayıcıya
bağlıdır; işlenmiş önizlemeler H.264/AAC MP4'tür.

Tüm galeriyi CPU'da işlerken OpenCV/Torch thread sayısını sınırlayan
`src.demo.render_gallery` kullanılabilir. `--worker-count 3` ve farklı
`--worker-index 0`, `1`, `2` komutları ayrık video alt kümelerini işler;
aynı worker indeksini aynı çıktı köküne eşzamanlı başlatmayın. Varsayılan
`--threads 2` yalnız Torch içindir; OpenCV 1 thread ile çalışır.
Her video sonrası durum yazılır; tamamlanmış aynı model/filtre ayarlı videolar
korunur. İş parçacığı sayısı farklılığı model eğitimi veya eşik değişikliği
anlamına gelmez. Yeni çizimler mevcut eğitim ve test ölçümlerini değiştirmez.

```bash
python -m src.demo.render_gallery --manifest results/demo_v4/manifest.csv --results results/demo_v4 --detector /path/yolov8s.pt --rescue /path/yolov8m.pt --pose /path/superanimal_quadruped_rtmpose_s.pt --threads 1 --worker-count 3 --worker-index 0
```

Bu sürüm, doldurulmuş kuyruk/kulak/duruş/aktivite/vücut etiketlerini kullanır.
Doğrudan duygu etiketi bulunmadığı için duygu doğruluğu raporlanmaz. Happy,
Sad, Angry, Relaxed veya Belirsiz gösterimi davranış tahminlerinden türeyen,
`src/demo/train.py` içindeki açık örnek kurallardır.

## Akış

1. `src.demo.prepare`: Excel'in ilk sayfasını salt okunur işler. Beş davranışı
   da dolu videoları seçer; eksikleri audit.json'a yazar. Bilinen
   `IMG_5417.MOV` → `IMG_5417 kopyası.MOV` ad farkını kaydederek çözer.
2. `src.demo.extract`: ilk 60 saniyeyi yaklaşık 2 kare/sn ile örnekler.
   YOLOv8 kutusundaki en büyük köpeğin RTMPose noktalarını çıkarır. 0,3
   altındaki, kare dışındaki ve antler noktaları özellik hesabına alınmaz.
   Kutu merkezine göre ve diyagonal ile normalize koordinatların ve ardışık
   tek-köpek karelerindeki hızların özetleri kullanılır. Çoklu köpek karelerinde
   hareket çifti kurulmaz. Tracking bulunmaz.
3. Ses, aynı ilk 60 saniyeden mono 16 kHz alınır. RMS, sıfır geçiş oranı,
   spektral merkez/yayılım/düzlük/enerji ve 13 MFCC özeti hesaplanır.
4. `src.demo.train`: görüntü, ses ve erken özellik birleştirme için ayrı
   Random Forest modelleri kurar; her biri beş davranışı tahmin eder.
5. `src.demo.previews`: test videolarının 12 saniyelik, sesli ve poz noktalı
   önizlemelerini üretir. Noktalar seyrek örneklerden gelir; sürekli tracking değildir.
6. `src.demo.serve --static`: çevrimdışı, tarayıcıda doğrudan açılabilen ekranı
   oluşturur. İsteğe bağlı sunucu modu yalnız localhost'a bağlanır.

### V3: sık kareli ve çok köpekli iskelet önizlemesi

Eski 2 kare/sn ana-köpek özelliklerinden video çizmek yerine, her gösterilen
karede yeni detection/pose işlemi yapılır. Çıktı en çok 15 kare/sn'dir; tüm
algılanan köpeklere ayrı poz çıkarılır. Renkler ve Dog numaraları kısa süreli
kutu-IoU eşleştirmesidir; kalıcı kimlik tanıma değildir. 0,4 altında güvenli
olmayan, kutu/kare dışı ve aşırı sıçrayan noktalar çizilmez. Küçük hareketler
hafifçe yumuşatılır; köpeğin kutu hareketi telafi edilir. Görünmeyen köpeğin
eski noktaları sonraki karede çizilmez.

Mevcut projede tanımlanmış 26 noktalı / 18 bağlantılı çizim rehberi kullanılır.
Resmî SuperAnimal metadata'sı boş skeleton içerir; bağlantılar proje rehberidir.
Görünmeyen eklemler tamamlanmaz. Örtüşme, küçük köpekler ve model hataları
nedeniyle bazı kutular/noktalar yine yanlış veya eksik olabilir.

```bash
python -m src.demo.dense_previews --manifest results/demo_v1/manifest.csv --results results/demo_v1/trained --detector /path/yolov8n.pt --pose /path/superanimal_quadruped_rtmpose_s.pt
python -m src.demo.serve --results results/demo_v1/trained --static --clip-directory clips_v3 --filename canine_demo_v3.html
```

V3 klipleri ayrı `public/clips_v3` klasörüne yazılır; önceki klipler korunur.
Her video yanında kare sayısı, FPS, çoklu köpek kareleri, en yüksek detection
sayısı ve ayar/checkpoint hash'leri kaydedilir. Aynı ayarlı tamamlanmış
kliplerden devam edilebilir. Yeni klipler tek HTML'ye gömülür.

Bu değişiklik önizleme içindir. Davranış modelleri hâlâ insan etiketlerinin
ait olduğu ana köpek/video düzeyindedir. Diğer köpeklere aynı davranış veya
duygu tahmini kopyalanmaz. Ses tüm videoya aittir. Eğitim ve başarı değerleri
bu görselleştirme değişikliğiyle yeniden hesaplanmış değildir.

## Komutlar

Mevcut çalışan pose ortamını kullanın; gerekirse ek bağımlılıklar
`requirements-demo.txt` içindedir. Paket kurulumu her çalıştırmada gerekmez.
`python` mevcut pose ortamının Python'udur. Komutlar depo kökünde çalışır.

```bash
python -m src.demo.prepare --workbook /path/labels.xlsx --videos /path/videos --output results/demo_v1
python -m src.demo.extract --manifest results/demo_v1/manifest.csv --output results/demo_v1/features --detector /path/yolov8n.pt --pose /path/superanimal_quadruped_rtmpose_s.pt
python -m src.demo.train --manifest results/demo_v1/manifest.csv --features results/demo_v1/features/features.csv --output results/demo_v1/trained
python -m src.demo.previews --manifest results/demo_v1/manifest.csv --features results/demo_v1/features --results results/demo_v1/trained
python -m src.demo.serve --results results/demo_v1/trained --static
```

`results/demo_v1/trained/public/index.html` dosyasını tarayıcıda açın. Rapor ve
25 MP4 önizleme sayfanın içine gömülüdür; internet, sunucu veya ayrı clips
klasörü gerekmez. Tek HTML dosyasını indirip veya taşıyıp açabilirsiniz.
`--filename canine_demo_v2.html` ile farklı isimli taşınabilir çıktı üretilebilir.
İsterseniz `--static` olmadan yerel sunucu başlatıp
http://127.0.0.1:8765 adresini açabilirsiniz; durdurmak için Ctrl+C.
Özellik çıkarımı bitmiş, aynı ayarlı videoları önbellekten alır. Model eğitimi
için yeni çıktı klasörü gerekir; mevcut sonuçların üzerine yazılmaz.

Yeni bir video üzerinde uçtan uca tahmin:

```bash
python -m src.demo.predict --video /path/new.mov --model results/demo_v1/trained/models.joblib --detector /path/yolov8n.pt --pose /path/superanimal_quadruped_rtmpose_s.pt --output results/demo_v1/new_prediction.json
```

Yalnız güvenilir, yerelde oluşturulan joblib modellerini açın. Kaynak video
ve model checkpoint dosyaları değiştirilmez veya Git'e eklenmez.

## Sonuçların anlamı

- Tek sabit yaklaşık %75 eğitim / %25 test ayrımı kullanılır. Ayrım, sonuçlara
  bakılarak değil etiket sınıflarının kapsamına göre seçilir. Aynı dosya içeriğinin
  SHA256 değeri iki tarafa bölünmez. Köpek kimliği olmadığı için farklı
  videolarda aynı köpeğin bulunması engellenememiştir.
- Değerlendirilen sınıflarda `gorunmuyor`, davranış hedefi olarak kullanılmaz.
  Eksik poz/ses değerleri sıfır hareket veya sessizlik yerine eksik bırakılır.
  İmputer yalnız eğitim bölümünde öğrenilir. Final modeller de sadece eğitim
  bölümüne fit edilir; test ekranı eğitim tahminlerini göstermez.
- Doğruluk, balanced accuracy, macro F1 ve karışıklık matrisi kaydedilir.
  Majority baseline yalnız eğitim bölümünün en sık etiketini kullanır.
- Ses, insan konuşması ve ortam gürültüsünü de içerir. Köpek sesi ayrıştırılmaz.
- Vücut/duygu yorumları hayvanın gerçek iç durumunu doğrulamaz. Model skorları
  kalibre edilmiş olasılık değildir. Ekran davranış skorlarını el ile tanımlanmış
  ağırlıklarla karşılaştırıp en yakın duygu yorumunu gösterir. En yüksek kural
  desteği 0,60 altında veya ilk iki yorum arasındaki fark 0,12 altında ise
  destek sınırlı olarak işaretlenir ve yakın alternatif gösterilir. Bu eşikler
  kalibre edilmemiştir. JSON'da belirsiz sonuçlar `label: Belirsiz` olarak
  korunur; `candidate` yalnız en yakın örnek yorumdur. Kural puanları duygu
  olasılığı veya ölçülmüş duygu başarısı değildir.
- İlk sürümde 97 tam etiketli video seçilmiştir. Kalan 30 kayıt, boş/kısmi
  davranışlar nedeniyle eğitimden çıkarılmıştır.
