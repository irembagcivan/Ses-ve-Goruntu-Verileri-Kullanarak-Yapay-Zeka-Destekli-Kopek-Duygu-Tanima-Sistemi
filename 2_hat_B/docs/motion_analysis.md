# Görev 8 — JSONL'den zamansal hareket özellikleri

Bu modül mevcut Görev 7 JSONL verisini **offline** analiz eder. Video, YOLO,
DeepLabCut veya ağırlık dosyası yüklemez; inference, tracking, smoothing, ses
analizi veya duygu sınıflandırması yapmaz. CSV/JSON için yalnız Python standart
kütüphanesi gerekir; model isimleri depodaki mevcut metadata JSON'undan okunur.

## Mac üzerinde kullanım

Mevcut ortamınızı etkinleştirip depo kökünde:

```bash
python -m src.analysis.run_motion \
  --input results/dog_pose.jsonl \
  --output results/dog_motion.csv \
  --summary results/dog_motion_summary.json \
  --confidence 0.5 \
  --normalization bbox_diagonal \
  --max-step-ratio 0.5 \
  --min-region-points 2
```

Dosya adlarını gerçek JSONL yolunuzla değiştirin. Çıktı klasörleri önceden var
olmalı ve CSV/JSON hedefleri yeni dosyalar olmalıdır. Mevcut PNG üzerine
yazılmaz; grafik atlanır ve CSV/JSON analizi devam eder. Hiçbir giriş veya
mevcut çıktı üzerine yazılmaz. JSONL'nin companion metadata dosyası zorunlu
değildir; bu analiz export'un `complete` durumunu bağımsız doğrulamaz.

İsteğe bağlı grafik:

```bash
python -m src.analysis.run_motion \
  --input results/dog_pose.jsonl \
  --output results/dog_motion_plot_run.csv \
  --summary results/dog_motion_plot_run.json \
  --plot results/dog_motion.png
```

Matplotlib zaten varsa PNG üretilmeye çalışılır; yoksa kurulum yapılmaz.
Grafik başarısızlığı summary'de belirtilir; CSV/JSON korunur. Grafik en çok
yaklaşık 2000 örnekle seyreltilmiş bir görünüm sunar, tüm karelerin incelemesi
değildir. Kaynak zaman hataları ve seyreltilen aralıklardaki eksikler grafikte
her ayrıntısıyla görünmeyebilir; sayısal karar için CSV/JSON kullanın.

Komut, en az bir geçerli hareket karesi varsa 0; hiç geçerli hareket çifti
yoksa CSV/JSON üretip `no_valid_motion` durumuyla 2 döndürür. Yapısal dosya/
ayar hatalarında 2 verir. `complete`, analizin bittiğini belirtir; model
anatomik doğruluğunu veya davranış etiketini kanıtlamaz.

## Modüler yapı

- `reader.py`: tek geçişli okuma, veri/geçerlilik denetimi ve input SHA-256.
- `regions.py`: resmî isimlerden yüzey işareti grupları.
- `features.py`: ardışık kare hesapları, normalizasyon ve sıçrama işaretleri.
- `report.py`: CSV akışı ve çevrimiçi özet istatistikler.
- `plotting.py`: yalnız istendiğinde Matplotlib import eden opsiyonel katman.
- `run_motion.py`: komut satırı arayüzü.

Tüm video/kareler RAM'e alınmaz. Bellek, bir JSONL satırı/kare ve sabit sayıda
istatistikle sınırlıdır; aşırı büyük tek bir satır yine büyük bellek kullanabilir.
Grafik için CSV ikinci kez, seyreltilmiş noktalarla okunur.

## Güvenli tek köpek politikası

Bir kare ancak başlık alanları geçerliyse, **tam bir detection** varsa,
detection confidence eşiği sağlanıyorsa ve kutu sonlu/pozitif alanlı/kare
sınırları içindeyse adaydır. Detection eşiği `--detection-confidence` ile
ayarlanır; varsayılan 0.0'dır, fakat [0,1] dışı/eksik değerler daima reddedilir.

Sıfır detection → `no_dog`; birden fazla → `multiple_dogs`. Bu karelerde
hareket değerleri eksiktir ve önceki kare zinciri sıfırlanır. Böyle bir kareyi
atlayıp öncesi/sonrası birleştirilmez. Geçerli tek köpekle geri dönülen ilk kare
yalnız yeni başlangıçtır (`baseline`). Kalıcı dog_id veya detection indeksine
dayalı kimlik varsayımı yoktur.

Bu politika **ardışık iki tek-detection karesinde farklı köpeklerin görünmesini
kesin olarak engelleyemez**. Tracking yoktur; analiz öncelikle gerçekten tek
köpek içeren video içindir. Sıçrama filtresi kimlik doğrulama sistemi değildir.

Kare indeksleri artmalı ve hareket çifti tam ardışık olmalıdır. Eksik indeks
`frame_gap`; tekrarlanan/geriye giden indeks `nonincreasing_frame_index` üretir.
Çözünürlük değişiminde çift kullanılmaz. Pozitif zaman farkı olmayan çift
hesaplanmaz; `dt=0` veya negatif değerler sıfır hıza çevrilmez. Geçersiz çiftin
sonraki kullanılabilir karesi yeni zaman/konum referansı olabilir; eski bir
noktayla boşluk üzerinden hız üretilmez.

## Keypoint güvenilirliği

Noktanın JSONL'deki **orijinal indeks ve ismi** metadata ile uyuşmalıdır.
Eksik noktalar atlanır; yerlerine başka nokta veya sıfır konmaz. Tekrarlanan
indeksler o noktanın tamamını geçersiz yapar. X/Y sonlu ve kare içinde olmalı;
confidence seçilen eşik ile 1 arasında olmalı; `in_frame` ve `confidence_valid`
işaretleri açıkça `true` olmalıdır. İşaretlerin eksik olduğu eski/başka formatlı
dosyalar otomatik olarak güvenilir sayılmaz.

Dört antler noktası hareket özelliklerinden çıkarılır; diğer 35 isim global
harekete adaydır. Kaynak dosya, model isimleri veya sırası değiştirilmez.
Baş/gövde/kuyruk grupları aşağıda açıkça belirtilmiştir. Bir nokta iki karede
de güvenilir değilse o noktanın türevi hesaplanmaz. Eksik nokta üzerinden
interpolasyon veya geçmiş geçerli kareye atlama yapılmaz.

Malformed JSON/UTF-8 satırı eksik bir analiz satırı oluşturur ve zinciri keser.
Yinelenen JSON alanları belirsizlik nedeniyle reddedilir. Standart dışı NaN/
Infinity sabitleri `null` kabul edilir; sayıları summary'de belirtilir. Normal
Task 7 export'unda bu sabitler zaten bulunmaz. Nonnumeric/boolean değerler
koordinat veya confidence olarak kabul edilmez.

## Matematiksel tanımlar

Aynı isimli güvenilir nokta için ardışık zamanlar `t0,t1`, konumlar `p0,p1`:

```text
dt = t1 - t0 > 0
d = ||p1 - p0||₂
v = d / dt
```

`d` piksel, `v` piksel/saniyedir. Sabit bir nokta için d/v gerçekten 0'dır;
eksik veri CSV'de boş alan, JSON istatistiklerinde `null` olur.

Normalizasyon ölçeği:

```text
D0 = hypot(bbox_width_0, bbox_height_0)
D1 = hypot(bbox_width_1, bbox_height_1)
S = (D0 + D1) / 2
d_normalized = d / S
v_normalized = d / S / dt
```

Normalize displacement birimi kutu diyagonali, normalize hız birimi
kutu diyagonali/saniyedir. `--normalization none`, normalize sütunları boş
bırakır; piksel metrikleri hesaplanmaya devam eder. Sıçrama kontrolü yine
kutu diyagonalini kullanır. Aynı sahnenin çözünürlükle ölçeklenmesi için yararlıdır;
kutu boyutu, bakış açısı, zoom, poz ve detection hataları ölçeği etkiler.
**Kamera hareketini tamamen düzeltmez, fiziksel metre/saniye vermez.**

### Mutlak ve göreli hareket

Bölge için iki karede ortak güvenilir ve sıçrama olmayan noktaların d/v
ortalamaları ve maksimumları alınır. Varsayılan `--min-region-points 2`;
daha az ortak nokta varsa bölge özellikleri boş kalır. Her bölgede
`*_valid_points` sayısı ayrıca verilir. Global metrik en az bir kabul edilen
noktadan hesaplanabilir; kısmi kapsama `partial` olarak belirtilir.

Mutlak hareket orijinal kare piksellerindedir. Göreli hareket için her karenin
**detection kutu merkezi** `c` kullanılır:

```text
q0 = p0 - c0
q1 = p1 - c1
d_relative = ||q1 - q0||₂
```

Bu referansın seçilme nedeni belirli bir eklem/kalça/omuz konumu uydurmamaktır.
Baş veya gövdeye ait gerçek anatomik merkez olduğu iddia edilmez. Kutu merkezi
jitter'ı göreli hareket yaratabilir; bu yöntem dönüş/zoom/perspektif düzeltmez.
`bbox_step_ratio` ve `bbox_scale_ratio` tanı amaçlı ayrıca kaydedilir.

### Bölge eşlemeleri

| Grup | Model isimleri | Yorum sınırı |
|---|---|---|
| Baş | nose, upper_jaw, lower_jaw, mouth_end_right, mouth_end_left, right_eye, right_earbase, right_earend, left_eye, left_earbase, left_earend | Yüz/baş yüzey işaretleri; kafatası veya eklem ölçümü değil |
| Gövde | neck_base, neck_end, throat_base, throat_end, back_base, back_end, back_middle, belly_bottom, body_middle_right, body_middle_left | Boyun/gövde etiket grubu; kesin omurga/eklem veya kütle merkezi değil |
| Kuyruk | tail_base, tail_end | İki uç işareti; kuyruk eğriliği veya ara omurlar yok |

Diğer 12 uzuv etiketi yalnız global harekete katılabilir; `thai`/`knee`
anatomik eklemlere çevrilmez. Grup eşlemesi resmî bir anatomik skeleton değildir.

Kuyruk için ek göreli vektör:

```text
r0 = tail_end_0 - tail_base_0
r1 = tail_end_1 - tail_base_1
tail_vector_displacement = ||r1 - r0||₂
tail_vector_speed = tail_vector_displacement / dt
```

Dört uç gözlemi de güvenilir ve kabul edilmiş olmalıdır. Sabit bir kuyruk
vektörünün tamamı çevrilirse vektör değişimi 0 olur. Açı, wag frekansı, duygu
veya kuyruk eğriliği çıkarılmaz.

## Sıçrama ve eksik veri

`d / S > max_step_ratio` ise nokta çifti şüpheli sıçrama olarak işaretlenir
(varsayılan 0.5). Bu nokta özet hareket değerlerinden çıkarılır; adı/sayısı ve
`flagged_max_displacement_px` tutulur. Bütün noktalar işaretliyse hareket
sıfır değil eksiktir (`all_points_flagged`). Bazı noktalar kaldıysa özellikler
yalnız kalanlarla hesaplanır. Ham JSONL değiştirilmez.

Bu **kare adımı başına** heuristiktir; FPS'den bağımsız fiziksel eşik değildir.
Gerçek hızlı hareket/kamera hareketini yanlışlıkla dışlayabilir; daha küçük
hataları kaçırabilir. Eşik veri üzerinde incelenmelidir. Otomatik smoothing
uygulanmaz; ileride ham veri ve geçerlilik maskelerini koruyarak ayrı bir
smoothing değerlendirmesi yapılabilir.

## Çıktılar ve özetlerin yorumlanması

CSV'de kare/line kimliği, zaman kaynağı, durum, geçerli/eksik/sıçrama nokta
sayıları, dt, normalizasyon ve bölgesel kapsama bulunur. Özellik önekleri:

- `global_*`: 35 aday noktanın kabul edilen alt kümesi.
- `head_absolute_*`, `head_relative_*` ve body/tail karşılıkları.
- `tail_vector_*`: kuyruk tabanına göre vektör değişimi.

Her grup için mean/max displacement ve speed, piksel ve normalize biçimleri
vardır. Bunlar kabul edilmiş **aynı noktaların ayrı ayrı hareketlerinin**
özetidir; iki karenin farklı noktalı centroid'leri karşılaştırılmaz.

Summary JSON toplam/kimliği okunabilen kayıtları, geçerli/kısmi analiz
karelerini, uygun zaman çiftlerini, durum sayaçlarını, sıçrama sayılarını,
eşikleri, bölge eşlemelerini, input SHA-256'yı, Git commit/dirty bilgisini ve
sınırlamaları içerir. Dirty=true ise commit tüm çalışan kaynakları tek başına
tanımlamaz.
Her metrikte gözlem sayısı, mean/max ve eksik oranları bulunur. Ortalama,
mevcut **kare metriklerinin ağırlıksız ortalamasıdır**; zaman veya nokta sayısı
ile ağırlıklandırılmaz. Mean/max hız eksikse `null` kalır.

`total_frames`, boş olmayan giriş satırlarının sayısıdır; bozuk bir satırın
hangi kareye ait olduğu bilinmeyebilir. `identified_frame_records` bunun
ayrımını gösterir. Düzenli 834 satırlık export'ta toplam 834 olur; bu görevde
kullanıcının gerçek dosyası görülmediği için böyle bir sonuç üretilmedi.
İlk kare türev için baseline'dır ve doğal olarak hareket değeri yoktur.
`missing_rate_all_frames` bu kareyi de içerir;
`missing_rate_eligible_pairs` yalnız ardışık/geçerli zaman çiftlerini kullanır.
Hiç uygun çift yoksa oran `null` olur. Eksik oranı model doğruluk skoru değildir.

Summary'ye kullanıcı dosya adı, mutlak yol veya ham hata mesajı kaydedilmez.
Input hash'i dosyayı ilişkilendirebilir; raporları paylaşırken bunu değerlendirin.
Analiz hatasında bu çalışmanın kısmi CSV/JSON dosyaları temizlenir; mevcut
çıktılar üzerine yazılmaz. Kaynak JSONL değişmez. Başka süreç dosyayı analiz
sırasında değiştirmemelidir.

## Doğrulama kapsamı

Sentetik testler sabit noktayı, 3-4-5 mesafesi ve dt'yi, confidence/flag/null/
NaN/Inf filtrelerini, eksik noktaları, çoklu/boş köpek karelerinde zincir kesmeyi,
sıfır/negatif dt'yi, frame gap'i, sıçramaları, normalizasyonu, bölgesel mutlak/
göreli farkları ve kuyruk vektörünü doğrular. CSV/JSON, opsiyonel grafik
başarısızlığı ve aynı input/ayarlarla byte düzeyinde tekrar üretim de test edilir.
Bunlar gerçek video üzerinde davranış/anatomik doğruluk testi değildir.
