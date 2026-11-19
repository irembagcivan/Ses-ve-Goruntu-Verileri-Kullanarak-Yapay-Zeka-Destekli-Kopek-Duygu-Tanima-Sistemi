"""
Etiket şeması (Aşama 3)
-----------------------
Hem etiketleme aracı (04) hem özet script'i (05) bu dosyayı kullanır.
Şemayı değiştirirseniz ikisi de otomatik güncellenir.

Tasarım ilkeleri:
  * Duygu etiketi YOK; yalnızca gözlenebilir davranışlar etiketlenir.
  * Her özellik ayrı bir soru: köpek hem yürüyüp hem havlayabilir (hareket ve ses ayrı),
    kuyruk hem yukarıda hem sallanıyor olabilir (konum ve hareket ayrı).
  * Her özellik için BASKIN durum etiketlenir (kural aşağıda).
"""

RULES_VERSION = "v2 (12 Kasım)"

DOMINANT_RULE = "\n".join([
    "BASKIN DURUM: Bir özellik videoda değişiyorsa, videonun yarısından fazlasında görülen durumu seçin. "
    "Örnek: 11 sn'lik videoda 9. saniyede başlayan yürüme baskın değildir. Süreler eşitse videonun son yarısındaki durumu seçin.",
    "'DAVRANIŞ DEĞİŞİYOR' KUTUSU: Baskın OLMAYAN bir durum en az 2 saniye sürdüyse (veya köpek sayısı belirgin değiştiyse) "
    "işaretleyin. Baskın durumu seçmiş olmanız kutuyu işaretlemenize engel değildir.",
    "KÖPEK SAYISI: Videonun çoğunda net görünen köpekleri sayın. Kadrajdan yarıdan fazla kesilmiş köpekleri saymayın.",
    "ODAK KÖPEK YOKSA: Hiçbir köpek videonun yarısından fazlasında kadrajda kalmıyorsa 'Köpek sayısı: 3 veya daha fazla' "
    "ve 'Emin değilim' işaretleyin, notta 'odak köpek yok' yazın. Diğer sorular için en iyi tahmininizi verin.",
])

FIELDS = [
    dict(id="kopek_sayisi", kind="radio", title="Köpek sayısı",
         help="Videonun çoğunda net görünen köpekleri sayın; kadrajdan yarıdan fazla kesilmiş köpekler sayılmaz. Birden fazla köpek varsa en uzun süre kadrajda kalan (eşitse kameraya en yakın ve net) köpeği etiketleyin ve notta hangisi olduğunu yazın. Hiçbir köpek videonun yarısından fazlasında kadrajda kalmıyorsa '3 veya daha fazla' + 'Emin değilim' işaretleyin, notta 'odak köpek yok' yazın.",
         options=[("1", "1 köpek"), ("2", "2 köpek"), ("3_ve_uzeri", "3 veya daha fazla")]),
    dict(id="gorunurluk", kind="radio", title="Genel görünürlük",
         help="İyi: köpek net ve büyük ölçüde kadrajda. Kısmi: uzak, bulanık veya bir kısmı kadraj dışı. Kötü: etiketlemek çok zor.",
         options=[("iyi", "İyi"), ("kismi", "Kısmi"), ("kotu", "Kötü")]),
    dict(id="kuyruk_konum", kind="radio", title="Kuyruk konumu",
         help="Sallıyor olması konumu etkilemez; ayrı soruda işaretlenir. Kuyruk görünmüyorsa burada VE kuyruk hareketinde 'Görünmüyor' seçin. Yatan köpekte kuyruğun yerde durması 'bacak arası' sayılmaz; kuyruk karnın altına kıvrılmışsa sayılır.",
         options=[("yukari", "Yukarı (sırt hizasından yüksek / kıvrık)"),
                  ("notr", "Nötr (sırt hizasında / doğal sarkık)"),
                  ("asagi_bacak_arasi", "Aşağı / bacak arası"),
                  ("gorunmuyor", "Görünmüyor")]),
    dict(id="kuyruk_hareket", kind="radio", title="Kuyruk hareketi",
         help="Sallama = yanlara tekrarlı hareket. Emin olamazsanız videoyu 0.5x hızda izleyin. Kuyruk konumu 'Görünmüyor' ise burası da 'Görünmüyor' olmalı.",
         options=[("salliyor", "Sallıyor"), ("sallamiyor", "Sallamıyor"), ("gorunmuyor", "Görünmüyor")]),
    dict(id="kulak", kind="radio", title="Kulak durumu",
         help="Sarkık kulaklı ırklarda doğal duruş 'Nötr' sayılır.",
         options=[("dik_one", "Dik / öne dönük"), ("notr", "Nötr (doğal / gevşek)"),
                  ("geride_yassi", "Geriye yatık / başa yapışık"), ("gorunmuyor", "Görünmüyor")]),
    dict(id="durus", kind="radio", title="Duruş",
         help="Ayakta: dört ayak üzerinde (oyun selamında kalça havada olsa da ayakta). Oturuyor: kalça yerde, ön bacaklar dik, göğüs yerden yüksek. Yatıyor: göğüs veya karın yerde (başı kalkık olsa da).",
         options=[("ayakta", "Ayakta"), ("oturuyor", "Oturuyor"), ("yatiyor", "Yatıyor"),
                  ("sirtustu", "Sırtüstü")]),
    dict(id="hareket", kind="radio", title="Hareket",
         help="Hareketsiz: yerinde (baş çevirme olabilir). Yürüyor: yer değiştiriyor, her an en az bir ayağı yerde. Koşuyor/oynuyor: koşu veya oyun davranışı (zıplama, oyun selamı, kovalama).",
         options=[("hareketsiz", "Hareketsiz (yerinde)"), ("yuruyor", "Yürüyor (yavaş yer değiştirme)"),
                  ("kosuyor_oynuyor", "Koşuyor / oynuyor (hızlı hareket)")]),
    dict(id="ses", kind="radio", title="Ses (köpeğin çıkardığı)",
         help="Sesi açık dinleyin. Havlarken yürüyorsa hareket=Yürüyor, ses=Havlama seçin.",
         options=[("yok", "Yok"), ("havlama", "Havlama"), ("inleme", "İnleme / sızlanma"),
                  ("hirlama", "Hırlama"), ("diger", "Diğer")]),
    dict(id="degisken", kind="check", title="Davranış videoda değişiyor",
         label="Baskın olmayan bir durum en az 2 saniye sürdü (örn. oturuyordu, 5 sn sonra koşmaya başladı) ya da köpek sayısı belirgin değişti"),
    dict(id="guvensiz", kind="check", title="Emin değilim",
         label="Bu videodaki etiketlerimden emin değilim (sonra tekrar bakılacak)"),
    dict(id="not", kind="text", title="Not (isteğe bağlı)"),
]

# Birbiriyle çelişen cevaplar için araç uyarı gösterir (kaydı engellemez).
CONSISTENCY = [
    dict(a="kuyruk_konum", b="kuyruk_hareket", value="gorunmuyor",
         msg="Tutarsızlık: Kuyruk konumu ve kuyruk hareketinden biri 'Görünmüyor', diğeri değil. "
             "Kuyruk görünmüyorsa ikisinde de 'Görünmüyor' seçin."),
]

META_COLUMNS = ["video", "etiketleyici"]
CSV_COLUMNS = META_COLUMNS + [f["id"] for f in FIELDS] + ["son_guncelleme"]

RADIO_FIELDS = [f for f in FIELDS if f["kind"] == "radio"]


def allowed_values(field):
    if field["kind"] == "radio":
        return {v for v, _ in field["options"]}
    if field["kind"] == "check":
        return {"0", "1"}
    return None  # serbest metin


def is_complete(row):
    """Tüm zorunlu (radio) sorular cevaplanmışsa True."""
    return all(row.get(f["id"]) for f in RADIO_FIELDS)
