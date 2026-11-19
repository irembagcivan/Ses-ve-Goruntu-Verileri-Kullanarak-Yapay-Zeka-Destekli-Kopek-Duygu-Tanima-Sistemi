"""
Duygu etiketi şeması (Aşama 3f)
-------------------------------
Davranış etiketlerinden (schema.py) AYRI bir aşamadır. Etiketleme aracı --stage duygu ile bu şemayı kullanır.
Çıktı: labeling/labels_duygu_<etiketleyici>.csv

ÖNEMLİ (raporda da yazılacak): Bu etiketler köpeğin gerçek iç durumu değil,
İNSAN ETİKETLEYİCİNİN ALGILADIĞI duygudur. İki etiketleyici bağımsız etiketler; uyum (kappa) raporlanır.
"""

RULES_VERSION = "duygu-v1 (14 Kasım)"

DOMINANT_RULE = "\n".join([
    "BAĞIMSIZ ETİKETLEYİN: Afra ve irem birbirinin duygu etiketini görmeden, ayrı ayrı etiketleyin.",
    "BASKIN DUYGU: Videoda duygu değişiyorsa videonun yarısından fazlasında hissedilen duyguyu seçin; değişimi notta belirtin.",
    "SES ve GÖRÜNTÜ birlikte değerlendirilir: sesi açın. Havlama tek başına 'kızgın' demek değildir; vücut ve bağlama bakın.",
    "EMİN DEĞİLSENİZ: 'Belirsiz' seçmek serbesttir. Zorla bir sınıf seçmeyin. Belirsiz videolar modele girmez ve raporda sayısı verilir.",
    "ODAK KÖPEK: Birden fazla köpek varsa davranış aşamasında odaklandığınız köpeği etiketleyin.",
])

FIELDS = [
    dict(id="duygu", kind="radio", title="Algılanan duygu durumu",
         help=("Sakin: gevşek ve rahat, düşük heyecan; dinleniyor, yavaş yürüyor ya da çevreyi izliyor. "
               "Mutlu: neşeli ve pozitif yüksek heyecan; oyun, oyun selamı, gevşek vücutla sallanan kuyruk, neşeli havlama. "
               "Kızgın: tehditkâr ya da agresif; hırlama, diş gösterme, sert ve gergin havlama, dik gergin duruş. "
               "Üzgün: sönük, çekingen ya da sıkıntılı; sızlanma/inleme, düşük baş, bacak arası kuyruk, kaçınma (korku/stres belirtileri de buraya girer). "
               "Belirsiz: video hiçbir sınıfı güvenle seçtirmiyor."),
         options=[("sakin", "Sakin"), ("mutlu", "Mutlu"), ("kizgin", "Kızgın"),
                  ("uzgun", "Üzgün (sönük / çekingen / sıkıntılı)"), ("belirsiz", "Belirsiz")]),
    dict(id="guven", kind="radio", title="Bu etiketten ne kadar eminsiniz?",
         help="Raporda güvenilirlik analizi için kullanılır.",
         options=[("yuksek", "Emin"), ("orta", "Kısmen emin"), ("dusuk", "Emin değilim")]),
    dict(id="degisken", kind="check", title="Duygu videoda değişiyor",
         label="Baskın olmayan bir duygu en az 2 saniye sürdü (örn. sakindi, sonra oyuna başladı)"),
    dict(id="not", kind="text", title="Not (isteğe bağlı)"),
]

CONSISTENCY = []

META_COLUMNS = ["video", "etiketleyici"]
CSV_COLUMNS = META_COLUMNS + [f["id"] for f in FIELDS] + ["son_guncelleme"]

RADIO_FIELDS = [f for f in FIELDS if f["kind"] == "radio"]


def allowed_values(field):
    if field["kind"] == "radio":
        return {v for v, _ in field["options"]}
    if field["kind"] == "check":
        return {"0", "1"}
    return None


def is_complete(row):
    return all(row.get(f["id"]) for f in RADIO_FIELDS)
