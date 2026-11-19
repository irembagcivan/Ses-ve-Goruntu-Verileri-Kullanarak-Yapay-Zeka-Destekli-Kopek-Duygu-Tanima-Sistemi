# İnsan duygu etiketleriyle gün gruplu değerlendirme

125 video, sınıflar {'Relaxed': 50, 'Happy': 32, 'Angry': 17, 'Sad': 26}, gruplar {'A': 65, 'B': 43, 'C': 17}.

| Özellik | C | macro-F1 | %95 GA | Doğruluk |
|---|---:|---:|---|---:|
| gorsel | 0.01 | 0.411 | [0.324, 0.496] | 0.44 |
| gorsel | 0.1 | 0.444 | [0.35, 0.52] | 0.48 |
| gorsel | 1.0 | 0.434 | [0.341, 0.521] | 0.472 |
| ses | 0.01 | 0.34 | [0.255, 0.413] | 0.432 |
| ses | 0.1 | 0.309 | [0.235, 0.384] | 0.392 |
| ses | 1.0 | 0.286 | [0.213, 0.359] | 0.376 |
| birlesik | 0.01 | 0.418 | [0.323, 0.506] | 0.464 |
| birlesik | 0.1 | 0.425 | [0.326, 0.511] | 0.472 |
| birlesik | 1.0 | 0.4 | [0.305, 0.493] | 0.448 |

Tüm C değerleri raporlanır; en iyisi test skoruna göre seçilmedi. Tesadüf seviyesi yaklaşık 0.25.
Bu bir bağımsız test değildir; üç kayıt grubu vardır ve etiketler gün ile kısmen karışıktır.
