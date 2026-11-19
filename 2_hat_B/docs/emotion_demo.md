> **Tarihsel sürüm belgesi.** Beş hedef ve 72/25/30 sayıları güncel veri durumunu anlatmaz. Güncel altı hedef için [son etiket güncellemesi](labels_final_update.md), değerlendirme protokolü için [v14 denetimi](review_and_day_validation.md) kullanılmalıdır.

# Hızlı duygu sınıflandırma sürümü

Mevcut pose → motion CSV çıktısından video başına normalize hareket
istatistikleri çıkarılır. Random Forest veri setindeki duygu etiketlerini öğrenir.
Ses henüz kullanılmaz. Eksik poz ölçümleri sıfır hareket sayılmaz.

Eğitim manifesti CSV sütunları: `video_id,group_id,label,motion_csv`.
Her satır bir özgün videodur. `motion_csv` manifestin konumuna göre veya mutlak
yol olabilir. Etiketler: Happy, Sad, Angry, Relaxed. Aynı köpeğin veya özgün
kaydın kliplerine aynı group_id verin. Her sınıf için en az iki bağımsız grup
gerekir; daha fazla veri daha güvenilir sonuç sağlar.

```bash
python -m src.emotion.run train --manifest /absolute/path/emotions.csv --output results/emotion_v1
python -m src.emotion.run predict --model results/emotion_v1/model.joblib --motion results/dog_motion.csv
```

Çıktılar: model.joblib, metrics.json, validation_predictions.csv,
confusion_matrix.csv. Değerlendirme videoları grup bazında ayrılır;
metrics.json içindeki macro_f1 eğitimde kullanılmayan kat tahminlerinden gelir.
Model skorları kalibre edilmiş olasılık değildir. Yalnız yerelde oluşturduğunuz
güvenilir joblib dosyalarını açın.

İki günlük hedef: etiketli videoları bul → toplu pose/motion çıkar → manifest
oluştur → modeli eğit → ayrılmış videolardaki tahminleri göster.
Ardından ses özellikleri ve fusion eklenebilir. Hareket grafiği duygu tahmini
yerine sunulmamalıdır; eğitilmiş model olmadan duygu sonucu üretilmez.
