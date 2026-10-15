# Çok Modlu Köpek Duygu Durumu Tanıma Sistemi (Video ve Ses)

Bu depo, gerçek dünya video ve ses kayıtlarından oluşan çok modlu bir veri seti kullanarak köpeklerin duygu durumlarını (**Mutlu, Üzgün, Kızgın, Rahat**) tanımaya yönelik veri hazırlama ve ön işleme boru hattını (pipeline) içermektedir.

## Proje Akışı ve Boru Hattı
1. **Veri Seti:** Mendeley Data üzerindeki "A Real World Dataset of Canine Emotions Using Video and Audio" veri seti kullanılmıştır.
2. **Otomatik Etiketleme (Gözetimsiz Kümeleme - Pseudo-Labeling):** 
   - Önceden eğitilmiş **ResNet18** modeli ile görsellerden öznitelik çıkarılmıştır.
   - Akustik öznitelikler çıkarılarak **PCA** ve **K-Means Kümeleme** algoritmalarıyla etiketlenmemiş ham kayıtlar 4 farklı duygu sınıfına otomatik olarak ayrılmıştır.
3. **Veri Sızıntısı Koruması (Data Leakage Prevention):** 
   - Veri sızıntısını önlemek amacıyla, veri seti pencerelere bölünmeden *önce* benzersiz video ID'leri baz alınarak **Eğitim (%80)** ve **Test (%20)** olarak ayrılmıştır.
4. **Veri Artırma ve Ön İşleme:**
   - `.mov` video akışları ve ses kanalları `imageio-ffmpeg` ve `torchaudio` kullanılarak işlenmiştir.
   - Sesler için **Log-Mel Spektrogramlar** üretilmiş ve videolar zaman pencerelerine bölünerek veri seti 127 ham kayıttan 505 işlenmiş çok modlu örneğe çıkarılmıştır.
