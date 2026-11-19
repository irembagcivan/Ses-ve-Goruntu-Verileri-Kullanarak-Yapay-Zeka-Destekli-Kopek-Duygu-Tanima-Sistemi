"""Package source, trained neural heads and a reproducible feature snapshot."""
from pathlib import Path
import hashlib,json,platform,zipfile,sys,subprocess
import numpy,pandas,scipy,sklearn,torch
ROOT=Path(__file__).resolve().parents[1];full='--full' in sys.argv;output=ROOT/('results/canine_tam_proje_paketi.zip' if full else 'results/canine_sunum_paketi.zip')
files=[]
for directory in ['src','configs','docs','tests','scripts']:
 for p in (ROOT/directory).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.png','.jpg','.mp4','.mov']:files.append((p,p.relative_to(ROOT).as_posix()))
for name in ['AIYE_ILET.md','BASLA.html','DEGERLENDIRME.html','KURULUM.html','START_ANALYSIS.command','START_ANALYSIS.bat','PAYLASIM_KURULUM.md','README.md','requirements.txt','requirements-demo.txt','requirements-defense.txt','requirements-share.txt']:
 p=ROOT/name
 if p.exists():files.append((p,name))
for name in ['neural_visual.pt','neural_audio.pt','neural_fusion.pt','neural_visual.json','neural_audio.json','neural_fusion.json','feature_snapshot.npz','selection.json','experiment.json','uncertainty.json','comparison.csv','raw_video_smoke.json']:
 files.append((ROOT/'results/demo_v9'/name,'artifacts/'+name))
for name in ['sequence_cnn.pt','sequence_gru.pt','sequence_bigru.pt','sequence_cnn.json','sequence_gru.json','sequence_bigru.json','sequence_snapshot.npz','experiment.json','selection.json']:
 p=ROOT/'results/demo_v11'/name
 if p.exists():files.append((p,'sequence_artifacts/'+name))
active_path=ROOT/'configs/demo/active_release.json'
active=json.loads(active_path.read_text()) if active_path.exists() else None
if active:
 active_root=(ROOT/active['model']).parent
 for name in ['neural_visual','neural_audio','neural_fusion','sequence_cnn','sequence_gru','sequence_bigru','sequence_lstm_lstm_gru']:
  for ext in ['.pt','.json']:
   p=active_root/(name+ext);files.append((p,p.relative_to(ROOT).as_posix()))
 for name in ['classical.joblib','selection.json','audit.json','report.json','manifest.csv','feature_snapshot.npz']:
  p=active_root/name;files.append((p,p.relative_to(ROOT).as_posix()))
 files.append((ROOT/'results/labels_v13/source_labels_extracted.csv','results/labels_v13/source_labels_extracted.csv'))
 files.append((ROOT/'results/demo_v4/manifest.csv','results/demo_v4/manifest.csv'))
 if active.get('day_validation'):
  for name in ['report.json','protocol.json','folds.json','fold_1_selection.json','fold_2_selection.json','fold_3_selection.json']:
   p=ROOT/'results/day_validation_v14'/name;files.append((p,p.relative_to(ROOT).as_posix()))
  for name in ['recording_days.csv','audit.json']:
   p=ROOT/'results/recording_day_audit_v14'/name;files.append((p,p.relative_to(ROOT).as_posix()))
 if active.get('affect_model'):
  affect_root=(ROOT/active['affect_model']).parent
  for name in ['model.json','selection.json','neural.pt','classical.joblib','report.json','data_audit.json','features.npz','manifest.csv']:
   p=affect_root/name;files.append((p,p.relative_to(ROOT).as_posix()))
  for name in ['LICENSE','provenance.json','husky_train_labels.csv','husky_test_labels.csv','shiba_train_labels.csv','shiba_test_labels.csv']:
   p=ROOT/'results/affect_v15_external/data'/name;files.append((p,p.relative_to(ROOT).as_posix()))
 if active.get('late_fusion'):
  fusion_root=(ROOT/active['late_fusion']).parent
  for name in ['deployment.json','report.json','fold_1_selection.json','fold_2_selection.json','fold_3_selection.json']:
   p=fusion_root/name;files.append((p,p.relative_to(ROOT).as_posix()))
if full:
 for name in ['yamnet.h5','yolov8n.pt','yolov8m.pt','yolov8s-seg.pt','pose/superanimal_quadruped_rtmpose_s.pt']:files.append((ROOT/'models'/name,'models/'+name))
 for name in ['neural_fusion.pt','neural_fusion.json']:files.append((ROOT/'results/demo_v9'/name,'results/demo_v9/'+name))
 for kind in ['cnn','gru','bigru']:
  for ext in ['.pt','.json']:
   name='sequence_'+kind+ext;files.append((ROOT/'results/demo_v11'/name,'results/demo_v11/'+name))
files.append((ROOT/'results/defense_build/canine_sunum.pdf','sunum/canine_sunum.pdf'))
readme='''# Sunum paketi

1. Sunum PDF: sunum/canine_sunum.pdf
2. Etkileşimli slaytlar: docs/hoca_sunumu.html (ok tuşları, N notlar, F tam ekran)
3. Konuşma notları: docs/sunum_konusma_notlari.md
4. Mimari, deney ve komutlar: docs/deep_learning.md

Tam 127-videolu galeri bu bilgisayarda results/demo_v11/public/canine_demo_v11.html
konumundadır. Kaynak videolar bu arşive dahil değildir. Ön-eğitimli ağırlıkların kapsamı aşağıdaki paket türünde belirtilir.

## Ham videolar olmadan eğitimi tekrar çalıştırma

Bu klasörün kökünde, mevcut Python/torch ortamında:

python -m src.deep_learning.reproduce --snapshot artifacts/feature_snapshot.npz --metadata artifacts/neural_fusion.json --reference artifacts/neural_fusion.pt --output reproduced_fusion.pt

Aynı CPU ortamında 60 epoch tekrar eğitimi referansın bütün ağırlık tensörlerini
birebir yeniden üretti. Yeni çıktı dosyası adı kullan. Üç neural checkpoint,
normalizasyon parametreleri, özellikler, ayrım indeksleri, seçim ve eğriler arşivdedir.
Ortam sürümleri environment.json dosyasında kayıtlıdır.

## Sunumda doğru anlatım

Ön-eğitimli YOLOv8n, RTMPose-S ve YAMNet donmuştur. Eğittiğimiz kısım iki özellik
dalı, birleşim katmanı ve beş görev başlığıdır. Sonuçlar davranış etiketlerine aittir.
Duygu yorumları kurala dayanır. Kısa pencere doğruluğu ölçülmedi. Test küçük ve
önceden incelendi. Klasik CV referansı ortalama F1'de derin ağı geçiyor.
'''
environment={'python':platform.python_version(),**{m.__name__:m.__version__ for m in [numpy,pandas,scipy,sklearn,torch]}}
if active:
 readme='''# Son etiketler v13 / Gün değerlendirmesi v14 / Ses duygu v15

Son insan etiketleriyle altı hedef yeniden eğitildi. Yeni video analizi yeni
modelleri kullanır. Önce PAYLASIM_KURULUM.md dosyasını okuyun.

- Sunum: sunum/canine_sunum.pdf veya docs/hoca_sunumu.html
- Değişiklik ve deney raporu: docs/labels_final_update.md
- Arkadaşa / başka yapay zekâya tam bağlam: AIYE_ILET.md
- Gün gruplu yeniden doğrulama: docs/review_and_day_validation.md
- Yeni ses duygu modeli ve karar karışımı: docs/literature_and_affect_v15.md
- Kontrol noktaları ve sonuçlar: results/demo_v13_release/
- Mac: START_ANALYSIS.command; Windows: START_ANALYSIS.bat
- Hazır 127-videolu galeri ayrı HTML olarak verilir; ZIP'e dahil değildir.

Paket eski yeniden üretim özelliklerini de içerir. Ön-eğitimli çalışma ağırlıkları
yalnız tam pakettedir. Eski beş hedefli deneyler geçmiş karşılaştırma içindir;
güncel uygulama configs/demo/active_release.json sürümünü kullanır.

Son etiketli eğitimi tekrarlamak için, paket kökünde:
python -m src.deep_learning.final_labels --features artifacts/feature_snapshot.npz --sequences sequence_artifacts/sequence_snapshot.npz --metadata artifacts/neural_fusion.json --output results/demo_v13_repeat

Ham videolar eğitim tekrarında gerekli değildir; donmuş özellikler kullanılır.
Yeni ham video analizi için tam paket ağırlıkları gerekir. Veriler davranış ve
ses etiketidir; duygu sınıflandırması doğrulanmış değildir.
'''
if full:readme+='\n## Tam analiz paketi\nMac: START_ANALYSIS.command. Windows: START_ANALYSIS.bat. Önce PAYLASIM_KURULUM.md dosyasını oku.\nÖn-eğitimli çalışma ağırlıkları da bu tam pakete dahildir. Kaynak videolar ve büyük hazır galeri dahil değildir. Python ortamını hazırladıktan sonra START_ANALYSIS.command veya python -m src.demo.upload_server ile yeni video analizi açılır.\n'
hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for path,name in files}
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
 for path,name in files:archive.write(path,name)
 archive.writestr('RELEASE.json',json.dumps({'version':active['version'] if active else 'v12-segmentation','evaluation_version':active.get('evaluation_version') if active else None,'research_version':active.get('research_version') if active else None,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_tree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip(),'models_included':full},indent=2));archive.writestr('BASLA.md',readme);archive.writestr('environment.json',json.dumps(environment,indent=2));archive.writestr('CHECKSUMS.json',json.dumps(hashes,indent=2))
with zipfile.ZipFile(output) as archive:assert archive.testzip() is None
print('Validated release:',output,'files:',len(files),'bytes:',output.stat().st_size)
