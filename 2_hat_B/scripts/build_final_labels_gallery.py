"""Export the final-label gallery using existing verified video previews."""
import argparse
import base64
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build(results):
    report=json.loads((results/'report.json').read_text())
    if (ROOT/'results/affect_v15_clean/report.json').exists():
        affect=json.loads((ROOT/'results/affect_v15_clean/report.json').read_text())
        fusion=json.loads((ROOT/'results/late_fusion_v15/report.json').read_text())
        report['research_v15']={'affect_selected':affect['protocol']['selected'],
                                'affect_metrics':next(m['metrics'] for m in affect['results'] if m['model']==affect['protocol']['selected']),
                                'fusion_f1':fusion['mean_macro_f1']}
    day_path=ROOT/'results/day_validation_v14/report.json'
    if day_path.exists():
        day=json.loads(day_path.read_text())
        report['day_validation']={k:day[k] for k in ['protocol','models']}
    old=json.loads((ROOT/'results/demo_v11/report.json').read_text())
    old={s['source_sha256']:s for s in old['samples']}
    for sample in report['samples']:
        key=sample['source_sha256'];clip=ROOT/'results/demo_v4/public/clips_v4'/f'{key}.mp4'
        if not clip.is_file():raise FileNotFoundError(clip)
        sample['video_data']='data:video/mp4;base64,'+base64.b64encode(clip.read_bytes()).decode('ascii')
        sample['audio_analysis']=old[key].get('audio_analysis',{})
        import cv2
        cap=cv2.VideoCapture(str(clip))
        try:
            if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS)<=0:raise ValueError('Undecodable preview '+str(clip))
            sample['preview_seconds']=round(cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS),1)
        finally:cap.release()
    report['limits']=[
        'Gün gruplu v14 yeniden doğrulama ayrıca gösterilir. Kamera tarihi doğrulanmış oturum veya hayvan kimliği değildir. Eski dosya bölmesi bağımsız yeni test değildir.',
        'Görüntü ve ses kodlayıcıları donmuş ön-eğitimli modellerdir; yeni etiketlerle tahmin başlıkları yeniden eğitildi.',
        'Duygu etiketi yok; mutlu/korkmuş/öfkeli sınıflandırıcısı doğrulanmış değildir.',
        'Test grubu daha önce incelendi. Tek seed ve küçük veri; bağımsız genelleme kanıtı değildir.',
        'Çoklu köpekte en büyük kutu etiketleyenin odak köpeğiyle uyuşmayabilir; kalıcı birey takibi yok.',
        'Ses etiketleri odak köpeğe ait olabilir; YAMNet karma kaydı dinler. Arka plandaki köpek havlaması ile odak köpeğin sessizliği birlikte bulunabilir.',
        'Hırlama yalnız bir kayıtta etiketli. Nadir sınıflarda güvenilir başarı iddiası kurulamaz.',
        'Video düzeyindeki etiketler kısa zaman pencerelerinin doğruluğunu ölçmez. 36 video değişken işaretli.',
        'Segmentasyon yeni video analizinde çalışır; galerideki eski önizlemelere maske eklenmedi. Davranış modelleri maskelerle eğitilmedi.',
        'Etiket notları ve kalite bayrakları yalnız maskeleme ve raporlama için kullanıldı; model girdisine alınmadı.'
    ]
    public=results/'public';public.mkdir(exist_ok=True)
    template=(ROOT/'src/demo/labels_gallery.html').read_text()
    html=template.replace('/*REPORT_DATA*/',json.dumps(report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c'))
    path=public/'canine_demo_v13.html';path.write_text(html)
    print('Exported',path,'bytes',path.stat().st_size,'videos',len(report['samples']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'results/demo_v13_release');build(p.parse_args().results)
