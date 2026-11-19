"""Produce a standalone, filterable experiment report without external assets."""
import argparse
import html
import json
from pathlib import Path

TITLES={'aktivite':'Aktivite','vucut':'Vücut','durus':'Duruş','kuyruk':'Kuyruk','kulak':'Kulak'}
FEATURES={'visual':'Görüntü','audio':'Önceki ses','events':'Ses olayları','fusion':'Önceki birleşik',
          'fusion_events':'Birleşik + ses olayları','fusion_dog_events':'Birleşik + köpek olayları'}
ALGORITHMS={'logistic':'Lojistik regresyon','random_forest':'Random Forest','extra_trees':'Extra Trees'}


def render(experiment,baseline):
    records=experiment['candidates'];selected=[r for r in records if r['selected_on_training_cv']]
    esc=lambda text:html.escape(str(text))
    number=lambda value:f'{value:.3f}'
    def row(values):return '<tr>'+''.join('<td>'+str(v)+'</td>' for v in values)+'</tr>'
    def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(row(r) for r in rows)+'</tbody></table></div>'
    summary=[]
    for r in selected:
        old=next(m for m in baseline['metrics'] if m['target']==r['target'] and m['modality']=='fusion')
        lo,hi=r['test_delta_bootstrap_95_interval']
        summary.append([TITLES[r['target']],ALGORITHMS[r['algorithm']]+'<br><small>'+FEATURES[r['features']]+'</small>',
                        number(r['cv_macro_f1_mean'])+' ± '+number(r['cv_macro_f1_std']),
                        number(old['macro_f1']),number(r['test']['macro_f1']),
                        f"{r['test_delta_vs_old_fusion']:+.3f}",f'[{lo:+.3f}, {hi:+.3f}]'])
    feature_rows=[]
    for algorithm in ALGORITHMS:
        for features in FEATURES:
            group=[r for r in records if r['algorithm']==algorithm and r['features']==features]
            feature_rows.append([ALGORITHMS[algorithm],FEATURES[features],group[0]['feature_count'],
                                 number(sum(r['cv_macro_f1_mean'] for r in group)/len(group)),
                                 number(sum(r['test']['macro_f1'] for r in group)/len(group))])
    diagnostic=''
    for r in selected:
        target=r['target'];counts=experiment['class_counts'][target];labels=r['test']['classes'];matrix=r['test']['confusion_matrix']
        rows=[]
        for i,label in enumerate(labels):
            support=sum(matrix[i]);recall=matrix[i][i]/support if support else None
            rows.append([esc(label),counts['train'].get(label,0),counts['test'].get(label,0),number(recall) if recall is not None else 'Test örneği yok'])
        error_rows=[[esc(e['video_id']),esc(e['actual']),esc(e['predicted']),esc(e['old_fusion'])] for e in r['test_errors']]
        diagnostic+='<details><summary>'+TITLES[target]+' · sınıflar ve hatalar</summary>'
        diagnostic+=table(['Sınıf','Eğitim','Test','Test recall'],rows)
        diagnostic+=table(['Video','Gerçek davranış','Yeni tahmin','Önceki birleşik'],error_rows)
        diagnostic+=table(['Gerçek ↓ / tahmin →']+labels,[[esc(label)]+matrix[i] for i,label in enumerate(labels)])+'</details>'
    selected_mean=sum(r['test']['macro_f1'] for r in selected)/len(selected)
    old_mean=sum(m['macro_f1'] for m in baseline['metrics'] if m['modality']=='fusion')/len(selected)
    observations='<section><h2>Sonuçları nasıl okumalıyız?</h2><p>Beş davranışın test macro F1 ortalaması önceki birleşik modelde '+number(old_mean)+', eğitim CV seçimiyle '+number(selected_mean)+'. Bu sayı doğruluk yüzdesi değildir. Her davranış eşit ağırlıktadır.</p><p>Her davranışın seçilen modeli tabloda ayrı gösterilir. Bir duruş veya kulak pozisyonunun sesle ilişkili bulunması nedensellik veya başka köpeklere genelleme kanıtı değildir; ortam/köpek korelasyonu olabilir.</p><p class="notice">Önemli veri eksikliği: eğitimde '+str(experiment['class_counts']['durus']['train'].get('oturuyor',0))+' oturuyor, '+str(experiment['class_counts']['durus']['train'].get('sirtustu',0))+' sırtüstü ve '+str(experiment['class_counts']['kuyruk']['train'].get('asagi_bacak_arasi',0))+' kuyruk-aşağı/bacak-arası örneği var. Bazı CV katlarının eğitim kısmında nadir bir sınıf hiç bulunmuyor. Bu nedenle az görülen duruşları öğrenmek zor; yalnız model değiştirmek bu eksikliği çözmez. Öncelik eksik etiketleri tamamlamak ve bu sınıflara yeni, farklı köpeklerden örnek eklemek olmalı.</p></section>'
    payload=json.dumps([{k:v for k,v in r.items() if k not in {'oof','test_errors'}} for r in records],ensure_ascii=False).replace('<','\\u003c')
    return '''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Canine · Model deneyleri</title><style>
body{margin:0;background:#101815;color:#f1f5ec;font:15px system-ui;line-height:1.6}main{max-width:1250px;margin:auto;padding:30px}h1{font-size:36px;letter-spacing:-1px}h2{margin:0 0 12px}p,small{color:#afbfaf}section{background:#1a2520;border:1px solid #34463b;border-radius:18px;padding:24px;margin:24px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;border-bottom:1px solid #34463b;padding:12px;vertical-align:top}th{color:#b6ef83}select{padding:10px;background:#233329;color:#f1f5ec;border-radius:8px;border:1px solid #34463b;margin:8px}details{border-top:1px solid #34463b;padding:15px 0}summary{cursor:pointer;color:#b6ef83}.tag{display:inline-block;padding:8px 14px;background:#233329;border-radius:20px;margin:4px}.notice{border-left:4px solid #efcb83;padding-left:15px}tr.chosen{background:#2c3e2b}a{color:#b6ef83}</style><main>
<div class="tag">CANINE · 8 Kasım 2026</div><h1>Hangi model, hangi bilgiyle daha iyi?</h1>
<p>3 algoritma × 6 özellik grubu × 5 davranış = 90 aday değerlendirmesi. Her davranış için 18 aday yalnız eğitim verisindeki 3 katlı çapraz doğrulamayla karşılaştırıldı.</p>
<div class="tag">72 eğitim</div><div class="tag">25 test</div><div class="tag">30 eksik etiket · yalnız tahmin</div><div class="tag">270 CV eğitimi + 90 son eğitim</div>
<section><h2>Deney nasıl yapıldı?</h2><p>Kesin içerik kopyaları SHA256 ile aynı grupta tutuldu. Eğitim içindeki üç doğrulama katında aynı bölünmeler kullanıldı. Eksik değer doldurma ve ölçekleme her katın yalnız eğitim kısmında öğrenildi. Aday seçimi ortalama CV macro F1 ile yapıldı; eşitlikte sabit alfabetik sıra kullanıldı. Beş seçim kaydedildikten sonra test sonuçları hesaplandı.</p><p>Lojistik regresyon: standart ölçekleme, C=1, dengeli sınıf ağırlığı. Random Forest / Extra Trees: 300 ağaç, en çok 8 derinlik, yaprakta en az 2 örnek, dengeli sınıf ağırlığı, seed=42. Ses ve görüntü ilk 60 saniyeye dayanır. Ses olayları: sınıf başına ortalama, maksimum, yüzde 90 ve eşik üstü pencere oranı.</p><p class="notice">Bu test kümesi önceki geliştirmelerde incelendi; bağımsız yeni test değildir. Köpek kimlikleri bilinmediğinden aynı köpek veya ortam iki tarafta bulunabilir. Duygu doğruluğu ölçülmez. Sonuçlar keşifseldir; daha yüksek skor tek başına genelleme kanıtı değildir.</p></section>
<section><h2>Eğitim CV'sinin seçtiği modeller</h2><p>F1: 0–1, yüksek daha iyi. ± değer CV katlarının standart sapmasıdır; güven aralığı değildir. Son sütun, yeni ve önceki birleşik model arasındaki F1 farkının 1.000 eşleşmiş içerik-grubu bootstrap örneklemesinden %95 aralığıdır. Aralık sıfırı içeriyorsa artış konusunda güçlü bir çıkarım yapılamaz. Çoklu model seçimini düzeltmez.</p>'''+table(['Davranış','CV ile seçilen aday','CV F1 ± std','Önceki birleşik test F1','Seçilen test F1','Fark','Fark aralığı'],summary)+'''</section>'''+observations+'''
<section><h2>Özelliklerin katkısı: aynı algoritma, farklı girdiler</h2><p>Her satır beş davranışın macro F1 değerlerinin aritmetik ortalamasıdır; birleşik bir örnek doğruluğu değildir. Yalnız ses/görüntü, ikisinin birlikte kullanımı ve ses olaylarının eklenmesi ayrı ayrı karşılaştırılır. Köpek olayları varyantı konuşma/müzik olay skorlarını çıkarır; eski karma ortam sesi özelliklerini korur. Ses ayrıştırma yapılmaz.</p>'''+table(['Algoritma','Özellik grubu','Özellik sayısı','Ort. eğitim CV F1','Ort. test F1'],feature_rows)+'''</section>
<section><h2>90 adayın tamamı</h2><p>Satırlar eğitim CV skoruna göre sıralanır. Yeşil satır eğitimde seçilen adaydır. Test sütunu seçim ölçütü değildir. Eğitim doğruluğu ile test doğruluğu arasındaki fark aşırı uyuma işaret edebilir.</p><select id="target" aria-label="Davranış"><option value="all">Tüm davranışlar</option>'''+''.join('<option value="'+t+'">'+title+'</option>' for t,title in TITLES.items())+'''</select><select id="algorithm" aria-label="Algoritma"><option value="all">Tüm algoritmalar</option>'''+''.join('<option value="'+a+'">'+name+'</option>' for a,name in ALGORITHMS.items())+'''</select><div class="scroll"><table><thead><tr><th>Davranış</th><th>Algoritma</th><th>Girdi</th><th>CV F1</th><th>CV std</th><th>Test F1</th><th>Eğitim doğruluğu</th><th>Test doğruluğu</th></tr></thead><tbody id="candidates"></tbody></table></div></section>
<section><h2>Az görülen sınıflar ve somut hatalar</h2><p>Recall, o sınıfın test örneklerinden kaçının doğru bulunduğunu gösterir. Çok az örneği olan sınıflardaki oranlar kararsızdır. “Görünmüyor” etiketleri ilgili hedefin eğitim ve testinden çıkarılır.</p>'''+diagnostic+'''</section><footer><p>Ses olayları YAMNet; video davranışı sınıflandırıcıları yukarıda karşılaştırılır. İskelet çizimi ile sınıflandırma başarısı ayrı değerlendirilmelidir.</p></footer></main><script id="experiment-data" type="application/json">'''+payload+'''</script><script>
const rows=JSON.parse(document.getElementById('experiment-data').textContent),titles='''+json.dumps(TITLES,ensure_ascii=False)+''',features='''+json.dumps(FEATURES,ensure_ascii=False)+''',algorithms='''+json.dumps(ALGORITHMS,ensure_ascii=False)+''';
function draw(){const body=document.getElementById('candidates');body.replaceChildren();const target=document.getElementById('target').value,algorithm=document.getElementById('algorithm').value;rows.filter(r=>(target==='all'||r.target===target)&&(algorithm==='all'||r.algorithm===algorithm)).sort((a,b)=>b.cv_macro_f1_mean-a.cv_macro_f1_mean||a.candidate.localeCompare(b.candidate)).forEach(r=>{const tr=document.createElement('tr');if(r.selected_on_training_cv)tr.className='chosen';[titles[r.target],algorithms[r.algorithm],features[r.features],r.cv_macro_f1_mean.toFixed(3),r.cv_macro_f1_std.toFixed(3),r.test.macro_f1.toFixed(3),(r.train_accuracy*100).toFixed(1)+'%',(r.test.accuracy*100).toFixed(1)+'%'].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td)});body.append(tr)})}document.getElementById('target').onchange=draw;document.getElementById('algorithm').onchange=draw;draw();
</script></html>'''

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['experiment','baseline','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.write_text(render(json.loads(args.experiment.read_text()),json.loads(args.baseline.read_text())))
