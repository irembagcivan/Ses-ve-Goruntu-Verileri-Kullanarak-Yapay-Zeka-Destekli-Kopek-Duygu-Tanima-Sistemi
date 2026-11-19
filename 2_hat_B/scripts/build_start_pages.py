"""Export readable root-level entry/report pages; no server needed to read them."""
from pathlib import Path
from markdown_it import MarkdownIt

ROOT=Path(__file__).resolve().parents[1]
CSS='''*{box-sizing:border-box}body{margin:0;background:#101815;color:#eff5ed;font:17px system-ui;line-height:1.75}main{max-width:1100px;margin:auto;padding:30px 25px}h1,h2,h3,a{color:#b6ef83}h1{font-size:34px}h2{margin-top:40px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #3c5140;padding:12px;text-align:left;vertical-align:top}pre{overflow:auto;background:#1c2c22;padding:18px;border-radius:8px}code{overflow-wrap:anywhere}nav{display:flex;gap:24px;flex-wrap:wrap;padding:18px 0;border-bottom:1px solid #3c5140}blockquote{border-left:3px solid #b6ef83;padding-left:20px;color:#bdcbbd}'''
for source,destination,title in [('docs/review_and_day_validation.md','DEGERLENDIRME.html','Proje değerlendirmesi'),('PAYLASIM_KURULUM.md','KURULUM.html','Kurulum ve paylaşım')]:
    content=(ROOT/source).read_text()
    research=ROOT/'docs/literature_and_affect_v15.md'
    if destination=='DEGERLENDIRME.html' and research.exists():content=research.read_text()+'\n\n---\n\n'+content
    emotions=ROOT/'docs/four_emotions_bridge.md'
    if destination=='DEGERLENDIRME.html' and emotions.exists():content=emotions.read_text()+'\n\n---\n\n'+content
    body=MarkdownIt('commonmark',{'html':False}).enable('table').render(content)
    page='<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+CSS+'</style><main><nav><a href="BASLA.html">Proje başlangıcı</a><a href="http://127.0.0.1:8768/analyze">Video ekle ve analiz et</a><a href="http://127.0.0.1:8768/gallery">Galeri</a></nav>'+body+'</main></html>'
    (ROOT/destination).write_text(page)
    print('Exported',destination)
