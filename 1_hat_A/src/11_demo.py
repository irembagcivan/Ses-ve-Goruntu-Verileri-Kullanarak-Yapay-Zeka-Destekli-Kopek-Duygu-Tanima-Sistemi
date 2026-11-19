"""
Aşama 7 - Demo: yeni bir köpek videosundan davranış ve duygu tahmini
------------------------------------------------------------
Web sayfası (yerel, internete açılmaz):
    python src/11_demo.py                   -> tarayıcıda http://127.0.0.1:8100 açılır, video seçip "Analiz et"
Komut satırı:
    python src/11_demo.py --video yol/video.mov

Ne yapar: videodan 09_extract_features.py ile AYNI özellikleri çıkarır (MobileNetV2 + hareket + ses),
outputs/models/final_models.joblib içindeki modellerle (geç birleşim, lojistik regresyon) 5 hedefi tahmin eder.

UYARI (sayfada da yazar): Duygu etiketleri iki proje üyesinin insan gözlemine dayanır (Cohen kappa 0,55); modeller 127 video ile
eğitilmiştir, hata payı yüksektir. Bu bir araştırma prototipidir.
"""
import argparse
import importlib.util
import json
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

TR = {
    "ses3": ("Ses", {"yok": "Ses yok", "havlama": "Havlama", "diger": "Diğer ses (inleme/hırlama)"}),
    "durus3": ("Duruş", {"ayakta": "Ayakta", "yatiyor": "Yatıyor", "diger": "Diğer (oturuyor/sırtüstü)"}),
    "hareket3": ("Hareket", {"hareketsiz": "Hareketsiz", "yuruyor": "Yürüyor", "kosuyor_oynuyor": "Koşuyor / oynuyor"}),
    "kuyruk2": ("Kuyruk hareketi", {"salliyor": "Sallıyor", "sallamiyor": "Sallamıyor"}),
    "duygu4": ("Duygu (insan etiketi)", {"Happy": "Mutlu", "Sad": "Üzgün", "Angry": "Kızgın", "Relaxed": "Sakin"}),
}

HTML = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><title>Köpek davranış ve duygu demosu</title>
<style>
 body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#fafaf8;color:#0b0b0b}
 header{padding:18px 28px;border-bottom:1px solid #e6e5e0;background:#fff}
 h1{font-size:20px;margin:0 0 4px} .sub{color:#52514e;font-size:13px}
 main{display:flex;gap:28px;padding:22px 28px;flex-wrap:wrap}
 #left{width:460px;max-width:100%} #right{flex:1;min-width:360px}
 video{width:100%;background:#000;border-radius:10px}
 button{padding:10px 18px;border:0;border-radius:8px;background:#2a78d6;color:#fff;font-size:15px;cursor:pointer;margin-top:10px}
 button:disabled{background:#9db8dd;cursor:wait}
 input[type=file]{margin:10px 0;font-size:14px}
 .card{background:#fff;border:1px solid #e6e5e0;border-radius:12px;padding:12px 16px;margin-bottom:12px}
 .card h3{margin:0 0 8px;font-size:15px} .perf{font-size:12px;color:#52514e;margin-top:6px}
 .row{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:13px}
 .lab{width:210px} .bar{flex:1;height:14px;background:#efeee9;border-radius:4px;overflow:hidden}
 .fill{height:100%;background:#2a78d6;border-radius:4px} .fill.top{background:#1baf7a}
 .pct{width:44px;text-align:right;color:#52514e}
 .warn{background:#fff6e0;border:1px solid #ecd48a;border-radius:10px;padding:10px 14px;font-size:12.5px;color:#5c4a00}
 #msg{color:#52514e;font-size:13px;margin-top:8px}
 .low{color:#b25300;font-size:12px;margin-left:6px}
</style></head><body>
<header><h1>Köpek videosundan davranış ve duygu tahmini</h1>
<div class="sub">Ses + görüntü + hareket özellikleri, önceden eğitilmiş MobileNetV2 ve küçük sınıflandırıcılar (araştırma prototipi)</div></header>
<main><div id="left">
 <input id="file" type="file" accept="video/*,.mov,.mp4"><br>
 <video id="v" controls playsinline></video><br>
 <button id="go" disabled>Analiz et</button><div id="msg">Bir video seçin.</div>
</div><div id="right"><div id="out"></div>
 <div class="warn"><b>Önemli sınırlar:</b> Duygu etiketleri iki etiketleyicinin gözlemidir (orta düzey uyum), köpeğin gerçek iç durumunu doğrulamaz.
 Modeller yalnızca 127 videoyla eğitildi; skorlar düşüktür ve yeni ortamlarda değişebilir. Çok köpekli videolarda hangi köpeğin
 etiketlendiği belirsizdir. Yüzdeler kalibre edilmiş olasılık değildir; model olduğundan emin görünebilir. Tahminler karar vermek için değil, yöntemi göstermek içindir.</div>
</div></main>
<script>
const $=s=>document.querySelector(s);let blob=null;
$('#file').onchange=e=>{blob=e.target.files[0];if(!blob)return;$('#v').src=URL.createObjectURL(blob);$('#go').disabled=false;$('#msg').textContent=blob.name;$('#out').innerHTML='';};
$('#go').onclick=async()=>{
 $('#go').disabled=true;$('#msg').textContent='Analiz ediliyor (birkaç saniye)...';
 try{
  const r=await fetch('/api/predict?name='+encodeURIComponent(blob.name),{method:'POST',body:blob});
  if(!r.ok)throw new Error(await r.text());
  show(await r.json());$('#msg').textContent='Tamamlandı: '+blob.name;
 }catch(e){$('#msg').textContent='Hata: '+e.message;}
 $('#go').disabled=false;
};
function show(d){
 let h='';
 d.targets.forEach(t=>{
  const top=t.probs[0];
  h+=`<div class="card"><h3>${t.title}: <span style="color:#1baf7a">${top.label}</span>${top.p<0.5?'<span class="low">düşük güven</span>':''}</h3>`;
  t.probs.forEach((p,i)=>{h+=`<div class="row"><span class="lab">${p.label}</span><div class="bar"><div class="fill ${i==0?'top':''}" style="width:${(p.p*100).toFixed(0)}%"></div></div><span class="pct">${(p.p*100).toFixed(0)}%</span></div>`;});
  if(t.perf!=null)h+=`<div class="perf">Bu hedefte modelin görmediği günlerdeki başarısı (macro-F1): ${t.perf.toFixed(2)}  ·  rastgele tahmin referansı: ${t.base.toFixed(2)}</div>`;
  h+='</div>';
 });
 $('#out').innerHTML=h;
}
</script></body></html>"""


def load_extractor():
    spec = importlib.util.spec_from_file_location("extract", Path(__file__).with_name("09_extract_features.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor:
    def __init__(self, models_path, results_path, pretrained=True):
        self.ex = load_extractor()
        self.model, self.dev = self.ex.load_model(pretrained)
        self.models = joblib.load(models_path)
        self.lock = threading.Lock()
        self.perf = {}
        if Path(results_path).exists():
            m = pd.read_csv(results_path)
            m = m[(m.protokol == "oturum")]
            for t in TR:
                a = m[(m.hedef == t) & (m.model == "gec_birlesim_LR")]
                b = m[(m.hedef == t) & (m.model == "baz_cogunluk")]
                if len(a) and len(b):
                    self.perf[t] = (float(a.macro_f1.iloc[0]), float(b.macro_f1.iloc[0]))

    def predict(self, path):
        with self.lock:
            feats = self.ex.process(Path(path), self.model, self.dev)
        vm = np.concatenate([feats["vis"], feats["mot"]]).reshape(1, -1)
        aud = feats["aud"].reshape(1, -1)
        out = []
        for t, (title, names) in TR.items():
            m = self.models[t]
            p = (m["aud"].predict_proba(aud) + m["vm"].predict_proba(vm))[0] / 2
            order = np.argsort(-p)
            probs = [dict(label=names.get(m["classes"][i], m["classes"][i]), p=float(p[i])) for i in order]
            perf, base = self.perf.get(t, (None, None))
            out.append(dict(target=t, title=title, probs=probs, perf=perf, base=base))
        return dict(targets=out, frames=int(feats["n_frames"]))


class Handler(BaseHTTPRequestHandler):
    predictor: Predictor = None

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/":
            return self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        self._send(404, b"bulunamadi")

    def do_POST(self):
        if self.path.split("?")[0] != "/api/predict":
            return self._send(404, b"bulunamadi")
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > 1_500_000_000:
                raise ValueError("Geçersiz dosya boyutu")
            data = self.rfile.read(n)
            with tempfile.NamedTemporaryFile(suffix=".vid", delete=True) as f:
                f.write(data)
                f.flush()
                res = self.predictor.predict(f.name)
            self._send(200, json.dumps(res, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        except Exception as e:
            self._send(500, f"{type(e).__name__}: {e}".encode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, help="web sayfası yerine tek video için komut satırı tahmini")
    ap.add_argument("--models", default=Path("outputs/models/final_models.joblib"), type=Path)
    ap.add_argument("--results", default=Path("outputs/results/metrics.csv"), type=Path)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="yalnızca yazılım testi için")
    args = ap.parse_args()

    pred = Predictor(args.models, args.results, pretrained=not args.no_pretrained)
    if args.video:
        r = pred.predict(args.video)
        print(f"\nVideo: {args.video}  (kare: {r['frames']})")
        for t in r["targets"]:
            s = "  ".join(f"{p['label']} %{p['p'] * 100:.0f}" for p in t["probs"])
            print(f"  {t['title']:24s} {s}")
        print("\nUYARI: Duygu etiketleri insan gözlemidir; model 127 videoyla eğitildi, hata payı yüksektir.")
        return
    Handler.predictor = pred
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        raise SystemExit(f"{args.port} portu dolu. --port 8101 deneyin.")
    url = f"http://127.0.0.1:{args.port}"
    print(f"Demo hazır: {url}   (durdurmak için Ctrl+C)")
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nKapatıldı.")


if __name__ == "__main__":
    main()
