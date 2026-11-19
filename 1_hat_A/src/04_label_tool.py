"""
Aşama 3b - Yerel etiketleme aracı
---------------------------------
Videoyu tarayıcıda oynatır, yanındaki formdan davranış etiketlerini seçersiniz.
Her seçim otomatik kaydedilir: labeling/labels_<etiketleyici>.csv
Yalnızca kendi bilgisayarınızda (127.0.0.1) çalışır, internete açılmaz.

Çalıştırma (önce 03_make_proxies.py çalışmış olmalı):
    python src/04_label_tool.py --annotator irem
Tarayıcı kendiliğinden açılır; açılmazsa http://127.0.0.1:8000 adresine gidin.
Durdurmak için terminalde Ctrl+C.

Duygu aşaması (ayrı dosya labels_duygu_<ad>.csv):
    python src/04_label_tool.py --annotator irem --stage duygu

Arkadaşınız etiketlerken: --annotator arkadas_adi  (ayrı CSV dosyası oluşur;
daha sonra iki etiketleyici arasındaki uyumu bu dosyalardan hesaplayacağız.)
"""
import argparse
import csv
import json
import os
import re
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

import schema

HTML = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<title>Köpek davranış etiketleme</title>
<style>
 *{box-sizing:border-box}
 body{font-family:-apple-system,system-ui,sans-serif;margin:0;height:100vh;display:flex;color:#222;font-size:14px}
 #side{width:210px;border-right:1px solid #ddd;display:flex;flex-direction:column;background:#fafafa}
 #side h3{margin:10px;font-size:14px}
 #filter{margin:0 10px 8px;padding:4px}
 #list{overflow:auto;flex:1}
 .item{padding:5px 10px;cursor:pointer;border-left:5px solid #ccc;font-size:13px}
 .item:hover{background:#eee}
 .item.partial{border-left-color:#e0a800}
 .item.done{border-left-color:#28a745}
 .item.flag::after{content:" ⚑";color:#c00}
 .item.current{background:#dbe9ff;font-weight:600}
 #main{flex:1;display:flex;flex-direction:column;min-width:0}
 #top{padding:8px 14px;border-bottom:1px solid #ddd;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 #top .title{font-weight:700;font-size:16px}
 #status{color:#28a745;margin-left:auto}
 #status.err{color:#c00;font-weight:700}
 #warn{display:none;background:#ffe9e9;border:1px solid #e0a0a0;color:#900;border-radius:8px;padding:6px 10px;margin-bottom:8px;font-size:12px}
 #rule{white-space:pre-line}
 #ver{font-size:11px;color:#888}
 #body{flex:1;display:flex;min-height:0}
 #videoCol{flex:1;padding:10px;display:flex;flex-direction:column;align-items:center;background:#111;min-width:0}
 #v{max-width:100%;max-height:calc(100vh - 170px);background:#000}
 #ctrls{margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;justify-content:center}
 button{padding:6px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer;font-size:13px}
 button:hover{background:#eee}
 #formCol{width:430px;overflow:auto;padding:10px 14px;border-left:1px solid #ddd}
 fieldset{border:1px solid #ddd;border-radius:8px;margin:0 0 8px;padding:6px 10px}
 legend{font-weight:700;padding:0 4px}
 .help{font-size:12px;color:#666;margin-bottom:4px}
 label.opt{display:block;padding:2px 0;cursor:pointer}
 textarea{width:100%;height:50px}
 details{background:#fff7e0;border:1px solid #ecd48a;border-radius:8px;padding:6px 10px;margin-bottom:8px;font-size:12px}
 #nav{display:flex;gap:8px;margin:8px 0}
</style></head>
<body>
<div id="side">
 <h3>Videolar</h3>
 <select id="filter">
  <option value="all">Tümü</option><option value="todo">Tamamlanmamış</option>
  <option value="flag">Emin değilim ⚑</option><option value="done">Tamamlanan</option>
 </select>
 <div id="list"></div>
</div>
<div id="main">
 <div id="top"><span class="title" id="title"></span><span id="prog"></span><span id="ver"></span><span id="status"></span></div>
 <div id="body">
  <div id="videoCol">
   <video id="v" controls loop playsinline></video>
   <div id="ctrls">
    <button data-rate="0.25">0.25x</button><button data-rate="0.5">0.5x</button><button data-rate="1">1x</button>
    <button id="rewind">⏮ Başa sar</button><button id="pp">⏯ Oynat/Durdur (boşluk)</button>
   </div>
  </div>
  <div id="formCol">
   <details open><summary>Kurallar (özet)</summary><div id="rule"></div></details>
   <div id="warn"></div>
   <div id="form"></div>
   <div id="nav">
    <button id="prev">← Önceki (P)</button><button id="next">Sonraki → (N)</button>
    <button id="reset">Bu videoyu sıfırla</button>
   </div>
  </div>
 </div>
</div>
<script>
const $=s=>document.querySelector(s);
let SCHEMA,STATE,cur=0,saveTimer=null;
const RADIO=()=>SCHEMA.fields.filter(f=>f.kind==='radio');
const isDone=n=>{const l=STATE.labels[n]||{};return RADIO().every(f=>l[f.id]);};
const isStarted=n=>{const l=STATE.labels[n]||{};return Object.entries(l).some(([k,v])=>v&&v!=='0'&&k!=='etiketleyici'&&k!=='video'&&k!=='son_guncelleme');};
function setStatus(t,err){const s=$('#status');s.textContent=t;s.className=err?'err':'';}
function buildForm(){
 const box=$('#form');box.innerHTML='';
 SCHEMA.fields.forEach(f=>{
  const fs=document.createElement('fieldset');
  const lg=document.createElement('legend');lg.textContent=f.title;fs.appendChild(lg);
  if(f.help){const h=document.createElement('div');h.className='help';h.textContent=f.help;fs.appendChild(h);}
  if(f.kind==='radio'){
   f.options.forEach(([v,l])=>{const lb=document.createElement('label');lb.className='opt';
    const i=document.createElement('input');i.type='radio';i.name=f.id;i.value=v;i.addEventListener('change',onChange);
    lb.appendChild(i);lb.appendChild(document.createTextNode(' '+l));fs.appendChild(lb);});
  }else if(f.kind==='check'){
   const lb=document.createElement('label');lb.className='opt';
   const i=document.createElement('input');i.type='checkbox';i.name=f.id;i.addEventListener('change',onChange);
   lb.appendChild(i);lb.appendChild(document.createTextNode(' '+f.label));fs.appendChild(lb);
  }else{
   const t=document.createElement('textarea');t.name=f.id;t.addEventListener('input',onChange);fs.appendChild(t);
  }
  box.appendChild(fs);
 });
}
function fillForm(n){
 const l=STATE.labels[n]||{};
 SCHEMA.fields.forEach(f=>{
  if(f.kind==='radio'){document.querySelectorAll(`input[name="${f.id}"]`).forEach(i=>i.checked=(l[f.id]===i.value));}
  else if(f.kind==='check'){document.querySelector(`input[name="${f.id}"]`).checked=(l[f.id]==='1');}
  else{document.querySelector(`textarea[name="${f.id}"]`).value=l[f.id]||'';}
 });
 checkWarn();
}
function collect(){
 const v={};
 SCHEMA.fields.forEach(f=>{
  if(f.kind==='radio'){const c=document.querySelector(`input[name="${f.id}"]:checked`);v[f.id]=c?c.value:'';}
  else if(f.kind==='check'){v[f.id]=document.querySelector(`input[name="${f.id}"]`).checked?'1':'0';}
  else{v[f.id]=document.querySelector(`textarea[name="${f.id}"]`).value;}
 });
 return v;
}
function checkWarn(){
 const v=collect(),w=[];
 (SCHEMA.consistency||[]).forEach(r=>{const a=v[r.a],b=v[r.b];if(a&&b&&((a===r.value)!==(b===r.value)))w.push(r.msg);});
 const el=$('#warn');el.textContent=w.join(' ');el.style.display=w.length?'block':'none';
}
function onChange(e){
 checkWarn();
 if(e&&e.target&&e.target.type!=='textarea'&&document.activeElement)document.activeElement.blur();
 setStatus('Kaydediliyor...');clearTimeout(saveTimer);saveTimer=setTimeout(save,300);
}
async function save(){
 saveTimer=null;
 const idx=cur,name=STATE.videos[idx].name,values=collect();
 STATE.labels[name]=values;updateItem(idx);updateProgress();
 try{
  const r=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video:name,values})});
  if(!r.ok)throw new Error(await r.text());
  setStatus('Kaydedildi ✓');
 }catch(e){setStatus('KAYDEDİLEMEDİ: '+e.message,true);}
}
async function flush(){if(saveTimer){clearTimeout(saveTimer);await save();}}
function updateItem(i){
 const d=document.querySelector(`.item[data-i="${i}"]`);if(!d)return;
 const n=STATE.videos[i].name,done=isDone(n);
 d.classList.toggle('done',done);d.classList.toggle('partial',!done&&isStarted(n));
 d.classList.toggle('flag',(STATE.labels[n]||{}).guvensiz==='1');
 d.classList.toggle('current',i===cur);
 const f=$('#filter').value,fl=(STATE.labels[n]||{}).guvensiz==='1';
 d.style.display=(f==='all'||(f==='todo'&&!done)||(f==='done'&&done)||(f==='flag'&&fl)||i===cur)?'':'none';
}
function updateProgress(){
 let d=0,p=0;STATE.videos.forEach(v=>{if(isDone(v.name))d++;else if(isStarted(v.name))p++;});
 $('#prog').textContent=`Tamamlanan: ${d} / ${STATE.videos.length}  (yarım: ${p})`;
}
function renderList(){
 const L=$('#list');L.innerHTML='';
 STATE.videos.forEach((v,i)=>{const d=document.createElement('div');d.className='item';d.dataset.i=i;d.textContent=v.name;d.onclick=()=>select(i);L.appendChild(d);});
 STATE.videos.forEach((_,i)=>updateItem(i));
}
async function select(i){
 await flush();
 const old=cur;cur=Math.max(0,Math.min(STATE.videos.length-1,i));
 const v=STATE.videos[cur];
 $('#title').textContent=`${v.name}  (${cur+1}/${STATE.videos.length})`+(v.duration?`  ·  ${v.duration.toFixed(1)} sn`:'');
 const el=$('#v');el.src='/video/'+encodeURIComponent(v.name);el.playbackRate=1;el.play().catch(()=>{});
 fillForm(v.name);setStatus('');
 updateItem(old);STATE.videos.forEach((_,k)=>updateItem(k));
 const it=document.querySelector(`.item[data-i="${cur}"]`);if(it)it.scrollIntoView({block:'nearest'});
 $('#formCol').scrollTop=0;
}
async function init(){
 SCHEMA=await (await fetch('/api/schema')).json();
 STATE=await (await fetch('/api/state')).json();
 $('#rule').textContent=SCHEMA.dominant_rule;$('#ver').textContent='Kural sürümü: '+SCHEMA.rules_version;
 buildForm();renderList();updateProgress();
 const first=STATE.videos.findIndex(v=>!isDone(v.name));
 select(first<0?0:first);
}
document.addEventListener('keydown',e=>{
 const t=e.target.tagName;if(t==='TEXTAREA')return;
 if(e.key==='n'||e.key==='N'||e.key==='ArrowRight'){select(cur+1);}
 else if(e.key==='p'||e.key==='P'||e.key==='ArrowLeft'){select(cur-1);}
 else if(e.key===' '){e.preventDefault();const v=$('#v');v.paused?v.play():v.pause();}
});
$('#next').onclick=()=>select(cur+1);$('#prev').onclick=()=>select(cur-1);
$('#rewind').onclick=()=>{const v=$('#v');v.currentTime=0;v.play();};
$('#pp').onclick=()=>{const v=$('#v');v.paused?v.play():v.pause();};
document.querySelectorAll('[data-rate]').forEach(b=>b.onclick=()=>{$('#v').playbackRate=parseFloat(b.dataset.rate);});
$('#filter').onchange=()=>STATE.videos.forEach((_,k)=>updateItem(k));
$('#reset').onclick=()=>{if(!confirm('Bu videonun tüm etiketleri silinsin mi?'))return;
 STATE.labels[STATE.videos[cur].name]={};fillForm(STATE.videos[cur].name);save();};
init();
</script></body></html>
"""


class LabelStore:
    """Etiketleri bellekte tutar, her güncellemede CSV'yi atomik olarak yeniden yazar."""

    def __init__(self, path: Path, annotator: str):
        self.path, self.annotator = path, annotator
        self.lock = threading.Lock()
        self.rows = {}
        if path.exists():
            with open(path, newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    self.rows[r["video"]] = r

    def update(self, video, values):
        clean = {}
        for f in schema.FIELDS:
            v = str(values.get(f["id"], "")).replace("\r", " ").replace("\n", " ")[:500]
            allowed = schema.allowed_values(f)
            if allowed is not None and v != "" and v not in allowed:
                raise ValueError(f"Geçersiz değer: {f['id']}={v!r}")
            clean[f["id"]] = v
        row = {"video": video, "etiketleyici": self.annotator, **clean,
               "son_guncelleme": datetime.now().isoformat(timespec="seconds")}
        with self.lock:
            self.rows[video] = row
            self._write()

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=schema.CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for name in sorted(self.rows):
                w.writerow(self.rows[name])
        os.replace(tmp, self.path)


def natural_key(name):
    m = re.match(r"^(.*?)(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (name, -1)


def load_videos(proxy_dir: Path, inventory: Path):
    meta = {}
    if inventory.exists():
        inv = pd.read_csv(inventory)
        meta = {r.file: float(r.duration_s) for r in inv.itertuples() if pd.notna(r.duration_s)}
    names = sorted((p.name[:-4] for p in proxy_dir.glob("*.mp4") if not p.name.endswith(".tmp.mp4")),
                   key=natural_key)
    return [{"name": n, "duration": meta.get(n)} for n in names]


class Handler(BaseHTTPRequestHandler):
    store: LabelStore = None
    videos: list = []
    video_names: set = set()
    proxy_dir: Path = None

    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/favicon.ico":
            return self._send(204)
        if path == "/":
            return self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/api/schema":
            return self._json({"fields": schema.FIELDS, "dominant_rule": schema.DOMINANT_RULE,
                               "consistency": schema.CONSISTENCY, "rules_version": schema.RULES_VERSION})
        if path == "/api/state":
            labels = {n: {k: v for k, v in r.items() if k in {f["id"] for f in schema.FIELDS}}
                      for n, r in self.store.rows.items() if n in self.video_names}
            return self._json({"videos": self.videos, "labels": labels,
                               "annotator": self.store.annotator})
        if path.startswith("/video/"):
            return self._serve_video(unquote(path[len("/video/"):]))
        self._send(404, b"bulunamadi")

    def do_POST(self):
        if urlparse(self.path).path != "/api/label":
            return self._send(404, b"bulunamadi")
        try:
            n = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(n).decode("utf-8"))
            video = data["video"]
            if video not in self.video_names:
                raise ValueError("Bilinmeyen video")
            self.store.update(video, data.get("values", {}))
            self._json({"ok": True, "complete": schema.is_complete(self.store.rows[video])})
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            self._send(400, str(e).encode("utf-8"))

    def _serve_video(self, name):
        if name not in self.video_names:           # yol gezinmesini (../) engeller
            return self._send(404, b"bulunamadi")
        path = self.proxy_dir / f"{name}.mp4"
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            s, _, e = rng[6:].partition("-")
            try:
                if s:
                    start = int(s)
                    end = int(e) if e else size - 1
                elif e:
                    start = max(size - int(e), 0)
                code = 206
            except ValueError:
                pass
            end = min(end, size - 1)
            if start > end:
                return self._send(416, b"", extra={"Content-Range": f"bytes */{size}"})
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True, help="etiketleyici adı (ör. irem)")
    ap.add_argument("--stage", choices=["davranis", "duygu"], default="davranis",
                    help="davranis (varsayılan): davranış etiketleri; duygu: algılanan duygu etiketi (ayrı dosya)")
    ap.add_argument("--labeling-dir", default=Path("labeling"), type=Path)
    ap.add_argument("--inventory", default=Path("outputs/inventory.csv"), type=Path)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="tarayıcıyı otomatik açma")
    args = ap.parse_args()

    global schema
    if args.stage == "duygu":
        import schema_duygu
        schema = schema_duygu
    name = re.sub(r"[^A-Za-z0-9_-]", "_", args.annotator)
    prefix = "labels_duygu_" if args.stage == "duygu" else "labels_"
    proxy_dir = args.labeling_dir / "proxies"
    videos = load_videos(proxy_dir, args.inventory)
    if not videos:
        raise SystemExit("labeling/proxies içinde video yok. Önce 03_make_proxies.py çalıştırın.")

    Handler.store = LabelStore(args.labeling_dir / f"{prefix}{name}.csv", name)
    Handler.videos = videos
    Handler.video_names = {v["name"] for v in videos}
    Handler.proxy_dir = proxy_dir

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        raise SystemExit(f"{args.port} portu dolu. Başka bir port deneyin: --port 8001")
    url = f"http://127.0.0.1:{args.port}"
    print(f"{len(videos)} video yüklendi. Etiketler: {Handler.store.path}")
    print(f"Etiketleme sayfası: {url}   (durdurmak için Ctrl+C)")
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nKapatıldı. Etiketleriniz kaydedildi.")


if __name__ == "__main__":
    main()
