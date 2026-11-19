"""Local report/presentation collection, independent of generated source pages."""
import json
import re
import threading
import uuid
from pathlib import Path

MAX_DOCUMENT_BYTES=50*1024*1024
ALLOWED={'.pdf','.docx','.pptx','.txt','.md'}


class Documents:
    def __init__(self,root,project):
        self.root=Path(root);self.project=Path(project);self.lock=threading.Lock()
        self.root.mkdir(parents=True,exist_ok=True)
        self.index=self.root/'index.json'

    def read(self):
        if self.index.exists():return json.loads(self.index.read_text())
        return [{'id':'builtin-report','category':'report','name':'Hazır yöntem ve sonuç raporu','builtin':True},
                {'id':'builtin-slides','category':'slides','name':'Hazır proje sunumu','builtin':True}]

    def write(self,items):
        temp=self.index.with_suffix('.tmp');temp.write_text(json.dumps(items,ensure_ascii=False));temp.replace(self.index)

    def list(self,category):
        with self.lock:return [r for r in self.read() if r['category']==category]

    def add(self,category,name,raw):
        name=name.replace('\\','/').split('/')[-1].strip()
        if category not in {'report','slides'} or not name or len(name)>180 or '\x00' in name or Path(name).suffix.lower() not in ALLOWED:
            raise ValueError('PDF, DOCX, PPTX, TXT veya MD dosyası seç.')
        if not raw or len(raw)>MAX_DOCUMENT_BYTES:raise ValueError('Dosya boş veya 50 MB sınırını aşıyor.')
        if Path(name).suffix.lower()=='.pdf' and not raw.startswith(b'%PDF-'):raise ValueError('Geçerli bir PDF seç.')
        if Path(name).suffix.lower() in {'.docx','.pptx'} and not raw.startswith(b'PK'):raise ValueError('Geçerli bir Office dosyası seç.')
        record=dict(id=uuid.uuid4().hex,category=category,name=name,builtin=False)
        with self.lock:
            items=self.read();target=self.root/record['id'];target.write_bytes(raw)
            try:self.write(items+[record])
            except Exception:target.unlink();raise
        return record

    def file(self,category,key):
        if not re.fullmatch(r'(?:[a-f0-9]{32}|builtin-report|builtin-slides)',key):raise KeyError(key)
        with self.lock:record=next((r for r in self.read() if r['category']==category and r['id']==key),None)
        if record is None:raise KeyError(key)
        if record.get('builtin'):
            path=self.project/('DEGERLENDIRME.html' if key=='builtin-report' else 'docs/hoca_sunumu.html')
        else:path=self.root/key
        if not path.is_file():raise KeyError(key)
        return record,path

    def delete(self,category,key):
        with self.lock:
            items=self.read();record=next((r for r in items if r['category']==category and r['id']==key),None)
            if record is None:raise KeyError(key)
            self.write([r for r in items if r is not record])
            if not record.get('builtin'):(self.root/key).unlink(missing_ok=True)
