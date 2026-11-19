"""Fetch pinned EmotionalCanines WAVs and verify Git LFS SHA256 objects."""
import argparse,concurrent.futures,hashlib,json,re,ssl,threading
from pathlib import Path
import certifi,requests

COMMIT='6b73c5fce55fc132e97db6a76899f8594a955f59'
REPO='tmdang1101/EmotionalCanines'
LOCAL=threading.local()


def session():
    if not hasattr(LOCAL,'http'):
        LOCAL.http=requests.Session();LOCAL.http.verify=certifi.where()
    return LOCAL.http


def fetch(url):
    r=session().get(url,timeout=(15,45));r.raise_for_status();return r.content


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    base=f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/'
    tree=json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1'))
    for name in ['LICENSE','README.md','husky_train_labels.csv','husky_test_labels.csv','shiba_train_labels.csv','shiba_test_labels.csv']:
        (output/name).write_bytes(fetch(base+name))
    files=[f['path'] for f in tree['tree'] if f['path'].endswith('.wav')]
    if len(files)!=1400:raise ValueError('Unexpected source inventory')
    def download(path):
        pointer=fetch(base+path).decode();oid=re.search(r'oid sha256:([0-9a-f]{64})',pointer).group(1)
        size=int(re.search(r'^size (\d+)',pointer,re.M).group(1));target=output/path
        if target.exists() and target.stat().st_size==size and hashlib.sha256(target.read_bytes()).hexdigest()==oid:
            return dict(path=path,sha256=oid,bytes=size)
        if size>5_000_000:raise ValueError('Unexpected audio size')
        for retry in range(3):
            try:
                data=fetch(f'https://media.githubusercontent.com/media/{REPO}/{COMMIT}/{path}')
                if len(data)!=size or hashlib.sha256(data).hexdigest()!=oid:raise ValueError('LFS integrity mismatch: '+path)
                if data[:4] not in [b'RIFF',b'RF64']:raise ValueError('Not WAV')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
                return dict(path=path,sha256=oid,bytes=size)
            except (requests.RequestException,ValueError):
                if retry==2:raise
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for record in pool.map(download,files):
            records.append(record)
            if len(records)%50==0:print(f'Verified external audio {len(records)}/1400',flush=True)
            (output/'files.json').write_text(json.dumps(records))
    provenance=dict(repository=REPO,commit=COMMIT,files=records,license='CC-BY-NC-SA-4.0',
                    source='Human-perceived three-class arousal and valence, Husky/Shiba; not our behavior-rule emotion CSV')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2));print('Dataset ready',len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/affect_v15_external/data'));prepare(p.parse_args().output)
