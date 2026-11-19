"""Build a human-only annotation workspace; validate imports without altering splits."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

LABELS={'aktivite':['hareketsiz','yuruyor','oynuyor_kosuyor','havliyor_hirliyor'],
        'vucut':['gevsek','gergin','cekingen_kapali'],
        'durus':['ayakta','yatiyor','oturuyor','sirtustu'],
        'kuyruk':['sallaniyor','notr','yukari','asagi_bacak_arasi','gorunmuyor'],
        'kulak':['notr','dik_one','geride_yassi','gorunmuyor']}


def manifest_digest(manifest):
    columns=['video_id','source_sha256','split',*LABELS]
    rows=manifest.fillna('').reindex(columns=columns).sort_values('video_id').to_dict('records')
    return hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def review_rows(manifest,report):
    samples={s['video_id']:s for s in report['samples']};rows=[]
    counts={t:manifest.loc[manifest.split=='train',t].value_counts().to_dict() for t in LABELS}
    for row in manifest.fillna('').to_dict('records'):
        if row['split']!='unlabelled':continue
        sample=samples[row['video_id']]
        if sample['source_sha256']!=row['source_sha256']:raise ValueError('Source identity mismatch')
        pred=sample['predictions']['cv_selected'];scores=sample['scores']['cv_selected'];uncertainty=[];rarity=[]
        for target in LABELS:
            p=np.array(list(scores[target].values()))
            entropy=float(-np.sum(p[p>0]*np.log(p[p>0]))/np.log(len(p))) if len(p)>1 else 0.
            uncertainty.append(entropy);rarity.append(max(0.,1-counts[target].get(pred[target],0)/10))
        rows.append({'video_id':row['video_id'],'source_sha256':row['source_sha256'],
            'source_filename':row.get('source_filename',row['video_id']),
            'labels':{t:row[t] for t in LABELS},'note':row.get('note',''),
            'status':'pending','priority':round(float(.5*np.mean(uncertainty)+.5*np.mean(rarity)),4),
            'suggestions':{t:{'label':pred[t],'score':scores[t][pred[t]],'training_examples':counts[t].get(pred[t],0)} for t in LABELS}})
    return sorted(rows,key=lambda row:(-row['priority'],row['video_id']))


def validate_review(manifest,payload):
    """Only explicit full-recording human reviews may update incomplete rows."""
    if not isinstance(payload,dict) or not isinstance(payload.get('records'),list):
        raise ValueError('Expected a review object with records')
    if manifest.video_id.duplicated().any():raise ValueError('Duplicate manifest video IDs')
    if payload.get('schema_version')!=1 or payload.get('manifest_digest')!=manifest_digest(manifest):
        raise ValueError('Review belongs to a different manifest version')
    original=manifest.fillna('').set_index('video_id',drop=False)
    result=original.copy();seen=set();accepted=[]
    for record in payload.get('records',[]):
        if not isinstance(record,dict):raise ValueError('Expected a review record object')
        name=record.get('video_id')
        if not isinstance(name,str):raise ValueError('Expected a video ID string')
        if name in seen:raise ValueError('Duplicate review video')
        seen.add(name)
        if name not in original.index or original.loc[name,'split']!='unlabelled':
            raise ValueError('Only original incomplete videos may be reviewed')
        if record.get('source_sha256')!=original.loc[name,'source_sha256']:raise ValueError('Video SHA256 mismatch')
        if record.get('status') not in {'pending','preview_only','blocked','reviewed_full'}:raise ValueError('Invalid review status')
        labels=record.get('labels',{})
        if not isinstance(labels,dict) or set(labels)!=set(LABELS):raise ValueError('Expected all five label fields')
        for target,value in labels.items():
            if not isinstance(value,str) or value and value not in LABELS[target]:raise ValueError('Invalid behavior label')
        if record['status']!='reviewed_full':continue
        if not all(labels.values()):raise ValueError('Full review requires all five labels')
        note=record.get('note','')
        if not isinstance(note,str):raise ValueError('Note must be text')
        for target,value in labels.items():result.loc[name,target]=value
        result.loc[name,'note']=note
        accepted.append(name)
    if not accepted:raise ValueError('No completed full-recording human reviews to import')
    assert result['split'].equals(original['split'])
    return result.reset_index(drop=True),accepted


def build(args):
    manifest=pd.read_csv(args.manifest).fillna('');report=json.loads(args.report.read_text())
    rows=review_rows(manifest,report)
    for row in rows:
        path=args.clips/(row['source_sha256']+'.mp4')
        if not path.is_file():raise FileNotFoundError(path)
        row['preview']='data:video/mp4;base64,'+base64.b64encode(path.read_bytes()).decode()
    payload={'schema_version':1,'manifest_digest':manifest_digest(manifest),'records':rows,'labels':LABELS}
    template=Path(__file__).with_name('annotation_review.html').read_text()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError('Choose a new output file')
    args.output.write_text(template.replace('/*REVIEW_DATA*/',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')))
    print(f'Review ready: {len(rows)} videos; {args.output}')


def apply(args):
    manifest=pd.read_csv(args.manifest).fillna('');payload=json.loads(args.review.read_text())
    updated,accepted=validate_review(manifest,payload)
    args.output.mkdir(parents=True,exist_ok=False)
    updated.to_csv(args.output/'manifest_reviewed.csv',index=False)
    updated[updated.video_id.isin(accepted)].to_csv(args.output/'human_reviewed_rows.csv',index=False)
    provenance={'accepted':accepted,'manifest_digest':manifest_digest(manifest),
                'review_sha256':hashlib.sha256(args.review.read_bytes()).hexdigest(),
                'split_policy':'Original train/test/unlabelled values preserved. No automatic retraining.',
                'labels_are':'Explicit human full-recording annotations; not model predictions.'}
    (args.output/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(f'Imported {len(accepted)} human reviews; existing train/test unchanged. Saved: {args.output}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    build_parser=commands.add_parser('build')
    for name in ['manifest','report','clips','output']:build_parser.add_argument('--'+name,type=Path,required=True)
    apply_parser=commands.add_parser('apply')
    for name in ['manifest','review','output']:apply_parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();build(args) if args.command=='build' else apply(args)
