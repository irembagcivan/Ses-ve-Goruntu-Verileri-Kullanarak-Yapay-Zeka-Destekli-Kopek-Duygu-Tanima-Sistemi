"""Cohen kappa for two independent raw annotation forms, never consensus/rules."""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from src.deep_learning.final_labels import TARGETS,CLASSES


def agreement(left,right):
    for table in [left,right]:
        if {'kural','kaynak','konsensus_disi'}&set(table):
            raise ValueError('Independent raw rater forms required; consensus/rule-derived tables rejected')
        if {'video',*TARGETS}-set(table):raise ValueError('Missing raw rater columns')
    left=left.copy().fillna('');right=right.copy().fillna('')
    for table in [left,right]:
        table['key']=table.video.str.strip().str.casefold()
        if table.key.duplicated().any():raise ValueError('Duplicate rater video')
    merged=left.merge(right,on='key',suffixes=('_a','_b'),validate='one_to_one')
    results=[]
    for t in TARGETS:
        a=merged[t+'_a'];b=merged[t+'_b']
        for s in [a,b]:
            if not set(s)<=set(CLASSES[t])|{'','gorunmuyor'}:raise ValueError('Unknown rater category')
        valid=a.isin(CLASSES[t])&b.isin(CLASSES[t]);a=a[valid];b=b[valid]
        undefined=not len(a) or len(set(a)|set(b))<2
        results.append(dict(target=t,paired_labels=len(a),raw_agreement=float((a==b).mean()) if len(a) else None,
                            cohen_kappa=None if undefined else float(cohen_kappa_score(a,b)),
                            undefined_reason='no_pairs_or_single_combined_category' if undefined else None))
    return dict(left_videos=len(left),right_videos=len(right),matched_videos=len(merged),
                unmatched_left=sorted(set(left.key)-set(right.key)),unmatched_right=sorted(set(right.key)-set(left.key)),
                targets=results,scope='Unweighted nominal Cohen kappa; paired observed cells only. Raw forms must be independent, pre-consensus judgments.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['left','right','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.left.resolve()==a.right.resolve():raise ValueError('Cannot compare a form with itself')
    if a.output.exists():raise FileExistsError('Choose a new output path')
    r=agreement(pd.read_csv(a.left).fillna(''),pd.read_csv(a.right).fillna(''));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2))
