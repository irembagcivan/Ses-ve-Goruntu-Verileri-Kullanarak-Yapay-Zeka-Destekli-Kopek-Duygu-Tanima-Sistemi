"""Read embedded recording timestamps, never filesystem dates or filename guesses."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess

import imageio_ffmpeg
import pandas as pd


def parse_tags(stderr):
    # Ignore output tags: ffmpeg copies input metadata into its null output.
    before_output=stderr.split('Output #',1)[0]
    def first(key):
        match=re.search(r'^\s*'+re.escape(key)+r'\s*:\s*(\S+)\s*$',before_output,re.M)
        return match.group(1) if match else ''
    camera=first('com.apple.quicktime.creationdate');container=first('creation_time')
    selected=camera or container
    day='';status='missing_timestamp'
    if selected:
        try:
            stamp=datetime.fromisoformat(selected.replace('Z','+00:00'))
            if stamp.tzinfo is None or stamp.year<2000:
                raise ValueError('Unusable embedded timestamp')
            day=stamp.date().isoformat()
            status='camera_creationdate' if camera else 'container_creation_time_only'
        except ValueError:status='invalid_timestamp'
    disagree=False
    if camera and container:
        try:
            c=datetime.fromisoformat(camera.replace('Z','+00:00'))
            f=datetime.fromisoformat(container.replace('Z','+00:00'))
            if c.tzinfo is not None and f.tzinfo is not None:
                disagree=c.date()!=f.astimezone(c.tzinfo).date()
        except ValueError:pass
    return dict(camera_timestamp=camera,container_timestamp=container,recording_day_candidate=day,
                evidence=status,container_camera_day_disagree=disagree)


def audit(manifest,output):
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for row in pd.read_csv(manifest).itertuples():
        p=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-i',row.video_path,
                          '-map','0:v:0','-frames:v','0','-f','null','-'],capture_output=True,text=True,timeout=30)
        if p.returncode:raise ValueError('Metadata read failed: '+row.video_id)
        rows.append(dict(video_id=row.video_id,source_sha256=row.source_sha256,**parse_tags(p.stderr),
                         confirmed_session_id='',annotator_confirmed=False))
    pd.DataFrame(rows).to_csv(output/'recording_days.csv',index=False)
    summary=dict(videos=len(rows),day_counts=dict(Counter(r['recording_day_candidate'] or 'UNKNOWN' for r in rows)),
                 evidence_counts=dict(Counter(r['evidence'] for r in rows)),
                 metadata_conflicts=sum(r['container_camera_day_disagree'] for r in rows),
                 confirmed_sessions=0,
                 scope='Embedded metadata day candidates, not verified filming sessions or dog identities. Camera-local day preferred; container UTC day is fallback. Unknown dates never assigned from file names/mtime.')
    (output/'audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=Path('results/demo_v4/manifest.csv'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.manifest,a.output)
