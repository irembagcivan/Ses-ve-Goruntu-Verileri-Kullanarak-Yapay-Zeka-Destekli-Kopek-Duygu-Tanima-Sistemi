"""Bounded CPU rendering of disjoint gallery partitions, with per-video resume."""
import argparse
import json
from pathlib import Path
import time


def run(args):
    import cv2
    import pandas as pd
    import torch
    from src.demo import dense_previews
    from src.demo.extract import digest
    from src.demo.robust_detection import RobustDogDetector, OrientedPoseAdapter
    from src.pose_estimation.backend import DeepLabCutRTMPose
    from src.pose_estimation.adapter import DogPoseAdapter
    # OpenCV's default eight threads per process oversubscribes this Mac when
    # rendering two independent videos. Limit both libraries explicitly.
    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    report = json.loads((args.results/'report.json').read_text())
    table = pd.read_csv(args.manifest).set_index('video_id')
    destination = args.results/'public/clips_v4'
    destination.mkdir(parents=True,exist_ok=True)
    detector = RobustDogDetector(args.detector,args.rescue)
    adapter = OrientedPoseAdapter(DogPoseAdapter(DeepLabCutRTMPose(args.pose,device='cpu')))
    source_dir = Path(dense_previews.__file__).parent
    signature = {'version':3,'fps':args.fps,'detector':digest(args.detector),'pose':digest(args.pose),
                 'renderer':digest(dense_previews.__file__),'tracking':digest(source_dir/'tracking.py'),
                 'stable':True,'rescue':digest(args.rescue),'stabilizer':digest(source_dir/'stable_pose.py'),
                 'robust_detector':digest(source_dir/'robust_detection.py')}
    signature['pose_flow']=digest(source_dir/'pose_flow.py')
    samples = report['samples'][args.worker_index::args.worker_count]
    outcomes = []
    for number,sample in enumerate(samples,1):
        start = time.monotonic()
        key = sample['source_sha256']
        clip,metadata = destination/f'{key}.mp4', destination/f'{key}.json'
        try:
            if clip.exists() and metadata.exists() and json.loads(metadata.read_text()).get('signature')==signature:
                status = 'resumed'
                result = json.loads(metadata.read_text())
            else:
                seconds=12 if sample.get('split')=='test' else 6
                result = dense_previews.render(Path(table.loc[sample['video_id'],'video_path']),clip,
                                                detector,adapter,args.fps,seconds=seconds,stable=True)
                result.update(signature=signature,keypoint_confidence=.45,point_entry_confidence=.58,
                              edge_confidence=.58,seconds_limit=seconds,
                              runtime_threads={'torch':args.threads,'opencv':1})
                metadata.write_text(json.dumps(result,indent=2))
                status = 'complete'
            outcomes.append({'video_id':sample['video_id'],'status':status})
            print(f"[{number}/{len(samples)}] {sample['video_id']}: {status}; {result['frames']} frames; {time.monotonic()-start:.1f}s",flush=True)
        except Exception as exc:
            outcomes.append({'video_id':sample['video_id'],'status':'failed','error':str(exc)})
            print(f"FAILED {sample['video_id']}: {exc}",flush=True)
        (args.results/f'render_worker_{args.worker_index}.json').write_text(json.dumps(outcomes,indent=2))
    if any(row['status']=='failed' for row in outcomes):
        raise SystemExit(2)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','results','detector','rescue','pose']:
        parser.add_argument(f'--{name}',required=True,type=Path)
    parser.add_argument('--fps',type=float,default=12)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--worker-index',type=int,default=0)
    parser.add_argument('--worker-count',type=int,default=1)
    args = parser.parse_args()
    if not 1<=args.fps<=30 or not 1<=args.threads<=8 or not 0<=args.worker_index<args.worker_count:
        parser.error('Invalid FPS, thread count or worker partition')
    run(args)
