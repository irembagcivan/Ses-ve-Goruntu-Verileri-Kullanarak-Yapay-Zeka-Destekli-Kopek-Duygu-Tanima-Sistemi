"""Explicit higher-recall rendering for a user-selected difficult crowd video."""
import argparse
import json
from pathlib import Path


def run(args):
    import cv2
    import torch
    import pandas as pd
    from src.demo import dense_previews
    from src.demo.extract import digest
    from src.preprocessing.detection import YoloDogDetector
    from src.preprocessing.interfaces import DogDetection
    from src.pose_estimation.backend import DeepLabCutRTMPose
    from src.pose_estimation.adapter import DogPoseAdapter
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    class CrowdDetector(YoloDogDetector):
        def detect(self, frame):
            results=self.model.predict(frame.pixels,classes=[self.dog_id],conf=.25,imgsz=960,
                                       device='cpu',verbose=False,save=False)
            return [DogDetection(tuple(map(float,box)),float(score)) for result in results
                    for box,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist())]
    table=pd.read_csv(args.manifest).set_index('video_id')
    row=table.loc[args.video_id]
    detector=CrowdDetector(args.detector,'cpu',.25)
    adapter=DogPoseAdapter(DeepLabCutRTMPose(args.pose,device='cpu'))
    root=args.results/'public/clips_v4'
    root.mkdir(parents=True,exist_ok=True)
    clip=root/f'{row.source_sha256}.mp4'
    result=dense_previews.render(Path(row.video_path),clip,detector,adapter,args.fps,stable=True)
    source=Path(dense_previews.__file__).parent
    result.update(keypoint_confidence=.45,point_entry_confidence=.58,edge_confidence=.58,
                  detector_mode='explicit_high_recall_crowd_960',detector_confidence=.25,
                  signature={'renderer':digest(dense_previews.__file__),'crowd_renderer':digest(__file__),
                             'detector':digest(args.detector),'pose':digest(args.pose),
                             'stabilizer':digest(source/'stable_pose.py'),'pose_flow':digest(source/'pose_flow.py')})
    clip.with_suffix('.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'video_id':args.video_id,'frames':result['frames'],'max_dogs':result['max_dogs'],
                      'multi_dog_frames':result['multi_dog_frames']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','results','detector','pose']:
        parser.add_argument(f'--{name}',required=True,type=Path)
    parser.add_argument('--video-id',required=True)
    parser.add_argument('--fps',type=float,default=12)
    args=parser.parse_args()
    if not 1<=args.fps<=30:
        parser.error('FPS must be between 1 and 30')
    run(args)
