"""Apply the frozen neural fusion head to experimental short-window features."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch
from src.audio_analysis.yamnet import Yamnet,logmel_patches
from src.audio_analysis.run import extract as decode_audio
from src.audio_analysis.behavior import event_features
from src.demo.temporal import visual_window
from src.demo.extract import digest
from .train import FusionNet,normalize_apply,TARGETS


def main(args):
    torch.set_num_threads(2)
    report=json.loads(args.report.read_text());meta=json.loads(args.model.with_suffix('.json').read_text())
    model=FusionNet(meta['visual_dim'],meta['audio_dim'],meta['class_counts'],'fusion')
    model.load_state_dict(torch.load(args.model,weights_only=True,map_location='cpu'));model.eval()
    audio_model=Yamnet(args.weights);manifest=pd.read_csv(args.manifest).set_index('video_id')
    args.output.mkdir(parents=True,exist_ok=False)
    for index,sample in enumerate(report['samples'],1):
        key=sample['source_sha256'];row=manifest.loc[sample['video_id']]
        roots=[Path('results/demo_v1/features'),Path('results/demo_v4/new_features')]
        root=next(p for p in roots if (p/(key+'.pose.jsonl')).exists())
        records=[json.loads(line) for line in (root/(key+'.pose.jsonl')).read_text().splitlines()]
        cap=cv2.VideoCapture(row.video_path);width=cap.get(cv2.CAP_PROP_FRAME_WIDTH);height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT);cap.release()
        wave=decode_audio(row.video_path);embeddings=None
        if wave is not None:
            _,patches=logmel_patches(wave[:960000]);chunks=[]
            with torch.inference_mode():
                for start in range(0,len(patches),32):
                    _,values=audio_model(torch.from_numpy(patches[start:start+32]));chunks.append(values.numpy())
            embeddings=np.concatenate(chunks)
        windows=[]
        for old in sample['temporal_behavior']['windows']:
            start,end=old['start'],old['end'];visual,quality=visual_window(records,start,end,width,height)
            patch_ids=[i for i in range(len(embeddings)) if i*.48>=start and i*.48+.975<=end] if embeddings is not None else []
            audio_windows=[w for w in sample['audio_analysis']['windows'] if w['start']>=start and w['end']<=end]
            events=event_features({'status':sample['audio_analysis']['status'],'windows':audio_windows},seconds=end)
            item={'start':start,'end':end,'quality':quality,'predictions':{},'scores':{},'unavailable':{}}
            if visual['v_pose_available'] and patch_ids and events['y_available']:
                v=np.array([[visual[c] for c in meta['protocol']['visual_columns']]])
                a=np.concatenate([embeddings[patch_ids].mean(0),[events[c] for c in meta['protocol']['event_columns']]])[None]
                with torch.inference_mode():outputs=model(normalize_apply(v,meta['visual_scaler']),normalize_apply(a,meta['audio_scaler']))
                for t,output in zip(TARGETS,outputs):
                    probabilities=torch.softmax(output,dim=1)[0].numpy();classes=meta['protocol']['classes'][t]
                    item['predictions'][t]=classes[int(probabilities.argmax())]
                    item['scores'][t]=dict(zip(classes,map(float,probabilities)))
            else:item['unavailable']={t:'pose_or_audio_window_missing' for t in TARGETS}
            windows.append(item)
        sample['neural_temporal_behavior']={**{k:v for k,v in sample['temporal_behavior'].items() if k!='windows'},
            'windows':windows,'model':'neural_fusion','trained_on':'video-level labels, frozen encoders',
            'model_sha256':digest(args.model)}
        print(f'Neural windows {index}/{len(report["samples"])}: {sample["video_id"]}',flush=True)
    report['temporal_protocol']['neural_model']='Train-CV epoch-selected neural fusion, 4s windows; no temporal ground truth'
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['report','model','weights','manifest','output']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
