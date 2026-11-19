"""Infer five behaviors for a new raw video with the trained neural head."""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from src.demo.extract import visual_features,digest,clean_json
from src.audio_analysis.yamnet import Yamnet,logmel_patches
from src.audio_analysis.run import extract as decode_audio,summarize
from src.audio_analysis.behavior import event_features
from .train import FusionNet,normalize_apply,model_targets


def main(args):
    if args.output.exists():raise FileExistsError('Choose a new output file')
    torch.set_num_threads(2);meta=json.loads(args.model.with_suffix('.json').read_text());protocol=meta['protocol'];signature=protocol['visual_feature_extraction_signature']
    if digest(args.weights)!=protocol['yamnet_weights_sha256']:raise ValueError('YAMNet weights differ from training')
    if digest(args.detector)!=signature['detector_sha256'] or digest(args.pose)!=signature['pose_sha256']:raise ValueError('Visual weights differ from training')
    from src.preprocessing.detection import YoloDogDetector
    from src.pose_estimation.backend import DeepLabCutRTMPose
    from src.pose_estimation.adapter import DogPoseAdapter
    adapter=DogPoseAdapter(DeepLabCutRTMPose(args.pose,device='cpu'))
    visual,quality,_=visual_features(args.video,YoloDogDetector(args.detector,'cpu',.25),adapter,signature['sample_fps'],signature['seconds'],signature['confidence'])
    wave=decode_audio(args.video)
    if wave is None:raise ValueError('No readable audio: this trained fusion model requires both modalities')
    if not visual['v_pose_available']:raise ValueError('No usable pose: cannot infer fusion behavior')
    wave=wave[:960000];_,patches=logmel_patches(wave);audio_model=Yamnet(args.weights);scores=[];embeddings=[]
    with torch.inference_mode():
        for start in range(0,len(patches),32):
            a,b=audio_model(torch.from_numpy(patches[start:start+32]));scores.append(a.numpy());embeddings.append(b.numpy())
    names=[row['display_name'] for row in csv.DictReader(open('configs/audio/yamnet_class_map.csv'))]
    events=event_features(summarize(wave,np.concatenate(scores),names));v=np.array([[visual[c] for c in protocol['visual_columns']]])
    a=np.concatenate([np.concatenate(embeddings).mean(0),[events[c] for c in protocol['event_columns']]])[None]
    model=FusionNet(meta['visual_dim'],meta['audio_dim'],meta['class_counts'],meta['modality']);model.load_state_dict(torch.load(args.model,weights_only=True,map_location='cpu'));model.eval()
    with torch.inference_mode():outputs=model(normalize_apply(v,meta['visual_scaler']),normalize_apply(a,meta['audio_scaler']))
    predictions={};probabilities={}
    for target,output in zip(model_targets(meta),outputs):
        score=torch.softmax(output,dim=1)[0].numpy();classes=protocol['classes'][target]
        predictions[target]=classes[int(score.argmax())];probabilities[target]=dict(zip(classes,map(float,score)))
    result={'video':args.video.name,'predictions':predictions,'scores':probabilities,'quality':quality,
            'scope':'First 60 seconds, video-level, largest dog box per frame; no identity guarantee',
            'scores_calibrated':False,'emotion_model':False,'model_sha256':digest(args.model)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(clean_json(result),ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps(predictions,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['video','model','weights','detector','pose','output']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
