"""Analyze a copied user video locally; save playback, sound events and neural outputs."""
import os
# Avoid OpenMP CPU-affinity shared-memory probing in detached local workers.
os.environ.setdefault('KMP_AFFINITY','disabled')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,csv,json,subprocess
from pathlib import Path
import cv2,imageio_ffmpeg,numpy as np,torch
from src.deep_learning.train import FusionNet,normalize_apply,model_targets
from src.audio_analysis.yamnet import Yamnet,logmel_patches
from src.audio_analysis.run import extract as decode_audio,summarize
from src.audio_analysis.behavior import event_features
from src.demo.extract import visual_features,digest,clean_json
from src.demo.temporal import visual_window,intervals

def progress(output,stage):
    temporary=output/'progress.tmp';temporary.write_text(json.dumps({'stage':stage},ensure_ascii=False));temporary.replace(output/'progress.json')

def infer(meta,model,visual,audio):
    v=np.array([[visual[c] for c in meta['protocol']['visual_columns']]])
    a=audio[None]
    with torch.inference_mode():outputs=model(normalize_apply(v,meta['visual_scaler']),normalize_apply(a,meta['audio_scaler']))
    pred={};scores={}
    for target,output in zip(model_targets(meta),outputs):
        p=torch.softmax(output,1)[0].numpy();classes=meta['protocol']['classes'][target]
        pred[target]=classes[int(p.argmax())];scores[target]=dict(zip(classes,map(float,p)))
    return {'predictions':pred,'scores':scores}

def main(args):
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    meta=json.loads(args.model.with_suffix('.json').read_text());signature=meta['protocol']['visual_feature_extraction_signature']
    if digest(args.detector)!=signature['detector_sha256'] or digest(args.pose)!=signature['pose_sha256'] or digest(args.weights)!=meta['protocol']['yamnet_weights_sha256']:raise ValueError('Model checkpoints differ from training')
    cap=cv2.VideoCapture(str(args.video))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);frames=cap.get(cv2.CAP_PROP_FRAME_COUNT);width=cap.get(cv2.CAP_PROP_FRAME_WIDTH);height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not cap.isOpened() or not np.isfinite(fps) or fps<=0 or frames<=0 or width*height>33000000:raise ValueError('Unsupported or unreadable video')
        duration=frames/fps
    finally:cap.release()
    progress(args.output,'Oynatılabilir video hazırlanıyor')
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(args.video),'-t','60','-vf',"scale=w='trunc(min(960,iw)/2)*2':h=-2",'-c:v','libx264','-preset','veryfast','-crf','25','-c:a','aac','-movflags','+faststart',str(args.output/'video.mp4')]
    subprocess.run(command,check=True,capture_output=True,timeout=180)
    progress(args.output,'Köpek tespiti ve poz özellikleri çıkarılıyor')
    from src.preprocessing.detection import YoloDogDetector
    from src.pose_estimation.backend import DeepLabCutRTMPose
    from src.pose_estimation.adapter import DogPoseAdapter
    adapter=DogPoseAdapter(DeepLabCutRTMPose(args.pose,device='cpu'))
    visual,quality,records=visual_features(args.video,YoloDogDetector(args.detector,'cpu',.25),adapter,2,60,.3)
    valid_pose=any(np.isfinite(value) for key,value in visual.items() if key.startswith('v_pose_') and key!='v_pose_available');visual['v_pose_available']=int(valid_pose)
    progress(args.output,'Ses olayları ve YAMNet temsilleri çıkarılıyor')
    wave=decode_audio(args.video);embeddings=None;audio_analysis={'status':'no_audio'}
    if wave is not None:
        wave=wave[:960000];_,patches=logmel_patches(wave);yamnet=Yamnet(args.weights);scores=[];vectors=[]
        with torch.inference_mode():
            for start in range(0,len(patches),32):
                a,b=yamnet(torch.from_numpy(patches[start:start+32]));scores.append(a.numpy());vectors.append(b.numpy())
        embeddings=np.concatenate(vectors);names=[r['display_name'] for r in csv.DictReader(open('configs/audio/yamnet_class_map.csv'))];audio_analysis=summarize(wave,np.concatenate(scores),names)
    events=event_features(audio_analysis)
    audio=np.concatenate([embeddings.mean(0) if embeddings is not None else np.full(1024,np.nan),[events[c] for c in meta['protocol']['event_columns']]])
    model=FusionNet(meta['visual_dim'],meta['audio_dim'],meta['class_counts'],'fusion');model.load_state_dict(torch.load(args.model,weights_only=True,map_location='cpu'));model.eval()
    analysis={'schema_version':2,'annotation_version':meta.get('annotation_version','legacy-v12'),'targets':model_targets(meta),'source_duration':duration,'analyzed_until':min(60,duration),'quality':quality,
              'audio_analysis':audio_analysis,'behavior':None,'windows':[],
              'model':'FusionNet, frozen YOLOv8n/RTMPose/YAMNet features','emotion_model':False,
              'warnings':['İlk 60 saniye analiz edilir. Skorlar kalibre edilmiş olasılık değildir.','Çoklu köpekte kutu seçimi birey değiştirebilir; ses köpeğe atanamaz.','Kısa pencere davranış doğruluğu ölçülmedi.']}
    if valid_pose and embeddings is not None:analysis['behavior']=infer(meta,model,visual,audio)
    elif not valid_pose:analysis['warnings'].append('Yeterli köpek pozu bulunamadı; davranış sonucu üretilmedi.')
    else:analysis['warnings'].append('Ses kanalı yok; birleşik davranış sonucu üretilmedi. Görüntü ve ses ayrı raporlanır.')
    for start,end in intervals(min(60,duration)):
        v,q=visual_window(records,start,end,width,height);indices=[i for i in range(len(embeddings)) if i*.48>=start and i*.48+.975<=end] if embeddings is not None else []
        local_events=event_features({'status':audio_analysis['status'],'windows':[w for w in audio_analysis.get('windows',[]) if w['start']>=start and w['end']<=end]},seconds=end)
        value=None
        if v['v_pose_available'] and indices:
            a=np.concatenate([embeddings[indices].mean(0),[local_events[c] for c in meta['protocol']['event_columns']]])
            value=infer(meta,model,v,a)
        analysis['windows'].append({'start':start,'end':end,'quality':q,'behavior':value})
    analysis['model_comparisons']={'FusionNet':analysis['behavior']}
    analysis['preferred_model']='FusionNet'
    # A missing modality must not suppress the available single-modality model.
    for modality,available,title in [('visual',valid_pose,'Sinir ağı · görüntü'),('audio',embeddings is not None,'Sinir ağı · ses')]:
        path=args.model.parent/('neural_'+modality+'.pt')
        if not available or not path.exists():continue
        single_meta=json.loads(path.with_suffix('.json').read_text())
        if model_targets(single_meta)!=model_targets(meta):raise ValueError('Single-modality task schema mismatch')
        single=FusionNet(single_meta['visual_dim'],single_meta['audio_dim'],single_meta['class_counts'],modality)
        single.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'));single.eval()
        analysis['model_comparisons'][title]=infer(single_meta,single,visual,audio)
        if analysis['behavior'] is None:analysis['preferred_model']=title
    if valid_pose and embeddings is not None and meta.get('annotation_version')=='labels_final-v13':
        import joblib
        classical_path=args.model.parent/'classical.joblib'
        if classical_path.exists():
            bundle=joblib.load(classical_path)
            visual_array=np.array([[visual[c] for c in meta['protocol']['visual_columns']]],float)
            for name,data in [('visual',visual_array),('audio',audio[None]),('fusion',np.concatenate([visual_array,audio[None]],1))]:
                predictions={};values={}
                for t,estimator in zip(bundle['targets'],bundle['models'][name]):
                    p=np.zeros(len(bundle['classes'][t]));p[estimator.classes_.astype(int)]=estimator.predict_proba(data)[0]
                    predictions[t]=bundle['classes'][t][int(p.argmax())];values[t]=dict(zip(bundle['classes'][t],map(float,p)))
                analysis['model_comparisons'][{'visual':'Klasik · görüntü','audio':'Klasik · ses','fusion':'Klasik · birleşik'}[name]]={'predictions':predictions,'scores':values}
            selection=json.loads((args.model.parent/'selection.json').read_text())
            analysis['selected_by_training_cv']=selection['selected_by_training_cv']
            analysis['preferred_model']={'classical_visual':'Klasik · görüntü','classical_audio':'Klasik · ses','classical_fusion':'Klasik · birleşik','neural_fusion':'FusionNet','sequence_cnn':'CNN1D','sequence_gru':'GRU','sequence_bigru':'BiGRU','sequence_lstm_lstm_gru':'LSTM–LSTM–GRU'}.get(selection['selected_by_training_cv'],'FusionNet')
        analysis['warnings'].append('Son etiket sürümü: altı hedef; eski vücut etiketi kaldırıldı. Ses etiketi odak köpeğe ait olabilir; kayıt içindeki sesler bireye ayrılmaz. Hırlama eğitimde yalnız bir örnektir.')
    if valid_pose and embeddings is not None:
        from src.deep_learning.sequence import SequenceNet,sequence_from_parts,scale_apply
        length=min(60,max(1,int(np.ceil(len(wave)/16000))))
        seq=sequence_from_parts(records,embeddings,length)
        for kind in ['cnn','gru','bigru','lstm_lstm_gru']:
            path=(args.model.parent if meta.get('annotation_version')=='labels_final-v13' else Path('results/demo_v11'))/('sequence_'+kind+'.pt')
            if not path.exists():continue
            metadata=json.loads(path.with_suffix('.json').read_text())
            if kind=='lstm_lstm_gru':
                from src.deep_learning.paper_hybrid import HybridNet
                net=HybridNet(metadata['sizes'],kind)
            else:net=SequenceNet(metadata['sizes'],kind)
            net.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'));net.eval()
            with torch.inference_mode():outputs=net(scale_apply(seq[None],np.array([length]),metadata['scaler']),torch.tensor([length]))
            prediction={};values={}
            for t,output in zip(model_targets(metadata),outputs):
                p=torch.softmax(output,1)[0].numpy();classes=metadata['classes'][t];prediction[t]=classes[int(p.argmax())];values[t]=dict(zip(classes,map(float,p)))
            analysis['model_comparisons'][{'cnn':'CNN1D','gru':'GRU','bigru':'BiGRU','lstm_lstm_gru':'LSTM–LSTM–GRU'}[kind]]={'predictions':prediction,'scores':values}
    # Actual image evidence for the most populated sampled frame, not a fabricated skeleton.
    best=max(records,key=lambda r:(r['dog_count'],sum(float(s or 0)>=.3 for s in r.get('confidence',[]))))
    cap=cv2.VideoCapture(str(args.video));cap.set(cv2.CAP_PROP_POS_FRAMES,best['frame_index']);ok,pixels=cap.read();cap.release()
    if ok:
        from src.preprocessing.interfaces import VideoFrame
        rescue=Path('models/yolov8m.pt')
        detector=YoloDogDetector(rescue if rescue.exists() else args.detector,'cpu',.3)
        frame=VideoFrame(best['frame_index'],best['timestamp_seconds'],pixels)
        dogs=detector.detect(frame);poses=adapter.estimate(frame,dogs) if dogs else []
        evidence_dogs=[]
        for index,(dog,pose) in enumerate(zip(dogs,poses),1):
            x1,y1,x2,y2=map(int,dog.xyxy);cv2.rectangle(pixels,(x1,y1),(x2,y2),(132,239,182),2)
            cv2.putText(pixels,'Dog '+str(index),(max(0,x1),max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.65,(132,239,182),2)
            count=0
            for (x,y),confidence,name in zip(pose.xy,pose.confidence,pose.keypoint_names):
                valid=np.isfinite([x,y,confidence]).all() and confidence>=.45 and 'antler' not in name and 0<=x<width and 0<=y<height and x1-.05*(x2-x1)<=x<=x2+.05*(x2-x1) and y1-.05*(y2-y1)<=y<=y2+.05*(y2-y1)
                if valid:cv2.circle(pixels,(int(x),int(y)),3,(132,239,182),-1);count+=1
            evidence_dogs.append({'frame_local_id':index,'bbox_xyxy':list(dog.xyxy),'detection_score':dog.confidence,'visible_keypoints':count})
        cv2.imwrite(str(args.output/'evidence.jpg'),pixels)
        analysis['evidence']={'timestamp':best['timestamp_seconds'],'detected_dogs':len(dogs),'dogs':evidence_dogs,'detector':'YOLOv8m snapshot' if rescue.exists() else 'YOLOv8n snapshot','scope':'All detected dogs in one sampled frame; IDs are frame-local, not persistent identities'}
    progress(args.output,'Gerçek köpek maskeleri ve segmentasyon videosu hazırlanıyor')
    from src.demo.segmentation import render as render_segmentation
    segmentation=render_segmentation(args.video,args.output/'segmentation',args.segmenter,seconds=min(60,duration))
    analysis['segmentation']={key:value for key,value in segmentation.items() if key!='records'}
    if not segmentation['frames_with_dogs']:analysis['warnings'].append('Segmentasyon modeli örneklenen karelerde köpek maskesi bulamadı.')
    analysis['warnings'].append('Segmentasyon ön-eğitimli model çıktısıdır; maske doğruluğu ölçülmedi. Davranış modeli bu maskelerle yeniden eğitilmedi.')
    # One recommended behavior decision; members are selected on training CV.
    fusion_path=Path('results/late_fusion_v15/deployment.json')
    if fusion_path.exists():
        from src.evaluation.late_fusion import blend
        choice=json.loads(fusion_path.read_text());members={}
        for key in choice['weights']:
            item=analysis['model_comparisons'].get(choice['components'][key])
            if item is not None:members[key]=[np.array([[item['scores'][t][c] for c in choice['classes'][t]]]) for t in choice['targets']]
        if set(members)==set(choice['weights']):
            if any(digest(args.model.parent/name)!=fingerprint for name,fingerprint in choice['model_hashes'].items()):raise ValueError('Fusion members changed')
            p=blend(members,choice['weights']);prediction={};values={}
            for t,score in zip(choice['targets'],p):
                labels=choice['classes'][t];prediction[t]=labels[int(score[0].argmax())];values[t]=dict(zip(labels,map(float,score[0])))
            analysis['model_comparisons']['Önerilen karar birleşimi']={'predictions':prediction,'scores':values}
            analysis['preferred_model']='Önerilen karar birleşimi';analysis['late_fusion']={'recipe':choice['recipe'],'weights':choice['weights'],'selection_scope':choice['selection_scope']}
    # Independently human-labelled external affect, not a behavior-to-emotion rule.
    from src.affect.predict import infer_affect,assessment
    from src.demo.release import active_release
    affect_root=Path(active_release().get('affect_model','results/affect_v15_clean/model.json')).parent
    analysis['research_version']=active_release().get('research_version','legacy')
    analysis['audio_affect']=infer_affect(embeddings,np.concatenate(scores) if embeddings is not None else None,names if embeddings is not None else [],affect_root,digest(args.weights))
    analysis['emotion_model']=analysis['audio_affect']['status']!='model_unavailable'
    analysis['emotion_model_scope']='External human-perceived audio valence/arousal; local 127-video emotion accuracy unknown; not trained visual-audio emotion fusion'
    analysis['overall_assessment']=assessment(analysis)
    from src.emotion.behavior_bridge import infer as infer_four_emotions
    comparison=analysis['model_comparisons'].get(analysis['preferred_model']) if valid_pose else None
    from src.emotion.body import infer as infer_body
    analysis['body_state']=infer_body(visual,quality) if valid_pose else {'status':'unavailable'}
    if comparison and analysis['body_state']['status']=='estimated':
        comparison={'predictions':dict(comparison['predictions'],vucut=analysis['body_state']['prediction']),
                    'scores':dict(comparison['scores'],vucut=analysis['body_state']['scores'])}
    analysis['behavior_emotion']=infer_four_emotions(comparison,audio_analysis=audio_analysis,quality=quality)
    if analysis['behavior_emotion']['status']=='human_label_estimate':
        analysis['overall_assessment'].update(analysis['behavior_emotion'])
        analysis['overall_assessment']['scope']='Davranış ve ses olaylarından öğrenilen, insan duygu etiketleriyle eğitilmiş dört duygu tahmini'
    analysis['warnings'].append('Mutlu/Üzgün/Öfkeli/Rahat çıktısı iki kişinin insan duygu etiketlerinden (kappa 0.553) öğrenilir; gün bazlı doğrulamada başarısı düşüktür (macro-F1 yaklaşık 0.40-0.44, tesadüf yaklaşık 0.25), güven puanları kalibre değildir.')
    analysis['warnings'].append('Ses duygu boyutları dış veride insan algısı etiketleriyle öğrenilir; bu köpeklerde duygu doğruluğu henüz ölçülmedi. Ses kaynağı tek köpeğe atanmaz.')
    progress(args.output,'Analiz tamamlandı')
    (args.output/'result.json').write_text(json.dumps(clean_json(analysis),ensure_ascii=False,allow_nan=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['video','model','weights','detector','pose','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--segmenter',type=Path,default=Path('models/yolov8s-seg.pt'))
    main(parser.parse_args())
