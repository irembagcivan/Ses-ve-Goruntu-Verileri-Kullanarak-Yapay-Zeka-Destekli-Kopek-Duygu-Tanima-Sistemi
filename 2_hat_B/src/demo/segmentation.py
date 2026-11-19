"""Render real per-instance masks on every exported frame and retain mask PNGs."""
import argparse,json,subprocess,zipfile
from pathlib import Path
import cv2,imageio_ffmpeg,numpy as np
from src.preprocessing.interfaces import VideoFrame
from src.preprocessing.segmentation import YoloDogSegmenter
from src.demo.extract import digest

COLORS=[(132,239,182),(105,191,255),(255,151,189),(215,215,115),(160,140,255)]

def overlay(pixels,instances):
    canvas=pixels.copy()
    for index,instance in enumerate(instances):
        color=np.array(COLORS[index%len(COLORS)],float)
        canvas[instance.mask]=np.rint(.6*canvas[instance.mask]+.4*color).astype(np.uint8)
        contours,_=cv2.findContours(instance.mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas,contours,-1,COLORS[index%len(COLORS)],2)
    return canvas

def render(source,output,weights,seconds=60,fps=8,segmenter=None):
    if not np.isfinite(seconds) or not np.isfinite(fps) or seconds<=0 or fps<=0:raise ValueError('Positive finite segmentation duration and FPS required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);masks_dir=output/'masks';masks_dir.mkdir()
    segmenter=segmenter or YoloDogSegmenter(weights);capture=cv2.VideoCapture(str(source))
    if not capture.isOpened():raise ValueError('Cannot open segmentation source')
    source_fps=capture.get(cv2.CAP_PROP_FPS);source_width=capture.get(cv2.CAP_PROP_FRAME_WIDTH);source_height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if not np.isfinite([source_fps,source_width,source_height]).all() or source_fps<=0 or source_width<=0 or source_height<=0:capture.release();raise ValueError('Invalid segmentation source FPS')
    ratio=min(1,960/max(source_width,source_height));width=max(2,int(source_width*ratio)//2*2);height=max(2,int(source_height*ratio)//2*2);out_fps=min(fps,source_fps)
    writer=cv2.VideoWriter(str(output/'silent.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),out_fps,(width,height))
    if not writer.isOpened():capture.release();raise ValueError('Cannot create segmented video')
    source_index=0;emitted=0;records=[];best=None;best_count=-1
    try:
        while emitted<int(seconds*out_fps):
            target=int(round(emitted*source_fps/out_fps))
            while source_index<target:
                if not capture.grab():break
                source_index+=1
            ok,pixels=capture.read()
            if not ok:break
            actual=source_index;source_index+=1;timestamp=actual/source_fps
            pixels=cv2.resize(pixels,(width,height));instances=segmenter.segment(VideoFrame(actual,timestamp,pixels));canvas=overlay(pixels,instances);writer.write(canvas)
            frame_record={'frame':emitted,'source_frame':actual,'timestamp':round(timestamp,6),'instances':[]}
            for index,instance in enumerate(instances,1):
                name=f'frame_{emitted:05d}_dog_{index:02d}.png'
                if not cv2.imwrite(str(masks_dir/name),instance.mask.astype(np.uint8)*255):raise IOError('Cannot save instance mask')
                frame_record['instances'].append({'frame_local_id':index,'score':instance.detection.confidence,'bbox_xyxy':list(instance.detection.xyxy),'area_pixels':int(instance.mask.sum()),'mask':name})
            records.append(frame_record)
            if len(instances)>best_count:
                union=np.logical_or.reduce([obj.mask for obj in instances]) if instances else np.zeros((height,width),bool)
                best=(canvas.copy(),union.astype(np.uint8)*255,timestamp);best_count=len(instances)
            emitted+=1
    finally:capture.release();writer.release()
    if not emitted:raise ValueError('No segmentation frames decoded')
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(output/'silent.mp4'),'-i',str(source),'-map','0:v:0','-map','1:a:0?','-t',str(emitted/out_fps),'-c:v','libx264','-preset','veryfast','-crf','24','-c:a','aac','-movflags','+faststart',str(output/'segmented.mp4')]
    subprocess.run(command,check=True,capture_output=True,timeout=180);(output/'silent.mp4').unlink()
    cv2.imwrite(str(output/'overlay.jpg'),best[0]);cv2.imwrite(str(output/'mask.png'),best[1])
    metadata={'status':'complete','model':Path(weights).stem,'weights_sha256':digest(weights),'frames':emitted,'fps':out_fps,
        'seconds':emitted/out_fps,'source_size':[int(source_width),int(source_height)],'mask_size':[width,height],
        'max_dogs':max(len(r['instances']) for r in records),'frames_with_dogs':sum(bool(r['instances']) for r in records),
        'preview_timestamp':best[2],'instance_ids':'frame-local only, not persistent identities',
        'mask_quality_evaluated':False,'records':records}
    (output/'segmentation.json').write_text(json.dumps(metadata,ensure_ascii=False,allow_nan=False))
    with zipfile.ZipFile(output/'instance_masks.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(output/'segmentation.json','segmentation.json')
        for path in masks_dir.glob('*.png'):archive.write(path,'masks/'+path.name)
    return metadata

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--weights',type=Path,default=Path('models/yolov8s-seg.pt'));parser.add_argument('--seconds',type=float,default=60)
    args=parser.parse_args();print(json.dumps({k:v for k,v in render(args.video,args.output,args.weights,args.seconds).items() if k!='records'},indent=2))
