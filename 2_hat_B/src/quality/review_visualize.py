"""Four-panel views of the same decoded frame and existing floating-point pose."""
import csv
from pathlib import Path

from src.analysis.regions import model_names
from src.analysis.reader import finite
from src.quality.analyze import records, inspect_points
from src.quality.review import panel_points

PANELS=(('raw39',.3),('raw39',.5),('conservative',.3),('conservative',.5))


def render_comparisons(pose_path, video_path, selected_frames, output_dir):
    if not selected_frames:
        return {'status':'no_selected_frames','frames_written':0}
    selected={r['frame_index']:r for r in selected_frames}
    data_by_index={}
    for line,data in records(pose_path):
        if (isinstance(data,dict) and type(data.get('frame_index')) is int
                and data['frame_index'] in selected
                and selected[data['frame_index']]['line_number']==line):
            data_by_index[data['frame_index']]=data
    if len(data_by_index)!=len(selected): raise ValueError('Selected source record missing')
    import cv2
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=False)
    capture=cv2.VideoCapture(str(video_path)); written=0
    try:
        if not capture.isOpened(): raise ValueError('Video cannot be decoded')
        index=0
        while index<=max(selected):
            ok,pixels=capture.read()
            if not ok: raise ValueError('Video ended before selected frame')
            if index in selected:
                data=data_by_index[index];height,width=pixels.shape[:2]
                if (data.get('frame_width'),data.get('frame_height'))!=(width,height):
                    raise ValueError('Video dimensions do not match pose')
                names=model_names(); dogs=data['dogs']
                points=[inspect_points(dog,names,width,height) for dog in dogs]
                panels=[]
                for profile,cutoff in PANELS:
                    image=cv2.copyMakeBorder(pixels,70,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
                    total=0
                    for position,(dog,observations) in enumerate(zip(dogs,points)):
                        color=((60+position*73)%256,255,(80+position*53)%256)
                        for i,point in panel_points(observations,names,profile,cutoff).items():
                            x,y=point['xy'];score=point['confidence']
                            xy=(min(width-1,round(x)),70+min(height-1,round(y)))
                            cv2.circle(image,xy,3,color,-1)
                            label=f'D{position}:{names[i]} {score:.3f}'
                            cv2.putText(image,label,(min(xy[0]+4,max(0,width-250)),max(80,xy[1]-4)),
                                        cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
                            total+=1
                        if isinstance(dog,dict):
                            box=dog.get('bbox_xyxy')
                            if (isinstance(box,list) and len(box)==4 and all(finite(v) for v in box)
                                    and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):
                                cv2.rectangle(image,(round(box[0]),70+round(box[1])),
                                    (min(width-1,round(box[2])),70+min(height-1,round(box[3]))),color,1)
                    scores=[dog.get('detection_confidence') if isinstance(dog,dict) else None for dog in dogs]
                    text=' '.join(f'D{i}={v:.3f}' if finite(v) else f'D{i}=NA' for i,v in enumerate(scores))
                    for message,y in ((f'frame={index} {profile} confidence>={cutoff} visible={total} dogs={len(dogs)}',25),
                                      (text,50)):
                        cv2.putText(image,message,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                    path=root/f'frame-{index:08d}-{profile}-{cutoff}.png'
                    if not cv2.imwrite(str(path),image): raise OSError('PNG write failed')
                    panels.append(image)
                grid=cv2.vconcat([cv2.hconcat(panels[:2]),cv2.hconcat(panels[2:])])
                if not cv2.imwrite(str(root/f'frame-{index:08d}-comparison.png'),grid):
                    raise OSError('Grid write failed')
                with (root/f'frame-{index:08d}-points.csv').open('x',encoding='utf-8',newline='') as stream:
                    writer=csv.writer(stream)
                    writer.writerow(('detection_index','keypoint_index','keypoint_name','x','y','confidence',
                                     'raw39_0_3','raw39_0_5','conservative_0_3','conservative_0_5'))
                    for position,observations in enumerate(points):
                        masks=[panel_points(observations,names,p,t) for p,t in PANELS]
                        for i,name in enumerate(names):
                            point=observations.get(i,{})
                            xy=point.get('xy') or (None,None)
                            writer.writerow((position,i,name,*xy,point.get('confidence'),*(i in m for m in masks)))
                written+=1
            index+=1
    finally:
        capture.release()
    return {'status':'complete','frames_written':written,'png_count':written*5,
            'source_video_identity_verified':False,'anatomical_edges_drawn':False}
