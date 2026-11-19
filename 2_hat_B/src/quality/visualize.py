"""Optional decoded-frame PNG diagnostics, with no inference or source modification."""
import json
from pathlib import Path
from src.analysis.regions import model_names
from src.analysis.reader import finite
from src.quality.analyze import records, inspect_points, visible, THRESHOLDS


def export_samples(source_jsonl, detail_path, video_path, output_dir, threshold=.5, max_frames=8):
    if type(max_frames) is not int or not 1 <= max_frames <= 100:
        raise ValueError('Sample limit must be [1,100]')
    selected = {}; selected_lines = {}; categories = set()
    with Path(detail_path).open(encoding='utf-8') as stream:
        for raw in stream:
            row = json.loads(raw)
            if row['status'] != 'valid': continue
            reasons = []
            count = row['dog_count']
            if count == 0: reasons.append('no_dog')
            if count > 1: reasons.append('multiple_dogs')
            if count and sum(d['reliable_keypoints'] for d in row['dogs']) <= 5*count:
                reasons.append('few_keypoints')
            if row['temporal'] and row['temporal']['jump_indices']: reasons.append('jump')
            novel = set(reasons)-categories
            if novel and len(selected) < max_frames:
                selected[row['frame_index']] = None
                selected_lines[row['frame_index']] = row['line_number']
                categories.update(novel)
    if not selected:
        return {'status': 'no_flagged_frames', 'png_count': 0}
    # Keep only bounded selected records, not all video frames.
    for line, data in records(source_jsonl):
        if data and type(data.get('frame_index')) is int and data['frame_index'] in selected and selected_lines[data['frame_index']] == line:
            selected[data['frame_index']] = data
    import cv2
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video_path))
    written = 0; index = 0
    try:
        if not capture.isOpened(): raise ValueError('Video decode unavailable')
        while index <= max(selected):
            ok, pixels = capture.read()
            if not ok: raise ValueError('Selected frame unavailable')
            if index in selected:
                data = selected[index]
                height, width = pixels.shape[:2]
                if data is None or (data.get('frame_width'), data.get('frame_height')) != (width,height):
                    raise ValueError('Video and JSONL dimensions disagree')
                for cutoff in sorted(set((*THRESHOLDS,threshold))):
                    image = pixels.copy(); dogs = data['dogs']; reliable = 0
                    for position, dog in enumerate(dogs):
                        if not isinstance(dog, dict): continue
                        points = inspect_points(dog, model_names(), width, height)
                        good = visible(points,cutoff); reliable += len(good)
                        for i in good:
                            x,y = points[i]['xy']
                            cv2.circle(image, (min(width-1,round(x)),min(height-1,round(y))), 3,(0,255,0),-1)
                        box = dog.get('bbox_xyxy')
                        if (isinstance(box,list) and len(box)==4 and all(finite(v) for v in box)
                                and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):
                            cv2.rectangle(image, (round(box[0]),round(box[1])),
                                          (min(width-1,round(box[2])),min(height-1,round(box[3]))),(255,180,0),2)
                        score = dog.get('detection_confidence')
                        text = f'detection {position}: {score:.3f}' if finite(score) else f'detection {position}: unavailable'
                        cv2.putText(image,text,(10,50+position*20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                    cv2.putText(image,f'frame={index} dogs={len(dogs)} reliable={reliable} threshold={cutoff}',
                                (10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                    path = root / f'frame-{index:08d}-threshold-{cutoff:.6f}.png'
                    if path.exists(): raise FileExistsError('PNG collision')
                    if not cv2.imwrite(str(path),image): raise OSError('PNG export failed')
                    written += 1
            index += 1
    finally:
        capture.release()
    return {'status': 'complete', 'png_count': written,
            'source_video_identity_verified': False}
