"""Describe exported evidence, without interpreting pose accuracy or identity."""
import json
import math
from collections import Counter
from pathlib import Path

from src.analysis.regions import model_names
from src.analysis.reader import finite
from src.batch.runner import digest

THRESHOLDS = (.3, .5, .7)


def iou(a, b):
    if not all(isinstance(v, list) and len(v) == 4 and all(finite(x) for x in v)
               and v[2] > v[0] and v[3] > v[1] for v in (a, b)):
        return None
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-intersection)


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate key')
        result[key] = value
    return result


def records(path):
    with Path(path).open('rb') as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                data = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                                  parse_constant=lambda _: None)
                yield line, data if isinstance(data, dict) else None
            except (ValueError, UnicodeError, RecursionError):
                yield line, None


def inspect_points(dog, names, width, height):
    """Missing != low score != invalid coordinates; duplicates are unusable."""
    points = dog.get('keypoints') if isinstance(dog, dict) else None
    valid, duplicates = {}, set()
    for point in points if isinstance(points, list) else []:
        if not isinstance(point, dict):
            continue
        index = point.get('index')
        if type(index) is not int or not 0 <= index < len(names) or point.get('name') != names[index]:
            continue
        if index in valid:
            duplicates.add(index)
        valid[index] = point
    result = {}
    for index, point in valid.items():
        if index in duplicates:
            continue
        x, y, score = point.get('x'), point.get('y'), point.get('confidence')
        coordinates = (finite(x) and finite(y) and width is not None and height is not None
                       and 0 <= x < width and 0 <= y < height
                       and point.get('in_frame') is True)
        confidence = (finite(score) and 0 <= score <= 1
                      and point.get('confidence_valid') is True)
        result[index] = {'xy': (x, y) if coordinates else None,
                         'confidence': score if confidence else None}
    return result


def visible(points, threshold):
    return {i for i, p in points.items() if p['xy'] is not None
            and p['confidence'] is not None and p['confidence'] >= threshold}


def context_checks(input_hash, frames, metadata_path=None, motion_path=None):
    """Optional, allowlisted consistency checks; no source paths or arbitrary text."""
    result = {}
    for label, path in (('pose_metadata',metadata_path),('motion_summary',motion_path)):
        if path is None:
            result[label] = {'status': 'not_provided'}
            continue
        try:
            with Path(path).open(encoding='utf-8') as stream: data=json.load(stream)
            if not isinstance(data,dict): raise ValueError('Invalid context')
            if label == 'pose_metadata':
                profile=data.get('visualization_profile')
                result[label] = {'status': 'read', 'export_complete':data.get('run_status')=='complete',
                    'frame_count_matches':data.get('frames_written')==frames,
                    'visualization_profile':profile if profile in {'official','legacy','conservative'} else None,
                    'visualization_threshold':None}
                score=data.get('thresholds',{}).get('visualization_keypoint_confidence')
                if finite(score) and 0<=score<=1: result[label]['visualization_threshold']=score
            else:
                result[label] = {'status': 'read', 'frame_count_matches':data.get('total_frames')==frames,
                                 'input_hash_matches':data.get('input_sha256')==input_hash}
        except (OSError, ValueError, TypeError, AttributeError):
            result[label] = {'status': 'unavailable_or_invalid'}
    return result


def analyze(input_path, output, threshold=.5, jump_ratio=.5, overlap=.8, metadata_path=None, motion_path=None, temporal_min_iou=.3):
    if not all(finite(v) and 0 <= v <= 1 for v in (threshold, overlap, temporal_min_iou)) or not finite(jump_ratio) or jump_ratio <= 0:
        raise ValueError('Invalid thresholds')
    source, target = Path(input_path), Path(output)
    detail = target.with_suffix('.frames.jsonl')
    if not source.is_file() or target.suffix.lower() != '.json':
        raise ValueError('Existing input and JSON output required')
    if len({source.resolve(), target.resolve(), detail.resolve()}) != 3:
        raise ValueError('Output aliases input')
    if any(p is not None and Path(p).resolve() in {target.resolve(), detail.resolve()}
           for p in (metadata_path,motion_path)):
        raise ValueError('Quality output aliases context input')
    if target.exists() or detail.exists():
        raise FileExistsError('Choose new quality output paths')
    names = model_names()
    counts = Counter({k: 0 for k in ('records','invalid_records','frames','no_dog','single_dog',
        'multi_dog','dog_observations','missing_keypoints','zero_reliable_frames',
        'detected_zero_reliable_frames','overlap_frames','temporal_pairs','visibility_changes','jump_pairs')}); reliable = Counter(); confidence_sum = detection_sum = 0.0
    previous = None; high_index = -1
    created = []
    try:
        with target.open('x', encoding='utf-8') as summary_stream:
            created.append(target)
            with detail.open('x', encoding='utf-8') as frame_stream:
                created.append(detail)
                for line, data in records(source):
                    counts['records'] += 1
                    row = {'line_number': line, 'frame_index': None, 'status': 'invalid_record',
                           'dogs': [], 'temporal': None, 'overlapping_pairs': []}
                    index = data.get('frame_index') if data else None
                    dogs = data.get('dogs') if data else None
                    if type(index) is not int or index < 0 or index <= high_index or not isinstance(dogs, list):
                        counts['invalid_records'] += 1; previous = None
                    else:
                        high_index = index
                        row.update(frame_index=index, status='valid', dog_count=len(dogs))
                        counts['frames'] += 1
                        counts['no_dog' if not dogs else 'single_dog' if len(dogs) == 1 else 'multi_dog'] += 1
                        width, height = data.get('frame_width'), data.get('frame_height')
                        width = width if type(width) is int and width > 0 else None
                        height = height if type(height) is int and height > 0 else None
                        frame_visible = 0; all_points = []
                        for position, dog in enumerate(dogs):
                            points = inspect_points(dog, names, width, height)
                            all_points.append(points)
                            score = dog.get('detection_confidence') if isinstance(dog, dict) else None
                            if finite(score) and 0 <= score <= 1:
                                detection_sum += score; counts['detection_scores'] += 1
                            else:
                                score = None
                            counts['dog_observations'] += 1
                            counts['missing_keypoints'] += len(names)-len(points)
                            for i, point in points.items():
                                if point['confidence'] is not None:
                                    confidence_sum += point['confidence']; counts['keypoint_scores'] += 1
                            for t in THRESHOLDS:
                                counts['visible_total_'+str(t)] += len(visible(points, t))
                            good = visible(points, threshold)
                            reliable.update(good); frame_visible += len(good)
                            scores = [p['confidence'] for p in points.values() if p['confidence'] is not None]
                            row['dogs'].append({'mean_keypoint_confidence': sum(scores)/len(scores) if scores else None,
                                'detection_index': position, 'detection_confidence': score,
                                'missing_keypoints': len(names)-len(points),
                                'invalid_keypoints': sum(p['xy'] is None or p['confidence'] is None for p in points.values()),
                                'reliable_keypoints': len(good),
                                'visible_counts': {str(t): len(visible(points, t)) for t in THRESHOLDS}})
                        if not frame_visible:
                            counts['zero_reliable_frames'] += 1
                            if dogs: counts['detected_zero_reliable_frames'] += 1
                        for a in range(len(dogs)):
                            for b in range(a+1, len(dogs)):
                                boxes = [d.get('bbox_xyxy') if isinstance(d, dict) else None for d in (dogs[a], dogs[b])]
                                value = iou(*boxes)
                                if value is not None and value >= overlap:
                                    row['overlapping_pairs'].append({'detection_indices': [a,b], 'iou': value})
                        if row['overlapping_pairs']: counts['overlap_frames'] += 1
                        current = None
                        if len(dogs) == 1:
                            points = all_points[0]
                            box = dogs[0].get('bbox_xyxy') if isinstance(dogs[0], dict) else None
                            scale = math.hypot(box[2]-box[0], box[3]-box[1]) if iou(box, box) is not None else None
                            current = (index, points, scale, width, height, box)
                            if (previous and index == previous[0]+1 and (width,height) == previous[3:5]
                                    and width is not None and height is not None
                                    and iou(box,previous[5]) is not None
                                    and iou(box,previous[5]) >= temporal_min_iou):
                                old = previous[1]; before = visible(old,threshold); now = visible(points,threshold)
                                transition = {'visible_gained': sorted(now-before), 'visible_lost': sorted(before-now),
                                    'below_threshold': [], 'missing': [], 'invalid': [], 'reappeared': [],
                                    'jump_indices': [], 'jump_scale_available': scale is not None and previous[2] is not None}
                                for i in before-now:
                                    point = points.get(i)
                                    key = ('missing' if point is None else 'below_threshold'
                                           if point['confidence'] is not None and point['confidence'] < threshold else 'invalid')
                                    transition[key].append(i)
                                transition['reappeared'] = sorted(now-before)
                                if scale is not None and previous[2] is not None:
                                    denominator = (scale+previous[2])/2
                                    for i in before & now:
                                        step = math.dist(old[i]['xy'],points[i]['xy'])
                                        if step/denominator > jump_ratio: transition['jump_indices'].append(i)
                                transition['jump_indices'].sort()
                                row['temporal'] = transition
                                counts['temporal_pairs'] += 1
                                counts['visibility_changes'] += len(now ^ before)
                                counts['jump_pairs'] += len(transition['jump_indices'])
                        previous = current
                    frame_stream.write(json.dumps(row, allow_nan=False)+'\n')
                frames = counts['frames']; observations = counts['dog_observations']
                ratio = lambda value, total: value/total if total else None
                result = {'quality_schema_version': 1, 'input_sha256': digest(source),
                    'counts': dict(counts), 'total_frames': frames,
                    'rates': {key: ratio(counts[key],frames) for key in
                              ('no_dog','single_dog','multi_dog','zero_reliable_frames','detected_zero_reliable_frames')},
                    'mean_visible_keypoints_per_dog': {str(t): ratio(counts['visible_total_'+str(t)],observations) for t in THRESHOLDS},
                    'mean_detection_confidence': ratio(detection_sum,counts['detection_scores']),
                    'mean_keypoint_confidence': ratio(confidence_sum,counts['keypoint_scores']),
                    'missing_keypoint_rate': ratio(counts['missing_keypoints'],observations*len(names)),
                    'mean_visibility_changes_per_eligible_pair': ratio(counts['visibility_changes'],counts['temporal_pairs']),
                    'per_keypoint': [{'index': i, 'name': name, 'reliable_observations': reliable[i],
                                      'reliability_rate': ratio(reliable[i], observations)} for i,name in enumerate(names)],
                    'settings': {'confidence': threshold, 'visible_thresholds': THRESHOLDS,
                                 'jump_bbox_diagonal_ratio': jump_ratio, 'overlap_iou': overlap,
                                 'temporal_min_iou': temporal_min_iou},
                    'limitations': ['Confidence is not accuracy; no ground truth or anatomical correctness assessment.',
                         'Temporal comparisons require consecutive single-dog frames and overlapping boxes; identity remains unverified.',
                         'Jumps may be camera motion, animal motion, box scale change or prediction error.',
                         'Zero detections do not establish whether a dog was actually visible.',
                         'Missing point means absent/duplicate/mismatched name-index; invalid coordinates are separate.',
                         'Frames are valid ordered JSONL records, not independently verified video frames.',
                         'All 39 markers are measured, unlike the conservative visualization subset.']}
                result['context_checks'] = context_checks(result['input_sha256'],frames,metadata_path,motion_path)
                summary_stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
        return result
    except BaseException:
        for path in created: path.unlink(missing_ok=True)
        raise
