"""Controlled offline review of existing pose and quality outputs."""
import csv
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from src.analysis.regions import model_names
from src.analysis.reader import finite
from src.batch.runner import digest, write_json
from src.pose_estimation.dog_skeleton import CONSERVATIVE_NAMES
from src.quality.analyze import inspect_points, records, unique, visible

CATEGORIES = ('no_dog', 'low_confidence', 'visibility_change', 'coordinate_jump')
TABLE_FIELDS = ('video_id', 'frame_index', 'category', 'detection_count',
                'mean_keypoint_confidence', 'visible_keypoints_0_3', 'visible_keypoints_0_5',
                'manual_dog_visible', 'manual_anatomy_correct', 'notes')


def read_json(path):
    def reject(_): raise ValueError('Nonstandard JSON constant')
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=unique, parse_constant=reject)


def histogram_add(bins, value):
    if finite(value) and 0 <= value <= 1:
        bins[min(9, int(value*10))] += 1


def histogram(bins):
    return {'edges': [i/10 for i in range(11)], 'counts': bins,
            'intervals': 'left_closed_right_open; final bin includes 1.0', 'sample_count': sum(bins)}


def panel_points(points, names, profile, threshold):
    """Same raw coordinates; only the visualization mask and cutoff differ."""
    return {i: points[i] for i in sorted(visible(points, threshold))
            if profile == 'raw39' or (profile == 'conservative' and names[i] in CONSERVATIVE_NAMES)}


def select_video(pose_path, quality_path, frames_path, per_category=2, low_threshold=.5,
                 visibility_min=8):
    if type(per_category) is not int or not 1 <= per_category <= 20:
        raise ValueError('Invalid sample count')
    if not finite(low_threshold) or not 0 <= low_threshold <= 1:
        raise ValueError('Invalid low confidence threshold')
    if type(visibility_min) is not int or visibility_min < 1:
        raise ValueError('Invalid visibility change threshold')
    quality = read_json(quality_path)
    if quality.get('quality_schema_version') != 1 or quality.get('input_sha256') != digest(pose_path):
        raise ValueError('Quality report does not match pose input')
    names = model_names(); counts = Counter(); pools = {k: [] for k in CATEGORIES}
    detection_bins = [0]*10; point_bins = [0]*10
    sums = Counter(); totals = Counter(); high = -1
    for source_item, detail_item in zip_longest(records(pose_path), records(frames_path)):
        if source_item is None or detail_item is None:
            raise ValueError('Quality detail record count mismatch')
        line, data = source_item
        _, detail = detail_item
        if not isinstance(detail,dict) or detail.get('line_number') != line:
            raise ValueError('Quality detail line mismatch')
        counts['records'] += 1
        if detail.get('status') != 'valid':
            counts['invalid_records'] += 1
            continue
        if not isinstance(data,dict): raise ValueError('Invalid source record')
        index = data.get('frame_index'); dogs = data.get('dogs')
        if (type(index) is not int or index < 0 or index <= high or not isinstance(dogs,list)
                or detail.get('frame_index') != index or detail.get('dog_count') != len(dogs)):
            raise ValueError('Source/detail frame mismatch')
        high = index; counts['frames'] += 1
        counts['no_dog' if not dogs else 'single_dog' if len(dogs)==1 else 'multi_dog'] += 1
        width,height = data.get('frame_width'),data.get('frame_height')
        width = width if type(width) is int and width>0 else None
        height = height if type(height) is int and height>0 else None
        scores=[]; frame_visibility={'0.3':0,'0.5':0}
        for dog in dogs:
            counts['dog_observations'] += 1
            if not isinstance(dog,dict): continue
            histogram_add(detection_bins,dog.get('detection_confidence'))
            points=inspect_points(dog,names,width,height)
            for point in points.values():
                value=point['confidence']
                histogram_add(point_bins,value)
                if value is not None: scores.append(value)
            for profile in ('raw39','conservative'):
                for cutoff in (.3,.5):
                    count=len(panel_points(points,names,profile,cutoff))
                    key=f'{profile}_{cutoff}'
                    sums[key]+=count;totals[key]+=1
                    if profile=='raw39': frame_visibility[str(cutoff)]+=count
        mean=sum(scores)/len(scores) if scores else None
        temporal=detail.get('temporal')
        changes=jumps=0
        if isinstance(temporal,dict):
            changes=sum(len(temporal.get(k,[])) for k in ('visible_lost','visible_gained'))
            jumps=len(temporal.get('jump_indices',[]))
            counts['eligible_temporal_pairs']+=1
            counts['visibility_changes']+=changes
            counts['jump_keypoint_pairs']+=jumps
            counts['frames_with_jumps']+=bool(jumps)
        candidate={'frame_index':index,'line_number':line,'detection_count':len(dogs),
                   'mean_keypoint_confidence':mean,'visible_keypoints_0_3':frame_visibility['0.3'],
                   'visible_keypoints_0_5':frame_visibility['0.5'],
                   'visibility_change_count':changes,'jump_keypoint_count':jumps}
        ranks={}
        if not dogs: ranks['no_dog']=(index,)
        if dogs and mean is not None and mean < low_threshold: ranks['low_confidence']=(mean,index)
        if changes>=visibility_min: ranks['visibility_change']=(-changes,index)
        if jumps: ranks['coordinate_jump']=(-jumps,index)
        for category,rank in ranks.items():
            pools[category].append((rank,candidate))
            pools[category].sort(key=lambda item:item[0])
            del pools[category][per_category:]
    if counts['frames'] != quality.get('total_frames') or counts['records'] != quality.get('counts',{}).get('records'):
        raise ValueError('Quality summary count mismatch')
    selected={}
    for category in CATEGORIES:
        for _, candidate in pools[category]:
            row=selected.setdefault(candidate['frame_index'],{**candidate,'categories':[]})
            row['categories'].append(category)
    frames=counts['frames']
    ratio=lambda n,d:n/d if d else None
    return {'quality_review_schema_version':1,
            'evidence_sha256':{'pose':digest(pose_path),'quality':digest(quality_path),'frames':digest(frames_path)},
            'total_frames':frames, 'invalid_records':counts['invalid_records'],
            'detection_coverage':ratio(frames-counts['no_dog'],frames),
            'detection_frame_rates':{key:ratio(counts[key],frames) for key in ('no_dog','single_dog','multi_dog')},
            'confidence_distributions':{'detection':histogram(detection_bins),'keypoint':histogram(point_bins)},
            'mean_visible_keypoints_per_dog':{f'{p}_{t}':ratio(sums[f'{p}_{t}'],totals[f'{p}_{t}'])
                                             for p in ('raw39','conservative') for t in (.3,.5)},
            'temporal_instability':{key:counts[key] for key in
                ('eligible_temporal_pairs','visibility_changes','jump_keypoint_pairs','frames_with_jumps')},
            'temporal_settings':{key:(quality.get('settings',{}).get(key)
                                    if finite(quality.get('settings',{}).get(key)) else None) for key in
                                ('confidence','jump_bbox_diagonal_ratio','temporal_min_iou')},
            'selected_frames':[selected[i] for i in sorted(selected)],
            'selection':{'per_category':per_category,'low_mean_confidence_below':low_threshold,
                         'visibility_changes_at_least':visibility_min,
                         'ranking':{'no_dog':'earliest','low_confidence':'lowest mean',
                                    'visibility_change':'largest count','coordinate_jump':'largest flagged count'},
                         'tie_break':'earliest frame_index'}}


def run_review(manifest_path, output_dir, per_category=2, low_threshold=.5, visibility_min=8, render=False):
    manifest_path=Path(manifest_path)
    manifest=read_json(manifest_path)
    videos=manifest.get('videos')
    if not isinstance(videos,list) or not 1<=len(videos)<=10:
        raise ValueError('Manifest must contain 1..10 videos')
    entries=[]
    for ordinal,item in enumerate(videos,1):
        if not isinstance(item,dict): raise ValueError('Invalid manifest entry')
        entry={'video_id':f'video{ordinal}'}
        for key in ('pose','quality','frames')+ (('video',) if render else ()):
            value=item.get(key)
            if not isinstance(value,str): raise ValueError('Required manifest path missing')
            path=Path(value)
            if not path.is_absolute(): path=manifest_path.parent/path
            if not path.is_file(): raise ValueError('Manifest input missing')
            entry[key]=path
        entries.append(entry)
    root=Path(output_dir)
    root.mkdir(parents=True,exist_ok=False)
    try:
        reports=[]
        for entry in entries:
            report=select_video(entry['pose'],entry['quality'],entry['frames'],per_category,low_threshold,visibility_min)
            report['video_id']=entry['video_id']
            report['visualization']={'status':'not_requested'}
            if render:
                try:
                    from src.quality.review_visualize import render_comparisons
                    report['visualization']=render_comparisons(entry['pose'],entry['video'],
                        report['selected_frames'],root/entry['video_id'])
                except Exception:
                    report['visualization']={'status':'failed_or_unavailable',
                                             'partial_pngs_may_exist':True}
            reports.append(report)
        with (root/'manual_review.csv').open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=TABLE_FIELDS);writer.writeheader()
            for report in reports:
                for selected in report['selected_frames']:
                    row={key:selected.get(key) for key in TABLE_FIELDS[:7]}
                    row.update(video_id=report['video_id'],category=';'.join(selected['categories']))
                    writer.writerow(row)
        result={'status':'complete','videos':reports,
                'temporal_settings_comparable':all(r['temporal_settings']==reports[0]['temporal_settings'] for r in reports),
                'limitations':['Confidence and detection coverage are not accuracy or anatomical correctness.',
                    'No-dog frames are not labeled false negatives without manual review.',
                    'Selected extremes/earliest examples are not an unbiased sample.',
                    'Temporal flags inherit the quality report settings; no identity tracking.',
                    'PNG source video identity is not cryptographically linked to pose JSONL.',
                    'All manual fields intentionally remain blank.']}
        write_json(root/'comparison.json',result)
        return result
    except BaseException as exc:
        write_json(root/'failure.json',{'status':'failed','error_category':type(exc).__name__})
        raise
