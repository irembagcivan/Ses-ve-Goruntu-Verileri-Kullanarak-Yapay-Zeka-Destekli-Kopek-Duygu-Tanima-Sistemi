"""Adjacent-frame motion in original pixels; no interpolation or smoothing."""
from dataclasses import dataclass, asdict
import math

from src.analysis.regions import REGIONS, analysis_names


@dataclass(frozen=True)
class MotionSettings:
    confidence: float = .5
    detection_confidence: float = 0.0
    normalization: str = 'bbox_diagonal'
    max_step_ratio: float = .5
    min_region_points: int = 2

    def __post_init__(self):
        if not 0<=self.confidence<=1 or not 0<=self.detection_confidence<=1:
            raise ValueError('Confidence thresholds must be in [0,1]')
        if self.normalization not in {'bbox_diagonal','none'}:
            raise ValueError('Unknown normalization')
        if not math.isfinite(self.max_step_ratio) or self.max_step_ratio<=0:
            raise ValueError('Max step ratio must be finite and positive')
        if type(self.min_region_points) is not int or self.min_region_points<1:
            raise ValueError('Minimum region point count must be positive')

    def to_dict(self):return asdict(self)


BASE_METRICS=('mean_displacement_px','max_displacement_px','mean_speed_px_s','max_speed_px_s',
              'mean_displacement_normalized','max_displacement_normalized',
              'mean_speed_normalized_s','max_speed_normalized_s')
PREFIXES=('global',)+tuple(f'{name}_{kind}' for name in REGIONS for kind in ('absolute','relative'))
VECTOR_METRICS=('tail_vector_displacement_px','tail_vector_speed_px_s',
                'tail_vector_displacement_normalized','tail_vector_speed_normalized_s')
METRIC_FIELDS=tuple(f'{prefix}_{metric}' for prefix in PREFIXES for metric in BASE_METRICS)+VECTOR_METRICS
BASE_FIELDS=('line_number','frame_index','timestamp_seconds','timestamp_source','status','eligible_pair',
             'dog_count','available_keypoints','missing_current_keypoints','point_errors','dt_seconds',
             'normalization_scale_px','bbox_step_ratio','bbox_scale_ratio','common_keypoints',
             'valid_keypoints','jump_keypoints','jump_keypoint_names','flagged_max_displacement_px',
             'head_valid_points','body_valid_points','tail_valid_points')
CSV_FIELDS=BASE_FIELDS+METRIC_FIELDS


def distance(a,b):return math.hypot(a[0]-b[0],a[1]-b[1])


def center(box):return (box[0]/2+box[2]/2,box[1]/2+box[3]/2)


def diagonal(box):return math.hypot(box[2]-box[0],box[3]-box[1])


def subtract(a,b):return (a[0]-b[0],a[1]-b[1])


def metrics(values,dt,scale):
    if not values:return None
    mean=sum(value/len(values) for value in values);maximum=max(values)
    result=dict(zip(BASE_METRICS,(mean,maximum,mean/dt,maximum/dt,
                                mean/scale if scale else None, maximum/scale if scale else None,
                                mean/scale/dt if scale else None, maximum/scale/dt if scale else None)))
    # Extremely small dt may overflow. Never serialize NaN/Inf or treat it as zero.
    return {key:value if value is None or math.isfinite(value) else None for key,value in result.items()}


class MotionAnalyzer:
    def __init__(self,names,settings):
        self.names=analysis_names(names);self.settings=settings;self.previous=None

    def process(self,frame):
        row={field:None for field in CSV_FIELDS}
        valid_current=set(frame.points)&set(self.names) if frame.reason is None else set()
        row.update(line_number=frame.line_number,frame_index=frame.index,timestamp_seconds=frame.timestamp,
                   timestamp_source=frame.timestamp_source,dog_count=frame.dog_count,
                   available_keypoints=len(valid_current),missing_current_keypoints=len(self.names)-len(valid_current),
                   point_errors=frame.point_errors,eligible_pair=False,common_keypoints=0,
                   valid_keypoints=0,jump_keypoints=0,jump_keypoint_names='')
        previous=self.previous
        self.previous=frame if frame.reason is None else None
        if frame.reason:
            row['status']=frame.reason;return row
        if previous is None:row['status']='baseline';return row
        if frame.index!=previous.index+1:row['status']='frame_gap';return row
        if (frame.width,frame.height)!=(previous.width,previous.height):
            row['status']='resolution_change';return row
        dt=frame.timestamp-previous.timestamp;row['dt_seconds']=dt
        if not math.isfinite(dt) or dt<=0:row['status']='nonpositive_dt';return row
        raw_scale=diagonal(frame.bbox)/2+diagonal(previous.bbox)/2
        if not math.isfinite(raw_scale) or raw_scale<=0:row['status']='invalid_scale';return row
        row['eligible_pair']=True
        scale=raw_scale if self.settings.normalization=='bbox_diagonal' else None
        row['normalization_scale_px']=scale
        old_center,new_center=center(previous.bbox),center(frame.bbox)
        row['bbox_step_ratio']=distance(new_center,old_center)/raw_scale
        row['bbox_scale_ratio']=diagonal(frame.bbox)/diagonal(previous.bbox)
        common=[name for name in self.names if name in previous.points and name in frame.points]
        row['common_keypoints']=len(common)
        if not common:row['status']='no_valid_keypoints';return row
        displacements={name:distance(frame.points[name],previous.points[name]) for name in common}
        flagged=[name for name in common if not math.isfinite(displacements[name])
                 or displacements[name]/raw_scale>self.settings.max_step_ratio]
        accepted=[name for name in common if name not in flagged]
        row.update(jump_keypoints=len(flagged),jump_keypoint_names=';'.join(flagged),valid_keypoints=len(accepted))
        if flagged:
            finite=[displacements[name] for name in flagged if math.isfinite(displacements[name])]
            row['flagged_max_displacement_px']=max(finite) if finite else None
        if not accepted:row['status']='all_points_flagged';return row
        global_values=metrics([displacements[n] for n in accepted],dt,scale)
        row.update({f'global_{key}':value for key,value in global_values.items()})
        for region,names in REGIONS.items():
            selected=[n for n in names if n in accepted]
            row[f'{region}_valid_points']=len(selected)
            if len(selected)<self.settings.min_region_points:continue
            absolute=metrics([displacements[n] for n in selected],dt,scale)
            relative=metrics([distance(subtract(frame.points[n],new_center),
                                       subtract(previous.points[n],old_center)) for n in selected],dt,scale)
            row.update({f'{region}_absolute_{key}':value for key,value in absolute.items()})
            row.update({f'{region}_relative_{key}':value for key,value in relative.items()})
        if all(name in accepted for name in ('tail_base','tail_end')):
            old=subtract(previous.points['tail_end'],previous.points['tail_base'])
            new=subtract(frame.points['tail_end'],frame.points['tail_base'])
            tail=distance(new,old)
            values=(tail,tail/dt,tail/scale if scale else None,tail/scale/dt if scale else None)
            row.update({key:value if value is None or math.isfinite(value) else None
                        for key,value in zip(VECTOR_METRICS,values)})
        row['status']='partial' if len(accepted)<len(self.names) else 'valid'
        if row['global_mean_speed_px_s'] is None:row['status']='nonfinite_motion'
        return row
