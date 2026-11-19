"""Streaming CSV and online aggregate statistics; optional plotting is separate."""
from collections import Counter
import csv
import json
import math
from pathlib import Path

from src.analysis.features import MotionAnalyzer, METRIC_FIELDS, CSV_FIELDS
from src.analysis.reader import JsonlFrames
from src.analysis.regions import model_names, REGIONS, analysis_names

LIMITATIONS=(
    'Descriptive image-plane motion, not emotion, anatomy accuracy or a behavior label.',
    'No tracking: consecutive single detections can still belong to different animals.',
    'No interpolation, smoothing or motion bridging across absent/multi-dog/malformed frames.',
    'Timestamps use input JSONL; frame_index_div_fps is approximate for variable-FPS video.',
    'BBox normalization and relative coordinates do not fully correct camera motion, rotation or zoom.',
    'BBox-center localization jitter can create or suppress relative motion.',
    'Jump flags are heuristics; genuine fast motion may be excluded and smaller errors may remain.',
    'Regional summaries are means of available surface markers, not joint or center-of-mass measurements.',
    'Summary means are unweighted means of available per-frame metrics, not time-weighted estimates.',
    'Partial JSONL input is analyzed as supplied; export completion is not independently verified.',
)


class Statistics:
    def __init__(self):
        self.total=0;self.mean=0.0;self.maximum=None
    def add(self,value):
        self.total+=1
        self.mean += (value-self.mean)/self.total
        self.maximum=value if self.maximum is None else max(self.maximum,value)
    def result(self,frames,eligible):
        return {'count':self.total,'mean':self.mean if self.total else None,'max':self.maximum,
                'missing_frames':frames-self.total,
                'missing_rate_all_frames':(frames-self.total)/frames if frames else None,
                'missing_rate_eligible_pairs':(eligible-self.total)/eligible if eligible else None}


class Summary:
    def __init__(self,settings,names):
        self.settings=settings;self.names=names;self.frames=0;self.identified=0
        self.valid=0;self.eligible=0;self.missing_points=0;self.jump_points=0;self.jump_frames=0
        self.reasons=Counter();self.sources=Counter();self.metrics={name:Statistics() for name in METRIC_FIELDS}
    def add(self,row):
        self.frames+=1;self.identified+=row['frame_index'] is not None
        self.eligible+=row['eligible_pair']
        self.valid+=row['global_mean_speed_px_s'] is not None
        self.missing_points+=row['missing_current_keypoints']
        self.jump_points+=row['jump_keypoints'];self.jump_frames+=row['jump_keypoints']>0
        self.reasons[row['status']]+=1;self.sources[row['timestamp_source']]+=1
        for name,stat in self.metrics.items():
            if row[name] is not None:stat.add(row[name])
    def result(self,reader):
        from scripts.collect_environment import git_info
        values={name:stat.result(self.frames,self.eligible) for name,stat in self.metrics.items()}
        count=len(analysis_names(self.names))
        return {'analysis_schema_version':1,'status':'complete' if self.valid else 'no_valid_motion',
                'total_frames':self.frames,'identified_frame_records':self.identified,
                'valid_analysis_frames':self.valid,'eligible_adjacent_time_pairs':self.eligible,
                'fully_valid_motion_frames':self.reasons.get('valid',0),
                'partial_motion_frames':self.reasons.get('partial',0),
                'missing_motion_rate':(self.frames-self.valid)/self.frames if self.frames else None,
                'missing_current_keypoint_rate':self.missing_points/(self.frames*count) if self.frames else None,
                'jump_keypoint_pairs':self.jump_points,'frames_with_jump_flags':self.jump_frames,
                'frame_status_counts':dict(self.reasons),'timestamp_source_counts':dict(self.sources),
                'blank_lines_skipped':reader.blank_lines,'nonstandard_json_constants':reader.nonstandard_constants,
                'input_sha256':reader.digest.hexdigest(),'git':git_info(),'settings':self.settings.to_dict(),
                'analysis_keypoints':list(analysis_names(self.names)),
                'excluded_keypoints':[n for n in self.names if n not in analysis_names(self.names)],
                'regions':{key:list(names) for key,names in REGIONS.items()},
                'metrics':values,
                'regional_summaries':{region:{key:val for key,val in values.items() if key.startswith(region+'_')}
                                      for region in REGIONS},
                'plot':{'status':'not_requested'},'limitations':list(LIMITATIONS)}


def analyze_file(input_path,csv_path,summary_path,settings,plot_path=None):
    source,csv_target,json_target=map(Path,(input_path,csv_path,summary_path))
    paths=[source,csv_target,json_target]+([Path(plot_path)] if plot_path else [])
    if len({p.resolve() for p in paths})!=len(paths):raise ValueError('Input and output paths must differ')
    if not source.is_file():raise FileNotFoundError('Input JSONL is missing')
    if csv_target.suffix.lower()!='.csv' or json_target.suffix.lower()!='.json':
        raise ValueError('CSV and summary JSON extensions are required')
    if plot_path and Path(plot_path).suffix.lower()!='.png':raise ValueError('Plot output must be PNG')
    if csv_target.exists() or json_target.exists():raise FileExistsError('An output already exists; choose new paths')
    names=model_names();analyzer=MotionAnalyzer(names,settings);summary=Summary(settings,names)
    reader=JsonlFrames(source,names,settings.confidence,settings.detection_confidence)
    created=[]
    try:
        with csv_target.open('x',encoding='utf-8',newline='') as csv_stream:
            created.append(csv_target)
            with json_target.open('x',encoding='utf-8') as json_stream:
                created.append(json_target)
                writer=csv.DictWriter(csv_stream,fieldnames=CSV_FIELDS);writer.writeheader()
                for frame in reader:
                    row=analyzer.process(frame)
                    row={key:None if isinstance(value,float) and not math.isfinite(value) else value
                         for key,value in row.items()}
                    writer.writerow(row);summary.add(row)
                csv_stream.flush()
                result=summary.result(reader)
                if plot_path:
                    try:
                        from src.analysis.plotting import plot_motion
                        result['plot']=plot_motion(csv_target,plot_path,summary.frames)
                    except Exception:
                        result['plot']={'status':'unavailable_or_failed','reason':'optional_plot_failed'}
                json_stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return result
    except BaseException:
        # Never leave partial outputs looking like completed analysis results.
        for path in created:path.unlink(missing_ok=True)
        raise
