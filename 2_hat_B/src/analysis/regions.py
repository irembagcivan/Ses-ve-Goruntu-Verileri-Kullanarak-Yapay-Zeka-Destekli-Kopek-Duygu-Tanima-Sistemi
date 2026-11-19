"""Named surface-marker groups; these are not validated joint/bone definitions."""
import json
from pathlib import Path

HEAD = ('nose','upper_jaw','lower_jaw','mouth_end_right','mouth_end_left',
        'right_eye','right_earbase','right_earend','left_eye','left_earbase','left_earend')
BODY = ('neck_base','neck_end','throat_base','throat_end','back_base','back_end',
        'back_middle','belly_bottom','body_middle_right','body_middle_left')
TAIL = ('tail_base','tail_end')
REGIONS = {'head':HEAD,'body':BODY,'tail':TAIL}


def model_names():
    path=Path(__file__).resolve().parents[2]/'configs/superanimal_quadruped_rtmpose_s.metadata.json'
    names=tuple(json.loads(path.read_text(encoding='utf-8'))['metadata']['bodyparts'])
    if len(names)!=39 or len(set(names))!=39:
        raise ValueError('Expected the existing 39-point model schema')
    if any(name not in names for group in REGIONS.values() for name in group):
        raise ValueError('Region definitions do not match the model schema')
    return names


def analysis_names(names):
    # Preserve original names/indices in input; antler markers are not dog features.
    return tuple(name for name in names if 'antler' not in name)
