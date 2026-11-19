"""Known synthetic motion, no inference/model/video requirements."""
import copy
import csv
import json
import math
import subprocess
import sys

import pytest

from src.analysis.features import MotionAnalyzer, MotionSettings
from src.analysis.reader import JsonlFrames, parse_frame
from src.analysis.regions import model_names
from src.analysis.report import analyze_file
from src.analysis.run_motion import main

NAMES=model_names()


def record(index,stamp=None,points=None,box=(0,0,60,80),dogs=1):
    if points is None:points={'nose':(10.,10.)}
    keypoints=[{'index':NAMES.index(name),'name':name,'x':xy[0],'y':xy[1],
                'confidence':.9,'in_frame':True,'confidence_valid':True} for name,xy in points.items()]
    dog={'detection_index':0,'bbox_xyxy':list(box),'detection_confidence':.95,'keypoints':keypoints}
    return {'frame_index':index,'timestamp_seconds':float(index) if stamp is None else stamp,
            'timestamp_source':'frame_index_div_fps','frame_width':200,'frame_height':200,
            'dogs':[copy.deepcopy(dog) for i in range(dogs)]}


def rows(records,settings=None):
    settings=settings or MotionSettings()
    analyzer=MotionAnalyzer(NAMES,settings)
    return [analyzer.process(parse_frame(data,i+1,NAMES,settings.confidence,settings.detection_confidence))
            for i,data in enumerate(records)]


def test_stationary_is_zero_not_missing():
    result=rows([record(0),record(1)])
    assert result[0]['global_mean_speed_px_s'] is None
    assert result[1]['global_mean_speed_px_s']==0
    assert result[1]['valid_keypoints']==1


def test_known_distance_dt_and_normalization():
    result=rows([record(0,0),record(1,.5,{'nose':(13.,14.)})])[1]
    assert result['global_mean_displacement_px']==5
    assert result['global_mean_speed_px_s']==10
    assert result['normalization_scale_px']==100
    assert result['global_mean_displacement_normalized']==.05
    assert result['global_mean_speed_normalized_s']==.1


def test_normalization_can_be_disabled():
    result=rows([record(0),record(1,points={'nose':(13.,14.)})],MotionSettings(normalization='none'))[1]
    assert result['global_mean_displacement_px']==5
    assert result['global_mean_displacement_normalized'] is None


def test_resolution_scaled_videos_have_equal_bbox_normalized_motion():
    a=[record(0),record(1,points={'nose':(13.,14.)})]
    b=copy.deepcopy(a)
    for frame in b:
        frame['frame_width']=frame['frame_height']=400
        frame['dogs'][0]['bbox_xyxy']=[0,0,120,160]
        for point in frame['dogs'][0]['keypoints']:point['x']*=2;point['y']*=2
    x,y=rows(a)[1],rows(b)[1]
    assert y['global_mean_speed_px_s']==2*x['global_mean_speed_px_s']
    assert y['global_mean_speed_normalized_s']==x['global_mean_speed_normalized_s']


@pytest.mark.parametrize('field,value',[('confidence',.1),('x',None),('x',math.nan),('y',math.inf),
    ('x',-1),('x',200),('in_frame',False),('confidence_valid',False)])
def test_unreliable_point_is_missing_not_zero(field,value):
    a,b=record(0),record(1)
    b['dogs'][0]['keypoints'][0][field]=value
    result=rows([a,b])[1]
    assert result['global_mean_speed_px_s'] is None
    assert result['status']=='no_valid_keypoints'


def test_missing_keypoint_and_duplicates_are_not_guessed():
    a,b=record(0),record(1)
    b['dogs'][0]['keypoints']=[]
    assert rows([a,b])[1]['valid_keypoints']==0
    b=record(1);b['dogs'][0]['keypoints']*=2
    assert rows([a,b])[1]['valid_keypoints']==0


def test_wrong_index_name_mapping_is_rejected():
    data=record(0);data['dogs'][0]['keypoints'][0]['index']=5
    frame=parse_frame(data,1,NAMES,.5,0)
    assert frame.points=={} and frame.point_errors==1


@pytest.mark.parametrize('count,reason',[(0,'no_dog'),(2,'multiple_dogs')])
def test_absent_and_multiple_dogs_break_chain(count,reason):
    data=[record(0),record(1,dogs=count),record(2),record(3)]
    result=rows(data)
    assert result[1]['status']==reason
    assert result[1]['global_mean_speed_px_s'] is None
    assert result[2]['status']=='baseline'
    assert result[3]['global_mean_speed_px_s']==0


@pytest.mark.parametrize('stamp',[0.,-.1])
def test_zero_or_negative_pair_dt_is_not_divided(stamp):
    # Absolute timestamps remain nonnegative; dt is zero or negative.
    a,b=record(0,1.),record(1,1.+stamp)
    result=rows([a,b])[1]
    assert result['status']=='nonpositive_dt' and result['global_mean_speed_px_s'] is None


def test_missing_frame_is_not_bridged():
    result=rows([record(0),record(2),record(3)])
    assert result[1]['status']=='frame_gap'
    assert result[2]['global_mean_speed_px_s']==0


def test_resolution_change_breaks_pair():
    a,b=record(0),record(1);b['frame_width']=300
    assert rows([a,b])[1]['status']=='resolution_change'


def test_jump_flag_is_not_real_motion_or_zero():
    result=rows([record(0),record(1,points={'nose':(90.,10.)})])[1]
    assert result['status']=='all_points_flagged'
    assert result['jump_keypoints']==1 and result['jump_keypoint_names']=='nose'
    assert result['flagged_max_displacement_px']==80
    assert result['global_mean_speed_px_s'] is None


def test_partial_jump_excludes_only_flagged_point():
    a=record(0,points={'nose':(10.,10.),'left_eye':(20.,20.)})
    b=record(1,points={'nose':(90.,10.),'left_eye':(23.,24.)})
    result=rows([a,b])[1]
    assert result['valid_keypoints']==1 and result['jump_keypoints']==1
    assert result['global_mean_displacement_px']==5


def test_regions_absolute_relative_and_tail_vector_translation():
    points={'nose':(10.,10.),'left_eye':(15.,10.),'back_base':(30.,40.),
            'back_middle':(35.,40.),'tail_base':(20.,20.),'tail_end':(30.,20.)}
    translated={name:(x+3,y+4) for name,(x,y) in points.items()}
    result=rows([record(0,0,points),record(1,.5,translated,(3,4,63,84))])[1]
    for region in ('head','body','tail'):
        assert result[f'{region}_absolute_mean_speed_px_s']==10
        assert result[f'{region}_relative_mean_speed_px_s']==0
    assert result['tail_vector_displacement_px']==0


def test_tail_vector_change_requires_both_endpoints():
    a=record(0,points={'tail_base':(20.,20.),'tail_end':(30.,20.)})
    b=record(1,points={'tail_base':(23.,20.),'tail_end':(35.,20.)})
    assert rows([a,b])[1]['tail_vector_displacement_px']==2
    b['dogs'][0]['keypoints'].pop()
    assert rows([a,b])[1]['tail_vector_displacement_px'] is None


def test_region_minimum_point_coverage():
    result=rows([record(0),record(1)])[1]
    assert result['head_valid_points']==1
    assert result['head_absolute_mean_speed_px_s'] is None


def test_reader_malformed_nan_duplicate_and_frame_order(tmp_path):
    source=tmp_path/'pose.jsonl'
    nan=record(4);nan['dogs'][0]['keypoints'][0]['x']=math.nan
    source.write_text('\n'+json.dumps(record(0))+'\nnot-json\n'+json.dumps(record(1))+'\n'+
                      json.dumps(record(1))+'\n'+json.dumps(nan)+'\n{"frame_index":5,"frame_index":6}\n')
    reader=JsonlFrames(source,NAMES)
    frames=list(reader)
    assert reader.blank_lines==1 and reader.nonstandard_constants==1
    assert frames[1].reason=='malformed_json'
    assert frames[3].reason=='nonincreasing_frame_index'
    assert frames[4].points=={}
    assert frames[5].reason=='malformed_json'
    analyzer=MotionAnalyzer(NAMES,MotionSettings())
    result=[analyzer.process(frame) for frame in frames]
    assert result[2]['status']=='baseline'


def test_csv_json_summary_missing_values_and_input_hash(tmp_path):
    import hashlib
    source=tmp_path/'private-user-pose.jsonl';output=tmp_path/'motion.csv';summary=tmp_path/'summary.json'
    source.write_text('\n'.join(json.dumps(data) for data in [record(0),record(1,points={'nose':(13.,14.)}),record(2,dogs=0)])+'\n')
    result=analyze_file(source,output,summary,MotionSettings())
    with output.open(newline='') as stream:data=list(csv.DictReader(stream))
    assert len(data)==3 and data[0]['global_mean_speed_px_s']==''
    assert float(data[1]['global_mean_speed_px_s'])==5
    assert data[2]['global_mean_speed_px_s']==''
    saved=json.loads(summary.read_text())
    assert saved==result and saved['total_frames']==3 and saved['valid_analysis_frames']==1
    assert saved['missing_motion_rate']==pytest.approx(2/3)
    assert saved['metrics']['global_mean_speed_px_s']['mean']==5
    assert saved['metrics']['global_mean_speed_px_s']['max']==5
    assert saved['input_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert 'private-user' not in summary.read_text() and str(tmp_path) not in summary.read_text()


def test_plot_failure_preserves_primary_outputs(tmp_path,monkeypatch):
    from src.analysis import plotting
    def fail(*args):raise ImportError('optional package absent')
    monkeypatch.setattr(plotting,'plot_motion',fail)
    source=tmp_path/'pose.jsonl';source.write_text(json.dumps(record(0))+'\n'+json.dumps(record(1))+'\n')
    output,summary=tmp_path/'out.csv',tmp_path/'out.json'
    result=analyze_file(source,output,summary,MotionSettings(),tmp_path/'optional.png')
    assert result['plot']['status']=='unavailable_or_failed'
    assert output.is_file() and summary.is_file()


def test_cli_existing_output_protection_and_no_valid_motion(tmp_path):
    source=tmp_path/'pose.jsonl';source.write_text(json.dumps(record(0))+'\n')
    output,summary=tmp_path/'out.csv',tmp_path/'out.json'
    args=['--input',str(source),'--output',str(output),'--summary',str(summary)]
    assert main(args)==2
    assert json.loads(summary.read_text())['status']=='no_valid_motion'
    before=output.read_bytes()
    with pytest.raises(SystemExit):main(args)
    assert output.read_bytes()==before


def test_analysis_help_has_no_ml_or_plot_imports():
    script='''
import sys
from src.analysis.run_motion import main
try:main(['--help'])
except SystemExit as exc:assert exc.code==0
assert not ({'torch','deeplabcut','ultralytics','cv2','numpy','matplotlib'} & sys.modules.keys())
'''
    subprocess.run([sys.executable,'-B','-c',script],check=True,capture_output=True)


def test_analysis_is_reproducible_for_same_bytes_and_settings(tmp_path):
    source=tmp_path/'pose.jsonl'
    source.write_text(json.dumps(record(0))+'\n'+json.dumps(record(1,points={'nose':(13.,14.)}))+'\n')
    outputs=[]
    for index in (1,2):
        csv_path,json_path=tmp_path/f'out{index}.csv',tmp_path/f'out{index}.json'
        analyze_file(source,csv_path,json_path,MotionSettings())
        outputs.append((csv_path.read_bytes(),json_path.read_bytes()))
    assert outputs[0]==outputs[1]


def test_empty_file_has_unknown_rates_not_fake_zero_motion(tmp_path):
    source=tmp_path/'empty.jsonl';source.write_text('\n')
    result=analyze_file(source,tmp_path/'out.csv',tmp_path/'out.json',MotionSettings())
    assert result['total_frames']==0 and result['blank_lines_skipped']==1
    assert result['missing_motion_rate'] is None
    assert result['metrics']['global_mean_speed_px_s']['mean'] is None


def test_existing_optional_png_does_not_block_csv_and_json(tmp_path):
    source=tmp_path/'pose.jsonl'
    source.write_text(json.dumps(record(0))+'\n'+json.dumps(record(1))+'\n')
    plot=tmp_path/'keep.png';plot.write_bytes(b'preserve existing image')
    result=analyze_file(source,tmp_path/'out.csv',tmp_path/'out.json',MotionSettings(),plot)
    assert result['plot']['status']=='output_exists'
    assert plot.read_bytes()==b'preserve existing image'
    assert (tmp_path/'out.csv').is_file() and (tmp_path/'out.json').is_file()
