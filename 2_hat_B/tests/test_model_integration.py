"""Synthetic integration contracts; no claim of real-model inference."""
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.pose_estimation import backend
from src.pose_estimation.backend import (
    DeepLabCutRTMPose, VerifiedRunnerAdapter, load_model_configuration,
    pose_from_dlc_output, validate_checkpoint_path,
)
from src.pose_estimation.device import select_device
from src.pose_estimation.metadata import MetadataError, load_superanimal_metadata
from src.pose_estimation.interfaces import DogPose
from src.preprocessing.image import validate_image_paths, process_image
from src.pose_estimation.run_image import main

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / 'configs/pose/rtmpose_s.yaml'


def test_official_metadata_order_and_empty_skeleton():
    evidence, schema = load_superanimal_metadata()
    assert evidence['framework_version'] == '3.0.2'
    assert len(schema.names) == 39
    assert schema.names[:3] == ('nose', 'upper_jaw', 'lower_jaw')
    assert schema.names[22:24] == ('tail_base', 'tail_end')
    assert schema.names[-1] == 'body_middle_left'
    assert schema.edges == ()  # No inferred edges substituted.
    assert evidence['checkpoint_sha256'] is None
    assert evidence['weights_license'] is None


def test_official_model_config_external_boxes_and_safe_loading():
    config, schema = load_model_configuration(MODEL_CONFIG)
    assert config['method'] == 'td'
    assert config['net_type'] == 'rtmpose_s'
    assert config['data']['colormode'] == 'RGB'
    assert config['metadata']['bodyparts'] == list(schema.names)
    assert config['runner']['load_weights_only'] is True
    assert 'detector' not in config
    assert config['model']['heads']['bodypart']['input_size'] == [256,256]


def test_modified_model_configuration_rejected(tmp_path):
    changed = tmp_path / 'changed.yaml'
    changed.write_bytes(MODEL_CONFIG.read_bytes() + b'\n# modified\n')
    with pytest.raises(MetadataError, match='provenance'):
        load_model_configuration(changed)


@pytest.mark.parametrize('name,content,error', [
    ('absent.pt',None,FileNotFoundError),('empty.pt',b'',ValueError),
    ('wrong.bin',b'fixture',FileNotFoundError),
])
def test_checkpoint_path_validation(tmp_path,name,content,error):
    p=tmp_path/name
    if content is not None:p.write_bytes(content)
    with pytest.raises(error):validate_checkpoint_path(p)


def test_checkpoint_hash_mismatch_rejected_before_dlc_import(tmp_path):
    p=tmp_path/'fixture.pt';p.write_bytes(b'not real weights')
    with pytest.raises(ValueError,match='SHA-256 mismatch'):
        DeepLabCutRTMPose(p,MODEL_CONFIG,checkpoint_sha256='0'*64)


def test_conversion_to_existing_pose_result():
    _,schema=load_superanimal_metadata()
    points=np.zeros((1,39,3));points[0,0]=[12,34,.75]
    pose=pose_from_dlc_output([{'bodyparts':points}],schema)
    assert isinstance(pose,DogPose)
    assert pose.keypoint_names==schema.names
    assert pose.xy[0]==(12,34) and pose.confidence[0]==.75


@pytest.mark.parametrize('result', [[],[{}],[{'bodyparts':np.zeros((1,17,3))}],
    [{'bodyparts':np.zeros((2,39,3))}],[{'bodyparts':np.zeros((39,3))}]])
def test_bad_output_shape_rejected(result):
    _,schema=load_superanimal_metadata()
    with pytest.raises(ValueError):pose_from_dlc_output(result,schema)


def test_runner_rgb_xywh_batch_contract():
    _,schema=load_superanimal_metadata()
    seen=[]
    runner=NS(inference=lambda **kwargs: seen.append(kwargs) or [{'bodyparts':np.zeros((1,39,3))}])
    adapter=VerifiedRunnerAdapter(runner,schema)
    crop=np.zeros((10,20,3),dtype=np.uint8);crop[:,:,0]=200
    xy,scores=adapter.predict_crop(crop)
    image,context=seen[0]['images'][0]
    assert image[0,0].tolist()==[0,0,200]
    assert context['bboxes'].tolist()==[[0,0,20,10]]
    assert len(xy)==len(scores)==39


def test_runner_initialized_once_and_reused(tmp_path,monkeypatch):
    p=tmp_path/'fixture.pt';p.write_bytes(b'synthetic fixture, not checkpoint weights')
    monkeypatch.setattr(backend.packages,'version',lambda name:'3.0.2')
    calls=[];predictions=[]
    runner=NS(inference=lambda **kwargs: predictions.append(kwargs) or [{'bodyparts':np.zeros((1,39,3))}])
    def build(**kwargs):calls.append(kwargs);return runner
    # Explicit fake module at the verified official import path.
    monkeypatch.setitem(sys.modules,'deeplabcut.pose_estimation_pytorch.apis.utils',NS(get_pose_inference_runner=build))
    with pytest.warns(UserWarning,match='skeleton is empty'):
        model=DeepLabCutRTMPose(p,MODEL_CONFIG)
    model.predict_crop(np.zeros((10,20,3),dtype=np.uint8))
    model.predict_crop(np.zeros((10,20,3),dtype=np.uint8))
    assert len(calls)==1 and len(predictions)==2
    assert calls[0]['batch_size']==calls[0]['max_individuals']==1
    assert calls[0]['device']=='cpu'
    assert calls[0]['snapshot_path']==str(p)


def test_unverified_dlc_version_rejected(tmp_path,monkeypatch):
    p=tmp_path/'fixture.pt';p.write_bytes(b'fixture')
    monkeypatch.setattr(backend.packages,'version',lambda name:'2.3.0')
    with pytest.raises(RuntimeError,match='3.0.2'):DeepLabCutRTMPose(p,MODEL_CONFIG)


def test_cpu_does_not_import_torch(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',None)
    assert select_device('cpu')=='cpu'


@pytest.mark.parametrize('suffix',['.jpg','.JPEG','.png'])
def test_image_paths(tmp_path,suffix):
    source=tmp_path/('input'+suffix);source.touch()
    assert validate_image_paths(source,tmp_path/'out.jpg')[0]==source


def test_image_invalid_paths(tmp_path):
    source=tmp_path/'in.jpg';source.touch()
    with pytest.raises(ValueError):validate_image_paths(source,source)
    with pytest.raises(ValueError):validate_image_paths(source,tmp_path/'out.mp4')
    with pytest.raises(FileNotFoundError):validate_image_paths(tmp_path/'missing.png',tmp_path/'out.jpg')
    output=tmp_path/'out.jpg';output.touch()
    with pytest.raises(FileExistsError):validate_image_paths(source,output)


def test_image_decode_failure(tmp_path):
    p=tmp_path/'broken.jpg';p.write_bytes(b'not jpeg')
    cv=NS(IMREAD_COLOR=1,imread=lambda *args:None)
    with pytest.raises(ValueError,match='decoded'):
        process_image(p,tmp_path/'out.jpg',NS(),NS(),cv_module=cv)


def test_image_cli_missing_checkpoint_no_fake_output(tmp_path,capsys):
    p=tmp_path/'in.jpg';p.touch();out=tmp_path/'out.jpg'
    with pytest.raises(SystemExit) as exc:
        main(['--input',str(p),'--output',str(out),'--detector-weights',str(tmp_path/'yolov8.pt'),
              '--pose-weights',str(tmp_path/'missing.pt')])
    assert exc.value.code==2
    assert 'checkpoint' in capsys.readouterr().err
    assert not out.exists()


def test_image_help_is_lightweight():
    import subprocess
    script='''
import sys
from src.pose_estimation.run_image import main
try: main(['--help'])
except SystemExit as exc: assert exc.code == 0
assert not ({'torch','cv2','numpy','deeplabcut','ultralytics'} & sys.modules.keys())
'''
    subprocess.run([sys.executable,'-B','-c',script],check=True,capture_output=True)


def test_synthetic_image_io_and_overlay_with_real_opencv(tmp_path):
    """Real PNG/JPEG I/O + drawing; detector/pose remain synthetic fixtures."""
    import cv2
    from src.pose_estimation.adapter import DogPoseAdapter
    from src.preprocessing.interfaces import DogDetection
    _,schema=load_superanimal_metadata()
    source=tmp_path/'synthetic.png';output=tmp_path/'annotated.jpg'
    assert cv2.imwrite(str(source),np.zeros((64,64,3),dtype=np.uint8))
    def predict(crop):
        points=[(10,10)]*39
        scores=[0.0]*39;scores[0]=.95
        return points,scores
    fake_backend=NS(schema=schema,predict_crop=predict)
    fake_detector=NS(detect=lambda frame:[DogDetection((5,5,50,50),.9)])
    report=process_image(source,output,fake_detector,DogPoseAdapter(fake_backend,margin=0))
    decoded=cv2.imread(str(output))
    assert decoded is not None and decoded.shape==(64,64,3)
    assert decoded.sum()>0
    assert report['dogs']==1 and report['skeleton_edges']==0


def test_empty_checkpoint_cli_error_before_heavy_import(tmp_path,capsys):
    source=tmp_path/'in.jpg';source.touch()
    checkpoint=tmp_path/'empty.pt';checkpoint.touch()
    with pytest.raises(SystemExit) as exc:
        main(['--input',str(source),'--output',str(tmp_path/'out.jpg'),
              '--detector-weights','missing.pt','--pose-weights',str(checkpoint)])
    assert exc.value.code==2
    assert 'empty' in capsys.readouterr().err


def test_malformed_configuration_error(tmp_path):
    checkpoint=tmp_path/'fixture.pt';checkpoint.write_bytes(b'fixture')
    config=tmp_path/'invalid.yaml';config.write_text('invalid: [')
    with pytest.raises(MetadataError,match='configuration'):
        DeepLabCutRTMPose(checkpoint,config)


def test_bad_checkpoint_hash_format(tmp_path):
    checkpoint=tmp_path/'fixture.pt';checkpoint.write_bytes(b'fixture')
    with pytest.raises(ValueError,match='64 hex'):
        DeepLabCutRTMPose(checkpoint,MODEL_CONFIG,checkpoint_sha256='not-a-hash')


def test_new_mac_environment_is_ignored():
    import subprocess
    assert subprocess.run(['git','check-ignore','--no-index','-q','.venv-pose/lib/test.py'],cwd=ROOT).returncode==0
