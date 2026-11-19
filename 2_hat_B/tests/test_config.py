from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, EMOTION_CLASSES, load_config


def write_config(tmp_path, config):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    return path


def test_defaults_are_safe_and_dataset_independent():
    config = load_config()
    assert config['project']['classes'] == list(EMOTION_CLASSES)
    assert config['data'] == {'root': None, 'manifest': None}
    assert config['runtime']['batch_size'] == 1
    assert config['runtime']['num_workers'] == 0
    assert config['models']['pose']['backend'] is None
    assert config['models']['pose']['keypoint_schema'] is None
    for name in ['detection', 'segmentation', 'pose']:
        assert config['models'][name]['weights'] is None


def test_default_path_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config()['project']['name'] == 'canine-multimodal-emotion'


def test_load_custom_config(tmp_path):
    config = load_config()
    config['runtime']['device'] = 'cpu'
    assert load_config(write_config(tmp_path, config))['runtime']['device'] == 'cpu'


def test_calls_do_not_share_mutable_state():
    config = load_config()
    config['project']['classes'].clear()
    assert load_config()['project']['classes'] == list(EMOTION_CLASSES)


@pytest.mark.parametrize('text', ['', '[]', 'runtime: [', '!!python/object:builtins.object {}'])
def test_invalid_yaml_and_top_level(tmp_path, text):
    path = tmp_path / 'bad.yaml'
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / 'missing.yaml')


@pytest.mark.parametrize('section,key,value', [
    ('runtime', 'device', 'tpu'), ('runtime', 'batch_size', 0),
    ('runtime', 'batch_size', True), ('runtime', 'num_workers', -1),
    ('project', 'classes', ['happy']),
])
def test_invalid_values(tmp_path, section, key, value):
    config = load_config()
    config[section][key] = value
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, config))


@pytest.mark.parametrize('section', ['project', 'runtime', 'data', 'preprocessing', 'models'])
def test_missing_section(tmp_path, section):
    config = load_config()
    del config[section]
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, config))


@pytest.mark.parametrize('model,key,value', [
    ('detection', 'architecture', 'yolo11'),
    ('segmentation', 'architecture', 'yolo11-seg'),
    ('pose', 'species', 'human'),
])
def test_model_contract(tmp_path, model, key, value):
    config = load_config()
    config['models'][model][key] = value
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, config))
