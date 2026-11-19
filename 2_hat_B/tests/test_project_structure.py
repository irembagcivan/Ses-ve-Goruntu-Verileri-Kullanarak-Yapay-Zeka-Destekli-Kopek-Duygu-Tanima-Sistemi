"""Scaffold checks that require no datasets, weights or ML dependencies."""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('folder', [
    'src/preprocessing', 'src/pose_estimation', 'src/models', 'src/evaluation',
    'configs', 'tests', 'notebooks', 'experiments', 'results',
])
def test_required_folders(folder):
    assert (ROOT / folder).is_dir()


@pytest.mark.parametrize('module', [
    'src.config', 'src.preprocessing.interfaces', 'src.pose_estimation.interfaces',
    'src.models.interfaces', 'src.evaluation.interfaces',
])
def test_module_import(module):
    assert importlib.import_module(module)


def test_imports_do_not_load_heavy_dependencies():
    script = '''
import sys
import src.config
import src.preprocessing.interfaces
import src.pose_estimation.interfaces
import src.models.interfaces
import src.evaluation.interfaces
assert not ({'torch', 'ultralytics', 'cv2', 'numpy', 'pandas', 'librosa', 'sklearn'} & sys.modules.keys())
'''
    subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT, check=True)


@pytest.mark.parametrize('path', [
    'data/raw/example.csv', 'datasets/labels.json', 'clip.MP4', 'bark.wav',
    'model.pt', '.venv/lib/example.py', 'src/__pycache__/file.pyc',
    'results/metrics.csv', 'experiments/trial/output.txt',
])
def test_generated_assets_are_ignored(path):
    result = subprocess.run(['git', 'check-ignore', '--no-index', '-q', path], cwd=ROOT)
    assert result.returncode == 0


def test_source_and_placeholders_are_not_ignored():
    for path in ['src/config.py', 'configs/default.yaml', 'results/.gitkeep', 'experiments/.gitkeep']:
        result = subprocess.run(['git', 'check-ignore', '--no-index', '-q', path], cwd=ROOT)
        assert result.returncode == 1
