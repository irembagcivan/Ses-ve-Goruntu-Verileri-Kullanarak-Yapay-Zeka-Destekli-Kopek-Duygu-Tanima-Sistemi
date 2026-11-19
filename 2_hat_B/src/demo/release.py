"""Resolve the active local model/gallery without replacing historical results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def active_release():
    path = ROOT/'configs/demo/active_release.json'
    if path.exists():
        return json.loads(path.read_text())
    return dict(version='legacy-v12', model='results/demo_v9/neural_fusion.pt',
                gallery='results/demo_v11/public/canine_demo_v11.html')
