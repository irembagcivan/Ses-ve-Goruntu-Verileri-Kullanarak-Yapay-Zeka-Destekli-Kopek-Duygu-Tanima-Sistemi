"""Feature and classification contracts, without training implementations."""
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from src.pose_estimation.interfaces import DogPose
from src.preprocessing.interfaces import DogSegmentation, VideoFrame

FeatureVector = Sequence[float]


class VisualFeatureExtractor(Protocol):
    def extract(self, frames: Sequence[VideoFrame], masks: Sequence[DogSegmentation],
                poses: Sequence[DogPose]) -> FeatureVector: ...


class AudioFeatureExtractor(Protocol):
    def extract(self, audio_path: Path, sample_rate: int) -> FeatureVector: ...


class MultimodalClassifier(Protocol):
    """Return probabilities for Happy, Sad, Angry and Relaxed in a mapping."""
    def predict_proba(self, visual: FeatureVector, audio: FeatureVector) -> Mapping[str, float]: ...
