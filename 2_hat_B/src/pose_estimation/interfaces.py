"""Dog-specific contracts; no human pose fallback or default weights."""
from dataclasses import dataclass
from typing import Protocol, Sequence

from src.preprocessing.interfaces import DogDetection, VideoFrame


@dataclass(frozen=True)
class DogPose:
    keypoint_names: tuple[str, ...]
    xy: tuple[tuple[float, float], ...]
    confidence: tuple[float, ...]


class DogPoseEstimator(Protocol):
    """Backend and keypoint schema require validation for dogs before use."""
    def estimate(self, frame: VideoFrame, dogs: Sequence[DogDetection]) -> Sequence[DogPose]: ...
