"""Contracts only; coordinates refer to the original frame in pixels."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, Sequence


@dataclass(frozen=True)
class VideoFrame:
    index: int
    timestamp_seconds: float
    pixels: object  # Backend-specific image array; no eager NumPy dependency.


@dataclass(frozen=True)
class DogDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class DogSegmentation:
    detection: DogDetection
    mask: object  # Boolean mask aligned with the original frame.


class VideoProcessor(Protocol):
    def iter_frames(self, path: Path) -> Iterator[VideoFrame]: ...


class AudioProcessor(Protocol):
    def extract(self, video_path: Path, output_path: Path) -> Path: ...


class DogDetector(Protocol):
    """Implement using YOLOv8 detection and explicit dog-class filtering."""
    def detect(self, frame: VideoFrame) -> Sequence[DogDetection]: ...


class DogSegmenter(Protocol):
    """Implement using YOLOv8 segmentation and explicit dog-class filtering."""
    def segment(self, frame: VideoFrame) -> Sequence[DogSegmentation]: ...
