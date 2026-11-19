"""Short-lived box association and confidence-aware pose smoothing for previews."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    left = np.maximum(a[:2], b[:2])
    right = np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(0, right-left))
    union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
    return float(intersection/union) if union > 0 else 0.


@dataclass
class Track:
    identifier: int
    bbox: np.ndarray
    xy: np.ndarray
    confidence: np.ndarray
    timestamp: float


class PreviewTracker:
    """Local track numbers, not recognized dog identities. Never bridge long gaps."""
    def __init__(self, confidence=.4, max_gap=.4):
        self.confidence = confidence
        self.max_gap = max_gap
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, poses, timestamp):
        if len(detections) != len(poses):
            raise ValueError("Detection and pose counts differ")
        self.tracks = {key: value for key, value in self.tracks.items()
                       if 0 <= timestamp-value.timestamp <= self.max_gap}
        previous = list(self.tracks.values())
        matches = {}
        if previous and detections:
            costs = np.array([[1-iou(track.bbox, np.asarray(d.xyxy)) for d in detections] for track in previous])
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] <= .85:
                    matches[column] = previous[row]
        visible = []
        for index, (detection, pose) in enumerate(zip(detections, poses)):
            box = np.asarray(detection.xyxy, dtype=float)
            xy = np.asarray(pose.xy, dtype=float).copy()
            scores = np.asarray(pose.confidence, dtype=float).copy()
            margin = .08*(box[2:]-box[:2])
            valid = np.isfinite(xy).all(axis=1) & np.isfinite(scores) & (scores >= self.confidence) & (scores <= 1)
            valid &= ((xy >= box[:2]-margin) & (xy <= box[2:]+margin)).all(axis=1)
            old = matches.get(index)
            identifier = old.identifier if old is not None else self.next_id
            if old is None:
                self.next_id += 1
            else:
                scale = np.linalg.norm(box[2:]-box[:2])
                common = valid & np.isfinite(old.xy).all(axis=1)
                # Compensate the dog's box translation before measuring jitter.
                translation = (box[:2]+box[2:]-old.bbox[:2]-old.bbox[2:])/2
                predicted = old.xy+translation
                step = np.linalg.norm(xy-predicted, axis=1)/max(scale, 1)
                # A teleporting keypoint is hidden for this frame, not held stale.
                rejected = common & (step > .28)
                valid[rejected] = False
                smooth = common & ~rejected
                alpha = np.where(step > .05, .9, .65)
                xy[smooth] = alpha[smooth, None]*xy[smooth]+(1-alpha[smooth, None])*predicted[smooth]
            xy[~valid] = np.nan
            scores[~valid] = 0
            track = Track(identifier, box, xy, scores, timestamp)
            self.tracks[identifier] = track
            visible.append(track)
        return visible
