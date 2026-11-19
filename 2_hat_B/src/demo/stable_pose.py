"""Strict skeleton display using fresh pose and verified image motion."""
from collections import deque
import cv2
import numpy as np

from src.demo.tracking import PreviewTracker


class FlowPreviewTracker(PreviewTracker):
    def __init__(self):
        super().__init__(confidence=.45, max_gap=.25)
        self.previous_gray = None
        self.previous_timestamp = None
        self.current_gray = None
        self.lengths = {}

    def set_frame(self, pixels):
        self.current_gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)

    def update(self, detections, poses, timestamp):
        old = dict(self.tracks)
        flows = {}
        current = self.current_gray
        already_propagated=getattr(self,'already_propagated',False)
        if not already_propagated and current is not None and self.previous_gray is not None and current.shape == self.previous_gray.shape:
            difference = np.mean(cv2.absdiff(current, self.previous_gray))
            # Texture motion can change many pixels while preserving the scene.
            # Require a large histogram change as well before dropping tracks.
            hist_now = np.histogram(current,bins=32,range=(0,256))[0]/current.size
            hist_old = np.histogram(self.previous_gray,bins=32,range=(0,256))[0]/current.size
            if difference > 45 and np.abs(hist_now-hist_old).sum() > .8:
                self.tracks.clear()
                old.clear()
                self.lengths.clear()
            for key, track in old.items():
                if track.timestamp != self.previous_timestamp:
                    continue
                indices = np.where(np.isfinite(track.xy).all(axis=1))[0]
                if not len(indices):
                    continue
                points = track.xy[indices].astype(np.float32).reshape(-1,1,2)
                levels = min(3,max(0,int(np.log2(min(current.shape)/40))))
                moved, status, error = cv2.calcOpticalFlowPyrLK(self.previous_gray, current, points, None,
                        winSize=(21,21), maxLevel=levels,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .02))
                if moved is None:
                    continue
                returned, back_status, _ = cv2.calcOpticalFlowPyrLK(current, self.previous_gray, moved, None,
                        winSize=(21,21), maxLevel=levels,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .02))
                if returned is None:
                    continue
                reliable = status.ravel().astype(bool) & back_status.ravel().astype(bool)
                reliable &= np.linalg.norm(returned.reshape(-1,2)-points.reshape(-1,2),axis=1) < 1.5
                reliable &= error.ravel() < 22
                flows[key] = {int(i): xy for i, xy, ok in zip(indices,moved.reshape(-1,2),reliable) if ok}
        tracks = super().update(detections, poses, timestamp)
        for track in tracks:
            old_track = old.get(track.identifier)
            scale = max(1, np.linalg.norm(track.bbox[2:]-track.bbox[:2]))
            for index in range(len(track.xy)):
                if not np.isfinite(track.xy[index]).all():
                    continue
                previously_visible = old_track is not None and np.isfinite(old_track.xy[index]).all()
                if not previously_visible and track.confidence[index] < .58:
                    track.xy[index] = np.nan
                    track.confidence[index] = 0
                    continue
                motion = flows.get(track.identifier, {}).get(index)
                if motion is not None:
                    disagreement = np.linalg.norm(track.xy[index]-motion)/scale
                    if disagreement > .10:
                        track.xy[index] = np.nan
                        track.confidence[index] = 0
                    else:
                        track.xy[index] = .3*track.xy[index]+.7*motion
                elif not already_propagated and old_track is not None and previously_visible:
                    translation = (track.bbox[:2]+track.bbox[2:]-old_track.bbox[:2]-old_track.bbox[2:])/2
                    residual = np.linalg.norm(track.xy[index]-old_track.xy[index]-translation)/scale
                    if residual > .07:
                        track.xy[index] = np.nan
                        track.confidence[index] = 0
        self.previous_gray = current
        self.previous_timestamp = timestamp
        return tracks

    def edge_allowed(self, track, a, b, names):
        if min(track.confidence[a], track.confidence[b]) < .58:
            return False
        scale = max(1, np.linalg.norm(track.bbox[2:]-track.bbox[:2]))
        length = np.linalg.norm(track.xy[a]-track.xy[b])/scale
        labels = (names[a], names[b])
        face = any(any(token in name for token in ("eye", "ear", "nose")) for name in labels)
        maximum = .19 if face else .40 if "tail" in labels[0] else .34
        if not .008 < length < maximum:
            return False
        key = (track.identifier, a, b)
        history = self.lengths.setdefault(key, deque(maxlen=7))
        if len(history) >= 3:
            reference = float(np.median(history))
            if abs(length-reference) > max(.035, .45*reference):
                history.clear()
                return False
        history.append(length)
        return len(history) >= 2
