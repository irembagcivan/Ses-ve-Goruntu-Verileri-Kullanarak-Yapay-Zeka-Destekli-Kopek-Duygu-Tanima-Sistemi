"""Sequential multi-dog crop adapter; does not assign persistent track IDs."""
from src.pose_estimation.geometry import padded_crop
from src.pose_estimation.interfaces import DogPose


class DogPoseAdapter:
    def __init__(self, backend, margin=0.15):
        if not 0 <= margin <= 0.5:
            raise ValueError("Margin must be in [0, 0.5]")
        self.backend, self.margin = backend, margin
        self.schema = backend.schema

    def estimate(self, frame, dogs):
        height, width = frame.pixels.shape[:2]
        poses = []
        for dog in dogs:
            transform = padded_crop(dog.xyxy, width, height, self.margin)
            xy, confidence = self.backend.predict_crop(transform.extract(frame.pixels))
            if len(xy) != len(self.schema.names) or len(confidence) != len(xy):
                raise ValueError("Pose output does not match verified keypoint schema")
            poses.append(DogPose(self.schema.names, transform.to_original(xy), tuple(confidence)))
        return poses
