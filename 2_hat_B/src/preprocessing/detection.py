"""Local YOLOv8 detection only; no implicit checkpoint download."""
from pathlib import Path
import math
from src.preprocessing.interfaces import DogDetection
from src.pose_estimation.geometry import xyxy_to_xywh


class YoloDogDetector:
    def __init__(self, weights=None, device="cpu", confidence=0.25, model=None):
        if not 0 <= confidence <= 1:
            raise ValueError("Detection confidence must be in [0, 1]")
        if model is None:
            if weights is None or not Path(weights).is_file():
                raise FileNotFoundError("Provide an existing local YOLOv8 detection checkpoint")
            from ultralytics import YOLO
            model = YOLO(str(Path(weights).resolve()), task="detect")
            # Official model YAML identifies v8 by its C2f backbone and Detect head.
            cfg = model.model.yaml
            layers = cfg.get("backbone", []) + cfg.get("head", [])
            if not any(layer[2] == "C2f" for layer in layers) or layers[-1][2] != "Detect":
                raise ValueError("Checkpoint is not a supported YOLOv8 detection architecture")
        if model.task != "detect":
            raise ValueError("A detection model is required")
        names = model.names
        entries = names.items() if isinstance(names, dict) else enumerate(names)
        ids = [int(i) for i, name in entries if str(name).strip().lower() == "dog"]
        if len(ids) != 1:
            raise ValueError("Model metadata must identify exactly one dog class")
        self.model, self.dog_id = model, ids[0]
        self.device, self.confidence = device, confidence

    def detect(self, frame):
        results = self.model.predict(source=frame.pixels, classes=[self.dog_id],
                                     conf=self.confidence, device=self.device,
                                     batch=1, verbose=False, save=False)
        dogs = []
        for result in results:
            if result.boxes is None:
                continue
            for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                       result.boxes.conf.cpu().tolist(),
                                       result.boxes.cls.cpu().tolist()):
                if int(cls) != self.dog_id or not math.isfinite(score) or score < self.confidence:
                    continue
                xyxy_to_xywh(box)
                dogs.append(DogDetection(tuple(map(float, box)), float(score)))
        return dogs
