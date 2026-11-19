"""Predict human-defined behaviors for a video using trained local models."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.demo.extract import audio_features, clean_json, digest, visual_features
from src.demo.train import emotion_rule


def predict(args):
    from src.preprocessing.detection import YoloDogDetector
    from src.pose_estimation.backend import DeepLabCutRTMPose
    from src.pose_estimation.adapter import DogPoseAdapter
    import torch
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError("Choose a new output path")
    bundle = joblib.load(args.model)
    signature = bundle["extraction_signature"]
    if digest(args.detector) != signature["detector_sha256"] or digest(args.pose) != signature["pose_sha256"]:
        raise ValueError("Checkpoints differ from feature extraction used for training")
    detector = YoloDogDetector(args.detector, "cpu", .25)
    adapter = DogPoseAdapter(DeepLabCutRTMPose(args.pose, device="cpu"))
    visual, quality, _ = visual_features(args.video, detector, adapter, signature["sample_fps"], signature["seconds"], signature["confidence"])
    audio, status = audio_features(args.video, signature["seconds"])
    frame = pd.DataFrame([{**visual, **audio}])
    predictions, scores, emotions = {}, {}, {}
    for modality, targets in bundle["models"].items():
        if (modality != "audio" and not visual["v_pose_available"]) or (modality != "visual" and not audio["a_available"]):
            predictions[modality] = {"status": "required_modality_missing"}
            continue
        predictions[modality], scores[modality] = {}, {}
        for target, entry in targets.items():
            x = frame.reindex(columns=entry["columns"])
            classifier = entry["model"]
            predictions[modality][target] = classifier.predict(x)[0]
            scores[modality][target] = dict(zip(classifier.classes_, map(float, classifier.predict_proba(x)[0])))
        emotions[modality] = emotion_rule(predictions[modality], scores[modality])
    result = clean_json({"predictions": predictions, "model_scores": scores, "emotion_rule": emotions,
                         "quality": quality, "audio_status": status, "scores_are_calibrated": False,
                         "emotion_is": "illustrative rule from predicted behaviors; not verified emotion"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["video", "model", "detector", "pose", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    predict(parser.parse_args())
