"""Sample videos, extract pose and audio features, and resume completed videos."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy import fft, signal


def aggregate(values, prefix):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    result = {}
    for i in range(array.shape[1]):
        column = array[:, i]
        finite = column[np.isfinite(column)]
        for key, value in {
            "mean": np.mean(finite) if len(finite) else np.nan,
            "std": np.std(finite) if len(finite) else np.nan,
            "median": np.median(finite) if len(finite) else np.nan,
            "p90": np.quantile(finite, .9) if len(finite) else np.nan,
        }.items():
            result[f"{prefix}_{i}_{key}"] = float(value)
    return result


def audio_features(path, seconds):
    import imageio_ffmpeg
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
               "-t", str(seconds), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]
    process = subprocess.run(command, capture_output=True, timeout=120)
    if process.returncode or len(process.stdout) < 2048:
        return {"a_available": 0}, "missing_or_unreadable"
    wave = np.frombuffer(process.stdout, dtype="<f4").astype(float)
    if not np.isfinite(wave).all():
        return {"a_available": 0}, "nonfinite_audio"
    return audio_features_from_wave(wave)


def audio_features_from_wave(wave):
    """The same legacy spectral features on an already-decoded 16 kHz segment."""
    wave = np.asarray(wave, dtype=float)
    if wave.ndim != 1 or len(wave) < 512 or not np.isfinite(wave).all():
        return {"a_available": 0}, "missing_or_unreadable"
    _, _, stft = signal.stft(wave, fs=16000, nperseg=512, noverlap=256)
    power = np.abs(stft) ** 2
    frequencies = np.linspace(0, 8000, power.shape[0])[:, None]
    total = power.sum(axis=0) + 1e-12
    centroid = (power * frequencies).sum(axis=0) / total
    spread = np.sqrt((power * (frequencies - centroid) ** 2).sum(axis=0) / total)
    flatness = np.exp(np.log(power + 1e-12).mean(axis=0)) / (power.mean(axis=0) + 1e-12)
    # Explicit triangular mel filters followed by DCT, no model download.
    mel = np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 42)
    hz = 700 * (10 ** (mel / 2595) - 1)
    filters = []
    f = frequencies[:, 0]
    for low, center, high in zip(hz, hz[1:], hz[2:]):
        filters.append(np.maximum(0, np.minimum((f-low)/(center-low), (high-f)/(high-center))))
    logmel = np.log(np.asarray(filters) @ power + 1e-10)
    mfcc = fft.dct(logmel, type=2, axis=0, norm="ortho")[:13].T
    zcr = np.mean(np.signbit(wave[1:]) != np.signbit(wave[:-1]))
    result = {"a_available": 1, "a_rms": float(np.sqrt(np.mean(wave**2))), "a_zcr": float(zcr)}
    result.update(aggregate(np.column_stack([centroid, spread, flatness, np.log(total)]), "a_spectral"))
    result.update(aggregate(mfcc, "a_mfcc"))
    return result, "complete"


def visual_features(path, detector, adapter, sample_fps=2, seconds=60, confidence=.3):
    import cv2
    from src.preprocessing.interfaces import VideoFrame
    capture = cv2.VideoCapture(str(path))
    poses, movements, timestamps, records = [], [], [], []
    sample_count = detections = multi = decoded = 0
    previous = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video cannot be opened or FPS invalid")
        stride = max(1, int(round(fps / sample_fps)))
        limit = min(source_frames, int(seconds * fps)) if source_frames > 0 else int(seconds * fps)
        for index in range(limit):
            if not capture.grab():
                break
            decoded += 1
            if index % stride:
                continue
            success, pixels = capture.retrieve()
            if not success:
                previous = None
                continue
            timestamp = index / fps
            frame = VideoFrame(index, timestamp, pixels)
            dogs = detector.detect(frame)
            sample_count += 1
            record = {"frame_index": index, "timestamp_seconds": timestamp, "dog_count": len(dogs)}
            if not dogs:
                previous = None
                records.append(record)
                continue
            detections += 1
            multi += int(len(dogs) > 1)
            dog = max(dogs, key=lambda d: (d.xyxy[2]-d.xyxy[0]) * (d.xyxy[3]-d.xyxy[1]))
            pose = adapter.estimate(frame, [dog])[0]
            xy, scores = np.asarray(pose.xy), np.asarray(pose.confidence)
            bbox = np.asarray(dog.xyxy)
            width, height = bbox[2:] - bbox[:2]
            scale = np.hypot(width, height)
            if scale <= 0:
                previous = None
                continue
            center = (bbox[:2]+bbox[2:])/2
            normalized = (xy-center)/scale
            valid = np.isfinite(xy).all(axis=1) & np.isfinite(scores) & (scores >= confidence)
            valid &= (xy[:, 0] >= 0) & (xy[:, 0] < pixels.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < pixels.shape[0])
            valid &= np.asarray(["antler" not in name for name in pose.keypoint_names])
            normalized[~valid] = np.nan
            poses.append(normalized.reshape(-1))
            timestamps.append(timestamp)
            if previous is not None and len(dogs) == 1:
                prev_xy, prev_t = previous
                delta = (normalized - prev_xy)/(timestamp-prev_t)
                movements.append(np.linalg.norm(delta, axis=1))
            previous = (normalized, timestamp) if len(dogs) == 1 else None
            record.update(bbox_xyxy=bbox.tolist(), keypoint_names=list(pose.keypoint_names),
                          xy=xy.tolist(), confidence=scores.tolist())
            records.append(record)
    finally:
        capture.release()
    if sample_count == 0:
        raise ValueError("No video frames decoded")
    n = len(adapter.schema.names)
    result = {"v_detection_rate": detections/sample_count, "v_multi_rate": multi/sample_count,
              "v_pose_available": int(bool(poses))}
    result.update(aggregate(poses if poses else np.full((1, n*2), np.nan), "v_pose"))
    result.update(aggregate(movements if movements else np.full((1, n), np.nan), "v_speed"))
    quality = {"sampled_frames": sample_count, "detected_frames": detections, "multi_dog_frames": multi,
               "sample_fps": sample_fps, "source_fps": fps, "decoded_frames": decoded,
               "source_frame_count": source_frames, "seconds_limit": seconds,
               "timestamp_source": "frame_index_div_fps", "dog_selection": "largest_box_per_frame_no_tracking"}
    return result, quality, records


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(args):
    from src.preprocessing.detection import YoloDogDetector
    from src.pose_estimation.backend import DeepLabCutRTMPose
    from src.pose_estimation.adapter import DogPoseAdapter
    import torch
    torch.set_num_threads(4)
    table = pd.read_csv(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    signature = {"schema": 1, "sample_fps": args.sample_fps, "seconds": args.seconds,
                 "confidence": .3, "device": "cpu", "detector_sha256": digest(args.detector),
                 "pose_sha256": digest(args.pose), "extractor_sha256": digest(__file__)}
    detector = YoloDogDetector(args.detector, "cpu", .25)
    adapter = DogPoseAdapter(DeepLabCutRTMPose(args.pose, device="cpu"))
    successes, failures = [], []
    for i, row in table.iterrows():
        start = time.monotonic()
        key = row.source_sha256
        target = args.output / f"{key}.json"
        try:
            if target.exists():
                cached = json.loads(target.read_text())
                if cached.get("signature") == signature and cached.get("source_sha256") == key:
                    successes.append({"video_id": row.video_id, **cached["features"]})
                    print(f"[{i+1}/{len(table)}] resumed {row.video_id}", flush=True)
                    continue
            path = Path(row.video_path)
            if digest(path) != key:
                raise ValueError("Source file changed since manifest creation")
            visual, quality, records = visual_features(path, detector, adapter, args.sample_fps, args.seconds)
            audio, audio_status = audio_features(path, args.seconds)
            content = clean_json({"video_id": row.video_id, "source_sha256": key, "signature": signature,
                                  "features": {**visual, **audio}, "quality": quality, "audio_status": audio_status})
            pose_path = args.output / f"{key}.pose.jsonl"
            pose_path.write_text("".join(json.dumps(clean_json(r), allow_nan=False)+"\n" for r in records))
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(content, ensure_ascii=False, allow_nan=False))
            temporary.replace(target)
            successes.append({"video_id": row.video_id, **content["features"]})
            print(f"[{i+1}/{len(table)}] {row.video_id}: dogs {quality['detected_frames']}/{quality['sampled_frames']}, audio {audio_status}, {time.monotonic()-start:.1f}s", flush=True)
        except Exception as exc:
            failures.append({"video_id": row.video_id, "error": str(exc)})
            print(f"[{i+1}/{len(table)}] FAILED {row.video_id}: {exc}", flush=True)
        pd.DataFrame(successes).to_csv(args.output / "features.csv", index=False)
        (args.output / "failures.json").write_text(json.dumps(failures, indent=2))
    pd.DataFrame(successes).to_csv(args.output / "features.csv", index=False)
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"Completed: {len(successes)}; failed: {len(failures)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=2)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    if not 0 < args.sample_fps <= 30 or not 0 < args.seconds <= 600:
        parser.error("sample-fps must be (0,30], seconds (0,600]")
    extract(args)
