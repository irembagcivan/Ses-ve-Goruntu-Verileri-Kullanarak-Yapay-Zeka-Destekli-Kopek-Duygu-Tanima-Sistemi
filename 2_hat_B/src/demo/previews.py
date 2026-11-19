"""Render short held-out clips with sampled pose points, and retain audio."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd


def previews(manifest, feature_dir, results):
    report = json.loads((results / "report.json").read_text())
    table = pd.read_csv(manifest).set_index("video_id")
    destination = results / "public" / "clips"
    destination.mkdir(parents=True, exist_ok=True)
    for sample in report["samples"]:
        key = sample["source_sha256"]
        target = destination / f"{key}.mp4"
        if target.exists():
            continue
        source = Path(table.loc[sample["video_id"], "video_path"])
        records = [json.loads(line) for line in (feature_dir / f"{key}.pose.jsonl").read_text().splitlines()]
        capture = cv2.VideoCapture(str(source))
        fps = capture.get(cv2.CAP_PROP_FPS)
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        scale = min(1, 720 / max(width, height))
        size = (max(2, int(width*scale)//2*2), max(2, int(height*scale)//2*2))
        temporary = target.with_suffix(".silent.mp4")
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Preview writer could not open")
        sample_index = 0
        try:
            for i in range(int(min(fps, 120)*12)):
                ok, frame = capture.read()
                if not ok:
                    break
                time = i / fps
                while sample_index+1 < len(records) and records[sample_index+1]["timestamp_seconds"] <= time:
                    sample_index += 1
                record = records[sample_index]
                frame = cv2.resize(frame, size)
                if "xy" in record and time-record["timestamp_seconds"] < .6:
                    for name, xy, score in zip(record["keypoint_names"], record["xy"], record["confidence"]):
                        if "antler" in name or score is None or score < .3 or any(v is None for v in xy):
                            continue
                        x, y = np.asarray(xy)*scale
                        if 0 <= x < size[0] and 0 <= y < size[1]:
                            cv2.circle(frame, (int(x), int(y)), 3, (130, 240, 180), -1, cv2.LINE_AA)
                cv2.putText(frame, "POSE / sampled 2 fps", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (180, 255, 200), 1, cv2.LINE_AA)
                writer.write(frame)
        finally:
            capture.release()
            writer.release()
        process = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(temporary),
                                  "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?", "-t", "12",
                                  "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
                                  "-c:a", "aac", "-movflags", "+faststart", str(target)], capture_output=True, timeout=120)
        if process.returncode:
            target.unlink(missing_ok=True)
            raise RuntimeError(process.stderr.decode(errors="replace"))
        temporary.unlink()
        print(f"Preview: {sample['video_id']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    previews(args.manifest, args.features, args.results)
