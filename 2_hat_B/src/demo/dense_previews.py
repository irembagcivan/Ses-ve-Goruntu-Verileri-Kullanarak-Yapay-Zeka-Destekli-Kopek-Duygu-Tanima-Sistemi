"""Fresh multi-dog inference on every displayed preview frame, with audio."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

from src.demo.extract import digest
from src.demo.tracking import PreviewTracker
from src.pose_estimation.dog_skeleton import dog_visualization_schema

COLORS = [(132, 239, 182), (255, 191, 105), (189, 151, 255), (115, 215, 255), (255, 155, 179)]


def draw_frame(pixels, dogs, poses, tracker, timestamp, schema):
    canvas = pixels.copy()
    if hasattr(tracker,"set_frame"):
        tracker.set_frame(pixels)
    tracks = tracker.update(dogs, poses, timestamp)
    height, width = canvas.shape[:2]
    for track in tracks:
        color = COLORS[(track.identifier-1) % len(COLORS)]
        diagonal = np.linalg.norm(track.bbox[2:]-track.bbox[:2])
        valid = np.isfinite(track.xy).all(axis=1)
        valid &= (track.xy[:, 0] >= 0) & (track.xy[:, 0] < width) & (track.xy[:, 1] >= 0) & (track.xy[:, 1] < height)
        selected = {i for edge in schema.edges for i in edge}
        for a, b in schema.edges:
            allowed = valid[a] and valid[b] and np.linalg.norm(track.xy[a]-track.xy[b]) <= .65*diagonal
            if allowed and hasattr(tracker,"edge_allowed"):
                allowed = tracker.edge_allowed(track,a,b,schema.names)
            if allowed:
                cv2.line(canvas, tuple(np.rint(track.xy[a]).astype(int)), tuple(np.rint(track.xy[b]).astype(int)), color, 2, cv2.LINE_AA)
        for index in selected:
            if valid[index]:
                xy = tuple(np.rint(track.xy[index]).astype(int))
                cv2.circle(canvas, xy, 3, (20, 30, 20), -1, cv2.LINE_AA)
                cv2.circle(canvas, xy, 2, color, -1, cv2.LINE_AA)
        x1, y1, x2, y2 = np.rint(track.bbox).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        cv2.putText(canvas, f"Dog #{track.identifier}", (max(0,x1), max(40,y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0,0), (width,29), (18,30,24), -1)
    cv2.putText(canvas, f"DOGS: {len(dogs)}  |  Multi-dog pose / fresh frame", (10,20), cv2.FONT_HERSHEY_SIMPLEX, .47, (220,245,225), 1, cv2.LINE_AA)
    return canvas, len(tracks)


def render(source, output, detector, adapter, fps_limit=15, seconds=12, stable=False):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Cannot open video")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise ValueError("Invalid source FPS")
    output_fps = min(fps_limit, source_fps)
    width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    resize = min(1, 960/max(width,height))
    size = (int(width*resize)//2*2, int(height*resize)//2*2)
    temporary = output.with_suffix(".silent.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, size)
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Cannot create video writer")
    if stable:
        from src.demo.stable_pose import FlowPreviewTracker
        tracker = FlowPreviewTracker()
    else:
        tracker = PreviewTracker()
    schema = dog_visualization_schema(adapter.schema.names, "legacy")
    from src.preprocessing.interfaces import VideoFrame
    source_index = emitted = 0
    counts = []
    previous_pixels=None
    previous_dogs,previous_poses=[],[]
    try:
        while emitted < int(seconds*output_fps):
            needed_index = int(round(emitted*source_fps/output_fps))
            while source_index < needed_index:
                if not capture.grab():
                    break
                source_index += 1
            ok, pixels = capture.read()
            if not ok:
                break
            actual_index = source_index
            source_index += 1
            pixels = cv2.resize(pixels, size)
            frame = VideoFrame(actual_index, actual_index/source_fps, pixels)
            use_flow=stable and emitted%3!=0 and previous_pixels is not None
            if use_flow:
                from src.demo.pose_flow import propagate
                dogs,poses=propagate(previous_pixels,pixels,previous_dogs,previous_poses)
            else:
                dogs = detector.detect(frame)
                poses = adapter.estimate(frame, dogs)
            tracker.already_propagated=use_flow
            canvas, count = draw_frame(pixels, dogs, poses, tracker, frame.timestamp_seconds, schema)
            writer.write(canvas)
            counts.append(count)
            emitted += 1
            previous_pixels=pixels
            previous_dogs,previous_poses=dogs,poses
    finally:
        capture.release()
        writer.release()
    if not emitted:
        temporary.unlink(missing_ok=True)
        raise ValueError("No decoded preview frames")
    part = output.with_suffix(".partial.mp4")
    process = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(temporary),
                              "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?", "-t", str(emitted/output_fps),
                              "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
                              "-c:a", "aac", "-movflags", "+faststart", str(part)], capture_output=True, timeout=120)
    if process.returncode:
        part.unlink(missing_ok=True)
        raise RuntimeError(process.stderr.decode(errors="replace"))
    part.replace(output)
    temporary.unlink()
    return {"frames": emitted, "fps": output_fps, "max_dogs": max(counts), "multi_dog_frames": sum(c>1 for c in counts),
            "fresh_inference_each_displayed_frame": not stable,
            "pose_model_fps": output_fps/3 if stable else output_fps,
            "between_model_frames": "forward_backward_checked_image_flow" if stable else "none",
            "tracking": "short_term_box_IoU_not_identity_recognition",
            "skeleton": "project_26_point_18_edge_guide", "keypoint_confidence": .4,
            "stabilization": "optical_flow_forward_backward_and_strict_edges" if stable else "box_translation_EMA",
            "behavior_scope": "Existing sidebar predictions describe main dog/video, not each track separately"}


def main(args):
    from src.preprocessing.detection import YoloDogDetector
    from src.pose_estimation.backend import DeepLabCutRTMPose
    from src.pose_estimation.adapter import DogPoseAdapter
    import torch
    torch.set_num_threads(4)
    report = json.loads((args.results / "report.json").read_text())
    table = pd.read_csv(args.manifest).set_index("video_id")
    destination = args.results / "public" / args.clip_directory
    destination.mkdir(parents=True, exist_ok=True)
    detector = YoloDogDetector(args.detector, "cpu", .25)
    adapter = DogPoseAdapter(DeepLabCutRTMPose(args.pose, device="cpu"))
    if args.rescue:
        from src.demo.robust_detection import RobustDogDetector, OrientedPoseAdapter
        detector = RobustDogDetector(args.detector,args.rescue)
        adapter = OrientedPoseAdapter(adapter)
    signature = {"version": 3, "fps": args.fps, "detector": digest(args.detector), "pose": digest(args.pose),
                 "renderer": digest(__file__), "tracking": digest(Path(__file__).with_name("tracking.py"))}
    signature.update(stable=args.stable,rescue=digest(args.rescue) if args.rescue else None)
    if args.stable:
        signature['stabilizer'] = digest(Path(__file__).with_name('stable_pose.py'))
        signature['pose_flow'] = digest(Path(__file__).with_name('pose_flow.py'))
    if args.rescue:
        signature['robust_detector'] = digest(Path(__file__).with_name('robust_detection.py'))
    selected = [s for s in report["samples"] if not args.only or s["video_id"] in args.only]
    for i, sample in enumerate(selected, 1):
        start = time.monotonic()
        key = sample["source_sha256"]
        target = destination / f"{key}.mp4"
        metadata = destination / f"{key}.json"
        if target.exists() and metadata.exists() and json.loads(metadata.read_text()).get("signature") == signature:
            print(f"[{i}/{len(selected)}] resumed {sample['video_id']}", flush=True)
            continue
        result = render(Path(table.loc[sample["video_id"], "video_path"]), target, detector, adapter, args.fps,stable=args.stable)
        metadata.write_text(json.dumps({**result, "signature": signature}, indent=2))
        print(f"[{i}/{len(selected)}] {sample['video_id']}: {result['frames']} frames, max {result['max_dogs']} dogs, multi {result['multi_dog_frames']}, {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["manifest", "results", "detector", "pose"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--only", action="append")
    parser.add_argument("--clip-directory",default="clips_v3")
    parser.add_argument("--stable",action="store_true")
    parser.add_argument("--rescue",type=Path)
    args = parser.parse_args()
    if not 1 <= args.fps <= 30:
        parser.error("fps must be between 1 and 30")
    main(args)
