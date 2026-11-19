"""Streaming OpenCV video pipeline. Output is constant-FPS and silent."""
from pathlib import Path
import math
from src.preprocessing.interfaces import VideoFrame
from src.pose_estimation.visualization import draw_poses
from src.pose_estimation.device import inference_error


def validate_video_paths(input_path, output_path):
    source, target = Path(input_path), Path(output_path)
    if source.suffix.lower() not in {".mov", ".mp4"}:
        raise ValueError("Input must be MOV or MP4")
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    if target.suffix.lower() != ".mp4":
        raise ValueError("Output must be MP4")
    if source.resolve() == target.resolve():
        raise ValueError("Input and output must differ")
    if target.exists():
        raise FileExistsError("Output exists; choose a new path")
    return source, target


def process_video(input_path, output_path, detector, estimator, threshold=0.5,
                  device="cpu", cv_module=None, dog_skeleton=False, pose_output=None, pose_metadata=None):
    source, target = validate_video_paths(input_path, output_path)
    if pose_output is not None:
        from src.pose_estimation.export import validate_export_paths
        validate_export_paths(pose_output,source,target)
    if dog_skeleton:
        from src.pose_estimation.dog_skeleton import dog_visualization_schema, visualization_profile
        dog_visualization_schema(estimator.schema.names, visualization_profile(dog_skeleton))
    if cv_module is None:
        import cv2 as cv_module
    capture = cv_module.VideoCapture(str(source))
    writer = None
    pose_writer = None
    completed = False
    try:
        if not capture.isOpened():
            raise ValueError("Video decoder could not open the input; check MOV/MP4 codec support")
        fps = capture.get(cv_module.CAP_PROP_FPS)
        width = int(capture.get(cv_module.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv_module.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video has invalid FPS or dimensions")
        ok, pixels = capture.read()
        if not ok:
            raise ValueError("Video contains no decodable frames")
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = cv_module.VideoWriter(str(target), cv_module.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("MP4 writer could not open; check codec support")
        if pose_output is not None:
            from src.pose_estimation.export import PoseJsonlWriter, run_metadata
            context = pose_metadata if pose_metadata is not None else run_metadata(
                device=device,detection_confidence=getattr(detector,"confidence",None),
                keypoint_confidence=threshold,margin=getattr(estimator,"margin",None),visualization=dog_skeleton)
            pose_writer = PoseJsonlWriter(pose_output,estimator.schema.names,fps,width,height,context)
        count = 0
        while ok:
            if pixels.shape[:2] != (height, width):
                raise ValueError("Frame resolution changed or disagrees with metadata")
            frame = VideoFrame(count, count/fps, pixels)
            try:
                dogs = detector.detect(frame)
                poses = estimator.estimate(frame, dogs)
            except RuntimeError as exc:
                raise inference_error(exc, device) from exc
            writer.write(draw_poses(pixels, dogs, poses, estimator.schema, threshold, cv_module,
                                    dog_skeleton=dog_skeleton))
            if pose_writer is not None:
                pose_writer.write(count,dogs,poses)
            count += 1
            ok, pixels = capture.read()
        if pose_writer is not None:
            pose_writer.close(complete=True)
        completed = True
        return {"frames": count, "fps": fps, "width": width, "height": height, "audio_preserved": False}
    finally:
        try:
            capture.release()
        finally:
            try:
                if writer is not None:
                    writer.release()
            finally:
                try:
                    if pose_writer is not None:
                        pose_writer.close(complete=completed)
                finally:
                    if writer is not None and not completed:
                        target.unlink(missing_ok=True)
