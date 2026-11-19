"""CLI with local-only checkpoints and fail-closed pose provenance checks."""
import argparse
import json
from src.preprocessing.video import validate_video_paths, process_video
from src.preprocessing.detection import YoloDogDetector
from src.pose_estimation.backend import DeepLabCutRTMPose
from src.pose_estimation.adapter import DogPoseAdapter
from src.pose_estimation.device import select_device


def main(argv=None):
    parser = argparse.ArgumentParser(description="YOLOv8 + SuperAnimal-Quadruped RTMPose-S (local checkpoint required)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detector-weights", required=True)
    parser.add_argument("--pose-weights", required=True)
    parser.add_argument("--pose-config", help="Unmodified DLC 3.0.2 rtmpose_s.yaml; defaults to bundled official YAML")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--keypoint-confidence", type=float, default=0.5)
    parser.add_argument("--dog-skeleton", action="store_true",
                        help="Project-authored dog visualization; not the official model skeleton")
    parser.add_argument("--dog-skeleton-profile", choices=["legacy", "conservative"],
                        help="Requires --dog-skeleton; conservative excludes uncertain torso/limb chains")
    parser.add_argument("--pose-output", help="Optional streaming JSONL; companion .metadata.json is also created")
    args = parser.parse_args(argv)
    if args.dog_skeleton_profile and not args.dog_skeleton:
        parser.error("--dog-skeleton-profile requires --dog-skeleton")
    visualization_mode = 'conservative' if args.dog_skeleton_profile == 'conservative' else args.dog_skeleton
    try:
        validate_video_paths(args.input, args.output)
        if not 0 <= args.margin <= 0.5 or not 0 <= args.keypoint_confidence <= 1:
            raise ValueError("Margin must be [0,0.5]; keypoint confidence [0,1]")
        if args.pose_output is not None:
            from src.pose_estimation.export import validate_export_paths
            validate_export_paths(args.pose_output,args.input,args.output,args.pose_weights,
                                  args.detector_weights,args.pose_config)
        device = select_device(args.device)
        # Pose validation comes first: never run a detector for an unusable pose backend.
        backend = DeepLabCutRTMPose(args.pose_weights, args.pose_config, device)
        detector = YoloDogDetector(args.detector_weights, device, args.confidence)
        export_options = {}
        if args.pose_output is not None:
            from src.pose_estimation.export import run_metadata
            export_options = {'pose_output':args.pose_output,'pose_metadata':run_metadata(
                args.detector_weights,args.pose_weights,device,args.confidence,
                args.keypoint_confidence,args.margin,visualization_mode)}
        result = process_video(args.input, args.output, detector, DogPoseAdapter(backend, args.margin),
                               args.keypoint_confidence, device, dog_skeleton=visualization_mode,**export_options)
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
