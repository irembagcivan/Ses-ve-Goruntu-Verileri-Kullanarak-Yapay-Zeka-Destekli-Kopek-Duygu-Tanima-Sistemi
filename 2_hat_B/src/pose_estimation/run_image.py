"""Single-image CLI. All checkpoint use is local and explicit."""
import argparse
import json

from src.pose_estimation.adapter import DogPoseAdapter
from src.pose_estimation.backend import DeepLabCutRTMPose
from src.pose_estimation.device import select_device, inference_error
from src.preprocessing.detection import YoloDogDetector
from src.preprocessing.image import process_image, validate_image_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description='YOLOv8 + SuperAnimal-Quadruped RTMPose-S image inference')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--detector-weights', required=True, help='Local YOLOv8 detection checkpoint')
    parser.add_argument('--pose-weights', required=True, help='Local official SuperAnimal RTMPose-S .pt')
    parser.add_argument('--pose-config', help='Unmodified official DLC 3.0.2 rtmpose_s.yaml (default: bundled official YAML)')
    parser.add_argument('--checkpoint-sha256', help='Optional expected checksum independently obtained from trusted source')
    parser.add_argument('--device', choices=['cpu', 'mps', 'auto'], default='cpu')
    parser.add_argument('--margin', type=float, default=0.15)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--keypoint-confidence', type=float, default=0.5)
    args = parser.parse_args(argv)
    device = args.device
    try:
        validate_image_paths(args.input, args.output)
        if not 0 <= args.margin <= 0.5 or not 0 <= args.confidence <= 1 or not 0 <= args.keypoint_confidence <= 1:
            raise ValueError('Margin must be [0, 0.5]; confidence thresholds must be [0, 1]')
        device = select_device(args.device)
        backend = DeepLabCutRTMPose(args.pose_weights, args.pose_config, device,
                                   checkpoint_sha256=args.checkpoint_sha256)
        detector = YoloDogDetector(args.detector_weights, device, args.confidence)
        report = process_image(args.input, args.output, detector, DogPoseAdapter(backend, args.margin),
                               args.keypoint_confidence)
        print(json.dumps(report))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        hint = inference_error(exc, device) if device == 'mps' else exc
        parser.exit(2, f'Error: {hint}\n')


if __name__ == '__main__':
    main()
