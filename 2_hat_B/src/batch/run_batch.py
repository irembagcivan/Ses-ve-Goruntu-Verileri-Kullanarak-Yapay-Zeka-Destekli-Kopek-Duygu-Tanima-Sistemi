"""Batch CLI: sequential processing with explicit local model paths."""
import argparse
import math
import platform
from importlib import metadata
from pathlib import Path

from src.batch.runner import LocalPipeline, digest, run_batch


def main(argv=None):
    parser = argparse.ArgumentParser(description='Sequential MOV/MP4 pose and motion batch')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--detector-weights', required=True)
    parser.add_argument('--pose-weights', required=True)
    parser.add_argument('--pose-config')
    parser.add_argument('--device', choices=['cpu', 'mps', 'auto'], default='cpu')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--keypoint-confidence', type=float, default=.5)
    parser.add_argument('--motion-confidence', type=float, default=.5)
    parser.add_argument('--margin', type=float, default=.15)
    args = parser.parse_args(argv)
    try:
        from scripts.collect_environment import git_info
        root = Path(__file__).resolve().parents[2]
        for value in (args.confidence, args.keypoint_confidence, args.motion_confidence):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid confidence')
        if not math.isfinite(args.margin) or not 0 <= args.margin <= .5:
            raise ValueError('Invalid margin')
        # Hash config and all source modules: resume invalidates on code/config changes,
        # including uncommitted changes. Record no absolute filenames.
        code = {str(p.relative_to(root)): digest(p)
                for folder in ('src', 'configs')
                for p in sorted((root / folder).rglob('*'))
                if p.is_file() and p.suffix in {'.py', '.json', '.yaml'}}
        versions = {}
        for package in ('torch', 'torchvision', 'deeplabcut', 'ultralytics', 'numpy', 'opencv-python', 'opencv-python-headless'):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        settings = {'runtime': {'python': platform.python_version(), 'system': platform.system(),
                                'machine': platform.machine(), 'packages': versions},
                    'schema_version': 1, 'device_requested': args.device,
                    'detector_sha256': digest(args.detector_weights),
                    'pose_sha256': digest(args.pose_weights),
                    'pose_config_sha256': digest(args.pose_config) if args.pose_config else None,
                    'confidence': args.confidence, 'keypoint_confidence': args.keypoint_confidence,
                    'motion_confidence': args.motion_confidence, 'margin': args.margin,
                    'visualization': 'conservative', 'code_sha256': code, 'git': git_info()}
        pipeline = LocalPipeline(args.detector_weights, args.pose_weights, args.pose_config,
                                 args.device, args.confidence, args.margin)
        report = run_batch(args.input_dir, args.output_dir, settings, pipeline,
                           args.limit, not args.no_resume)
    except (OSError, ValueError, RuntimeError):
        parser.exit(2, 'Error: invalid settings, missing local files, unsafe output or active/stale batch lock.\n')
    print(f"Videos={report['total']}; failed={report['failed']}; report: batch-run-*/report.json")
    return 2 if report['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
