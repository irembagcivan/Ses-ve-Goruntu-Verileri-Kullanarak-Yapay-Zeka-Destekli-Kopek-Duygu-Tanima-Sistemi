"""Offline quality CLI; optional PNG export never invokes detection/pose inference."""
import argparse
from src.quality.analyze import analyze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pose-metadata', help='Optional existing pose.metadata.json')
    parser.add_argument('--motion-summary', help='Optional existing motion_summary.json')
    parser.add_argument('--confidence', type=float, default=.5)
    parser.add_argument('--jump-ratio', type=float, default=.5)
    parser.add_argument('--temporal-min-iou', type=float, default=.3)
    parser.add_argument('--overlap-iou', type=float, default=.8)
    parser.add_argument('--video', help='Original video for optional diagnostic PNGs')
    parser.add_argument('--frames-dir', help='New directory for optional diagnostic PNGs')
    parser.add_argument('--max-frames', type=int, default=8)
    args = parser.parse_args(argv)
    if bool(args.video) != bool(args.frames_dir):
        parser.error('--video and --frames-dir must be provided together')
    if not 1 <= args.max_frames <= 100: parser.error('--max-frames must be [1,100]')
    try:
        result = analyze(args.input,args.output,args.confidence,args.jump_ratio,args.overlap_iou,
                         args.pose_metadata,args.motion_summary,args.temporal_min_iou)
    except (OSError, ValueError):
        parser.exit(2, 'Error: input/settings invalid or output already exists/unwritable.\n')
    print(f"Quality: frames={result['total_frames']}; invalid_records={result['counts'].get('invalid_records',0)}")
    if args.video:
        from pathlib import Path
        from src.quality.visualize import export_samples
        try:
            status = export_samples(args.input,Path(args.output).with_suffix('.frames.jsonl'),
                                    args.video,args.frames_dir,args.confidence,args.max_frames)
            print(f"PNG diagnostics: {status['status']}; png_count={status['png_count']}")
        except Exception:
            print('PNG diagnostics failed/unavailable; quality JSON and frame JSONL preserved.')
            return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
