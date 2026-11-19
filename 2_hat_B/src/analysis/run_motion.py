"""Offline JSONL motion CLI; does not import or run detection/pose models."""
import argparse

from src.analysis.features import MotionSettings
from src.analysis.report import analyze_file


def main(argv=None):
    parser=argparse.ArgumentParser(description='Single-dog adjacent-frame motion from existing pose JSONL')
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True,help='New per-frame CSV')
    parser.add_argument('--summary',required=True,help='New summary JSON')
    parser.add_argument('--confidence',type=float,default=.5)
    parser.add_argument('--detection-confidence',type=float,default=0.0)
    parser.add_argument('--normalization',choices=['bbox_diagonal','none'],default='bbox_diagonal')
    parser.add_argument('--max-step-ratio',type=float,default=.5,
                        help='Flag/exclude point steps above this fraction of mean bbox diagonal')
    parser.add_argument('--min-region-points',type=int,default=2)
    parser.add_argument('--plot',help='Optional PNG; failure does not prevent CSV/JSON output')
    args=parser.parse_args(argv)
    try:
        settings=MotionSettings(args.confidence,args.detection_confidence,args.normalization,
                                args.max_step_ratio,args.min_region_points)
        result=analyze_file(args.input,args.output,args.summary,settings,args.plot)
    except (ValueError,OSError):
        parser.exit(2,'Error: invalid input/settings or unreadable/unwritable output; use new output paths\n')
    print(f"Motion analysis: {result['status']}; records={result['total_frames']}; valid_motion_frames={result['valid_analysis_frames']}")
    return 0 if result['valid_analysis_frames'] else 2


if __name__=='__main__':
    raise SystemExit(main())
