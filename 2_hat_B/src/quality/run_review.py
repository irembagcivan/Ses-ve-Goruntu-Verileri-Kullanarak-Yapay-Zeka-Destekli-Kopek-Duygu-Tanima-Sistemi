"""Compare existing quality outputs; model inference is never invoked."""
import argparse
from src.quality.review import run_review


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,help='JSON videos list with pose/quality/frames paths')
    parser.add_argument('--output-dir',required=True,help='New directory; existing directories are refused')
    parser.add_argument('--per-category',type=int,default=2)
    parser.add_argument('--low-confidence',type=float,default=.5)
    parser.add_argument('--visibility-change-min',type=int,default=8)
    parser.add_argument('--render',action='store_true',help='Decode selected original frames for PNG comparisons')
    args=parser.parse_args(argv)
    try:
        result=run_review(args.manifest,args.output_dir,args.per_category,args.low_confidence,
                          args.visibility_change_min,args.render)
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        parser.exit(2,'Error: invalid/mismatched inputs or output directory already exists/unwritable.\n')
    failures=sum(r['visualization']['status']=='failed_or_unavailable' for r in result['videos'])
    print(f"Review: videos={len(result['videos'])}; optional_visualization_failures={failures}")
    return 2 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
