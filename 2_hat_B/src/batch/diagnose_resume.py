"""Read-only diagnosis of existing attempts; never loads models or starts inference."""
import argparse
from pathlib import Path
from src.batch.runner import completion_diagnostic, read_json, write_json


def diagnose(attempt, reference=None):
    attempt = Path(attempt)
    reference = Path(reference) if reference else attempt
    # Compare with the settings actually recorded for a second attempt.
    identity = read_json(reference / 'started.json')['identity']
    diagnostic, _ = completion_diagnostic(attempt, identity)
    return {'resume_diagnostic_schema_version': 1, 'comparison': diagnostic,
            'limitations': ['Compares recorded attempt identities, not the current source/model files.',
                            'Only the first failed validation stage is reported.',
                            'Different output roots or --no-resume require the original invocation.',
                            'Does not prove why a historical run chose a new attempt.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--reference', help='Second attempt whose started.json supplies expected identity')
    parser.add_argument('--output', required=True, help='New diagnosis JSON')
    args = parser.parse_args(argv)
    try:
        result = diagnose(args.attempt, args.reference)
        write_json(Path(args.output), result)
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(2, 'Error: missing/invalid attempt records or output exists/unwritable.\n')
    print(result['comparison']['reason'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
