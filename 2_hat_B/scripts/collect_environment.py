"""Collect a privacy-limited environment record using only the standard library.

Run with the same Python interpreter used for inference. Package imports are
limited to torch (MPS probe) and cv2 (actual runtime version). No downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess

REPOSITORY = Path(__file__).resolve().parents[1]
DISTRIBUTIONS = {
    'torch': ('torch',),
    'torchvision': ('torchvision',),
    'deeplabcut': ('deeplabcut',),
    'ultralytics': ('ultralytics',),
    'numpy': ('numpy',),
    'opencv': ('opencv-python', 'opencv-python-headless',
               'opencv-contrib-python', 'opencv-contrib-python-headless'),
}


def safe_version(value):
    """Exclude arbitrary build strings/local identifiers that may contain PII."""
    if not isinstance(value, str):
        return None
    # Local version suffixes can embed machine/user names. Never export them.
    public = value.split('+', 1)[0]
    if re.fullmatch(r'\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?', public):
        return public
    return None


def distribution_version(name):
    try:
        return safe_version(metadata.version(name))
    except Exception:
        # Do not serialize exception messages: they may contain user paths.
        return None


def runtime_probe():
    result = {'mps_available': None, 'mps_status': 'unavailable_or_unusable',
              'opencv_version': None, 'opencv_status': 'unavailable_or_unusable'}
    try:
        torch = importlib.import_module('torch')
        result['mps_available'] = bool(torch.backends.mps.is_available())
        result['mps_status'] = 'checked'
    except Exception:
        pass
    try:
        cv2 = importlib.import_module('cv2')
        result['opencv_version'] = safe_version(cv2.__version__)
        result['opencv_status'] = 'checked' if result['opencv_version'] else 'unknown'
    except Exception:
        pass
    return result


def git_info(repository=REPOSITORY):
    """Only commit and dirty state; never remotes, author identities or paths."""
    result = {'commit': None, 'dirty': None}
    try:
        head = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD^{commit}'],
                              cwd=repository, capture_output=True, text=True, timeout=10)
        commit = head.stdout.strip()
        if head.returncode == 0 and re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', commit):
            result['commit'] = commit
        status = subprocess.run(['git', 'status', '--porcelain=v1', '--untracked-files=normal'],
                                cwd=repository, capture_output=True, text=True, timeout=10)
        if status.returncode == 0:
            result['dirty'] = bool(status.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return result


def checkpoint_info(path, index):
    """Stream bytes; opaque ordinal IDs prevent filenames leaking user names."""
    record = {'id': f'checkpoint_{index}', 'sha256': None, 'status': 'unreadable'}
    try:
        source = Path(path)
        if not source.exists():
            record['status'] = 'missing'
        elif not source.is_file():
            record['status'] = 'not_a_file'
        else:
            digest = hashlib.sha256()
            with source.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            record['sha256'] = digest.hexdigest()
            record['status'] = 'hashed'
    except OSError:
        pass
    return record


def collect_environment(checkpoints=(), repository=REPOSITORY):
    system = platform.system()
    known_system = system if system in {'Darwin', 'Linux', 'Windows'} else None
    raw_release = platform.mac_ver()[0] if system == 'Darwin' else platform.release()
    match = re.match(r'\d+(?:\.\d+)*', raw_release)
    architecture = platform.machine()
    if architecture not in {'arm64', 'aarch64', 'x86_64', 'AMD64', 'i386', 'i686', 'armv7l', 'ppc64le', 's390x', 'riscv64'}:
        architecture = None
    packages = {key: {name: distribution_version(name) for name in names}
                for key, names in DISTRIBUTIONS.items()}
    return {
        'schema_version': 1,
        'python_version': safe_version(platform.python_version()),
        'os': {'system': known_system, 'version': match.group(0) if match else None},
        'architecture': architecture,
        'packages': packages,
        'runtime': runtime_probe(),
        'git': git_info(repository),
        'checkpoints': [checkpoint_info(path, i) for i, path in enumerate(checkpoints, 1)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Save privacy-limited environment metadata; never downloads models')
    parser.add_argument('--output', required=True, help='New JSON file; existing files are never overwritten')
    parser.add_argument('--checkpoint', action='append', default=[], metavar='PATH',
                        help='Optional local checkpoint to hash; repeatable, paths are not saved')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.suffix.lower() != '.json':
        parser.exit(2, 'Error: output must be a JSON file\n')
    if any(Path(path).resolve() == output.resolve() for path in args.checkpoint):
        parser.exit(2, 'Error: output must differ from checkpoint inputs\n')
    # Reserve output before collection; never overwrite a checkpoint or existing file.
    try:
        with output.open('x', encoding='utf-8') as stream:
            try:
                report = collect_environment(args.checkpoint)
                stream.write(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + '\n')
            except BaseException:
                output.unlink(missing_ok=True)
                raise
    except FileExistsError:
        parser.exit(2, 'Error: output already exists; choose a new file\n')
    except OSError:
        parser.exit(2, 'Error: output could not be written; check its parent directory and permissions\n')
    print('Environment JSON saved. Missing or unprobeable components are recorded as null.')
    return 0


if __name__ == '__main__':
    main()
