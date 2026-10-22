"""Rebuild the final mercedes table from its 67 captured LLM batch responses.

    python baselines/runners/merge_mercedes.py --check
    python baselines/runners/merge_mercedes.py --out /tmp/mercedes_cleaned.csv

Uses the same offline replay and hash checks as run_llm.py. No API call is made.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_llm import main as replay_main  # noqa: E402


def merge_mercedes(out=None, check=False):
    arguments = ['--dataset', 'mercedes']
    if out is not None:
        arguments.extend(['--out', str(out)])
    if check:
        arguments.append('--check')
    return replay_main(arguments)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true', help='replay and compare with the saved result; write nothing')
    ap.add_argument('--out', type=Path, help='write the full indexed cleaned CSV to this path')
    args = ap.parse_args(argv)
    return merge_mercedes(args.out, args.check)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        sys.exit(1)
