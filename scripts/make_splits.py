#!/usr/bin/env python3
"""Freeze the three evaluation splits from the released membership file."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myobridge.splits import build_splits  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--membership', type=Path, required=True)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    membership = json.loads(args.membership.read_text())
    result = build_splits(membership, args.split_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for name, split in result['protocols'].items():
        print(name, 'train', len(split['train']), 'test', len(split['test']))


if __name__ == '__main__':
    main()
