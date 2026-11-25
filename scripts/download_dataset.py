#!/usr/bin/env python3
"""Download the public EgoMyo dataset repository from the Hugging Face Hub."""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', default='long2333/EgoMyo')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--local-dir', type=Path, default=Path('data/egomyo'))
    parser.add_argument('--allow-pattern', action='append', dest='allow_patterns',
                        help='Optional file glob to download; repeat to include more patterns')
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit('Install dependencies with: pip install -r requirements.txt') from error

    destination = args.local_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type='dataset',
        revision=args.revision,
        local_dir=str(destination),
        allow_patterns=args.allow_patterns or None,
    )
    print(f'Dataset snapshot ready at {destination}')


if __name__ == '__main__':
    main()
