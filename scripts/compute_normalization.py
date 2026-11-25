#!/usr/bin/env python3
"""Compute per-channel EMG mean and standard deviation from TRAIN trials only.

The statistics are written next to the dataset and reused unchanged for every
model seed and for evaluation on the held-out split.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--protocol', default='cross_trial')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    membership = json.loads((args.data / 'membership.json').read_text())
    splits = json.loads(args.splits.read_text())
    ids = splits['protocols'][args.protocol]['train']
    counts = np.zeros(18, dtype=np.float64)
    total = np.zeros(18, dtype=np.float64)
    squared = np.zeros(18, dtype=np.float64)
    for uid in ids:
        with np.load(args.data / 'targets' / f'{uid}.npz', allow_pickle=False) as archive:
            emg = archive['emg'].astype(np.float64)
            valid = archive['input_valid'].astype(bool) if 'input_valid' in archive.files \
                else np.ones(len(emg), bool)
        emg = emg[valid].transpose(1, 0, 2).reshape(18, -1)
        counts += emg.shape[1]
        total += emg.sum(1)
        squared += (emg ** 2).sum(1)
    mean = total / counts
    variance = squared / counts - mean ** 2
    std = np.sqrt(np.maximum(variance, 0.))
    std[std < 1e-6] = 1.
    args.out.write_text(json.dumps(dict(mean=mean.tolist(), std=std.tolist(),
                                        protocol=args.protocol, trials=len(ids)), indent=2))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
