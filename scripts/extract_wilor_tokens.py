#!/usr/bin/env python3
"""Cache frozen WiLoR backbone tokens for every trial.

Each trial folder stores ``wilor.npy`` with RGB crops [N,3,256,256]. The output
is ``features/wilor/<trial_id>.npy`` [N,13,1280] in float16, the 4x3 adaptive
average pooled feature map plus one global mean token. No target label is read
by this step.

WiLoR is third party software. Install it from its official repository and set
``--adapter`` to a module implementing the interface in
``third_party/wilor_adapter.py``.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_adapter(path):
    spec = importlib.util.spec_from_file_location('wilor_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--crops', type=Path, required=True,
                        help='Directory with <trial_id>/wilor.npy crops')
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True,
                        help='Official WiLoR model_config.yaml')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    adapter = load_adapter(args.adapter)
    extract = adapter.load(args.checkpoint, config=args.config, device=args.device)
    membership = json.loads((args.data / 'membership.json').read_text())
    out = args.data / 'features' / 'wilor'
    out.mkdir(parents=True, exist_ok=True)
    for row in membership['trials']:
        uid = row['trial_id']
        destination = out / f'{uid}.npy'
        if destination.exists():
            continue
        pixels = np.load(args.crops / uid / 'wilor.npy', mmap_mode='r')
        if pixels.shape[1:] != (3, 256, 256):
            raise ValueError(f'Unexpected crop shape {pixels.shape} for {uid}')
        pooled = []
        for start in range(0, len(pixels), args.batch):
            crops = torch.from_numpy(np.array(pixels[start:start + args.batch], np.float32))
            feature = extract(crops.to(args.device))
            tokens = torch.cat((F.adaptive_avg_pool2d(feature, (4, 3)).flatten(2),
                                feature.mean((2, 3))[:, :, None]), 2).transpose(1, 2)
            pooled.append(tokens.half().cpu())
        values = torch.cat(pooled).numpy()
        if values.shape[1:] != (13, 1280) or not np.isfinite(values).all():
            raise ValueError(f'Bad pooled tokens for {uid}: {values.shape}')
        np.save(destination, values)
        print(uid, values.shape, flush=True)


if __name__ == '__main__':
    main()
