#!/usr/bin/env python3
"""Optional: cache frozen DINOv2 features for the adapter variant.

Each trial folder stores ``dino.npy`` with RGB crops [N,3,224,224]. The output
is ``features/dino/<trial_id>.npy`` [N,34,768] in float16, made of the class
token, the patch mean and a 4x8 adaptive average pooling of the patch grid.
This variant is optional and is not used by the default WiLoR configuration.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--crops', type=Path, required=True,
                        help='Directory with <trial_id>/dino.npy crops')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--arch', default='dinov2_vitb14')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    model = torch.hub.load('facebookresearch/dinov2', args.arch)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu',
                                     weights_only=False), strict=False)
    model = model.to(args.device).eval().requires_grad_(False)
    membership = json.loads((args.data / 'membership.json').read_text())
    out = args.data / 'features' / 'dino'
    out.mkdir(parents=True, exist_ok=True)
    for row in membership['trials']:
        uid = row['trial_id']
        destination = out / f'{uid}.npy'
        if destination.exists():
            continue
        pixels = np.load(args.crops / uid / 'dino.npy', mmap_mode='r')
        if pixels.shape[1:] != (3, 224, 224):
            raise ValueError(f'Unexpected crop shape {pixels.shape} for {uid}')
        pooled = []
        for start in range(0, len(pixels), args.batch):
            crops = torch.from_numpy(np.array(pixels[start:start + args.batch], np.float32))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                 enabled=args.device == 'cuda'):
                output = model.forward_features(crops.to(args.device))
            patches = output['x_norm_patchtokens']
            grid = int(round(patches.shape[1] ** .5))
            grid = patches.transpose(1, 2).reshape(len(patches), -1, grid, grid)
            tokens = torch.cat((output['x_norm_clstoken'][:, None],
                                patches.mean(1)[:, None],
                                F.adaptive_avg_pool2d(grid, (4, 8)).flatten(2).transpose(1, 2)), 1)
            pooled.append(tokens.float().cpu())
        values = torch.cat(pooled).numpy().astype(np.float16)
        if values.shape[1:] != (34, 768) or not np.isfinite(values).all():
            raise ValueError(f'Bad pooled tokens for {uid}: {values.shape}')
        np.save(destination, values)
        print(uid, values.shape, flush=True)


if __name__ == '__main__':
    main()
