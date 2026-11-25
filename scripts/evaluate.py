#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the held-out split.

The test split is opened only by this script. Weights come from the final epoch
of training, without any selection on test performance. Predictions are scored
with root-relative geometry, region and surface contact, and regional force.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myobridge.data import TrialDataset, collate  # noqa: E402
from myobridge.mano import EmbeddedMANO  # noqa: E402
from myobridge.metrics import Metrics, geodesic_neighborhood  # noqa: E402
from myobridge.model import build_model  # noqa: E402
from myobridge.utils import load_partition, read_json, set_seed  # noqa: E402

OUTPUTS = ('joints', 'mesh', 'regional_contact', 'regional_force',
           'vertex_contact', 'vertex_force')


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--normalization', type=Path,
                        help='Same train-only normalization JSON used during training; defaults to <data>/normalization.json')
    parser.add_argument('--protocol', required=True,
                        choices=('cross_trial', 'cross_session', 'cross_subject'))
    parser.add_argument('--mano', type=Path, required=True)
    parser.add_argument('--partition', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--visual', default='wilor', choices=('wilor', 'dino'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=0,
                        help='Checkpoint epoch to load, 0 means the stored final epoch')
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    splits = read_json(args.splits)
    test_ids = splits['protocols'][args.protocol]['test']
    normalization_path = args.normalization or args.data / 'normalization.json'
    normalization = read_json(normalization_path)
    normalization_protocol = normalization.get('protocol')
    if normalization_protocol and normalization_protocol != args.protocol:
        raise ValueError(f'Normalization was computed for {normalization_protocol}, '
                         f'not requested protocol {args.protocol}')

    mano = EmbeddedMANO.from_npz(args.mano).to(device)
    partition = load_partition(args.partition)
    model = build_model(mano, partition, seed=args.seed, visual=args.visual).to(device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    expected_epoch = args.epoch or saved['epoch']
    if saved['epoch'] != expected_epoch:
        raise ValueError(f'Checkpoint is at epoch {saved["epoch"]}, requested {expected_epoch}')
    missing, unexpected = model.load_state_dict(saved['model'], strict=False)
    if unexpected or any(not name.startswith('mano.') for name in missing):
        raise RuntimeError(f'Checkpoint mismatch, missing {missing[:3]}, unexpected {unexpected[:3]}')
    model.eval()

    near = geodesic_neighborhood(mano.v_template.cpu().numpy(), mano.faces_tensor.cpu().numpy())
    metrics = Metrics(partition, near, device)
    data = TrialDataset(args.data, test_ids, visual=args.visual,
                        normalization=normalization)
    loader = DataLoader(data, batch_size=args.batch, sampler=range(len(data)),
                        num_workers=args.workers, collate_fn=collate,
                        pin_memory=device == 'cuda')
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    frames = 0
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for host in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in host.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                prediction = model(emg=batch['emg'], is_left=batch['is_left'],
                                   wtok=batch.get('wtok'), dino=batch.get('dino'))
            if not all(torch.isfinite(prediction[key]).all() for key in OUTPUTS):
                raise FloatingPointError('Nonfinite prediction')
            metrics.update(prediction, batch)
            frames += len(batch['is_left'])
    if frames != len(data):
        raise RuntimeError(f'Scored {frames} of {len(data)} frames')
    if not all(torch.equal(before[key], value.detach().cpu())
               for key, value in model.state_dict().items()):
        raise RuntimeError('Model state changed during evaluation')

    result = dict(complete=True, protocol=args.protocol, seed=args.seed,
                  visual=args.visual, checkpoint_epoch=saved['epoch'],
                  weights='raw final epoch', selection_by_test=False,
                  normalization_file=str(normalization_path),
                  normalization_protocol=normalization_protocol or 'unspecified',
                  coverage=dict(frames=frames, trials=len(data.trials)),
                  metrics=metrics.result())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    for key, value in result['metrics'].items():
        if isinstance(value, float):
            print(f'{key}: {value:.4f}')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
