#!/usr/bin/env python3
"""Train MyoBridge on one split protocol and seed.

The visual stream is frozen WiLoR tokens by default. With ``--visual dino`` the
cached DINOv2 features pass through a trained residual adapter instead. The
checkpoint stores learned weights only, never the licensed MANO buffers.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myobridge.data import TrialBlockBatches, TrialDataset, collate  # noqa: E402
from myobridge.losses import augment, objective  # noqa: E402
from myobridge.mano import EmbeddedMANO  # noqa: E402
from myobridge.model import build_model  # noqa: E402
from myobridge.utils import (load_partition, read_json, save_checkpoint,  # noqa: E402
                             set_seed)


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='Dataset root with membership.json and normalization.json')
    parser.add_argument('--splits', type=Path, required=True, help='splits.json')
    parser.add_argument('--protocol', required=True,
                        choices=('cross_trial', 'cross_session', 'cross_subject'))
    parser.add_argument('--mano', type=Path, required=True, help='MANO buffer archive')
    parser.add_argument('--partition', type=Path, required=True,
                        help='anatomical_partition.npz with a 778-entry partition')
    parser.add_argument('--visual', default='wilor', choices=('wilor', 'dino'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--block', type=int, default=128)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--adapter-lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--clip-norm', type=float, default=5.)
    parser.add_argument('--out', type=Path, required=True, help='Run directory')
    parser.add_argument('--limit-trials', type=int, default=0,
                        help='Debug: keep only the first N training trials')
    return parser.parse_args()


def main():
    args = parse()
    set_seed(args.seed)
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    splits = read_json(args.splits)
    train_ids = splits['protocols'][args.protocol]['train']
    if args.limit_trials:
        train_ids = train_ids[:args.limit_trials]

    mano = EmbeddedMANO.from_npz(args.mano).to(device)
    partition = load_partition(args.partition)
    model = build_model(mano, partition, seed=args.seed, visual=args.visual).to(device)
    model.train()

    adapter = list(model.dino_adapter.parameters()) if args.visual == 'dino' else []
    adapter_ids = {id(parameter) for parameter in adapter}
    downstream = [parameter for parameter in model.parameters() if id(parameter) not in adapter_ids]
    groups = [dict(params=downstream, lr=args.lr, base_lr=args.lr)]
    if adapter:
        groups.append(dict(params=adapter, lr=args.adapter_lr, base_lr=args.adapter_lr))
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    data = TrialDataset(args.data, train_ids, visual=args.visual)
    sampler = TrialBlockBatches(data, args.batch, args.seed, args.block)
    loader = DataLoader(data, batch_sampler=sampler, num_workers=args.workers,
                        collate_fn=collate, pin_memory=device == 'cuda',
                        persistent_workers=args.workers > 0,
                        prefetch_factor=2 if args.workers else None)
    augmentation_rng = torch.Generator(device=device).manual_seed(40000 + args.seed)
    metadata = dict(protocol=args.protocol, seed=args.seed, visual=args.visual,
                    epochs=args.epochs, batch=args.batch, lr=args.lr,
                    weight_decay=args.weight_decay, train_trials=len(train_ids),
                    train_frames=len(data))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'train_config.json').write_text(json.dumps(metadata, indent=2))

    for epoch in range(1, args.epochs + 1):
        model.train()
        factor = min(1., epoch / 2) * .5 * (1 + math.cos(math.pi * (epoch - 1) / args.epochs))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * factor
        started = time.monotonic()
        steps, frames, loss_sum = 0, 0, 0.
        for host in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in host.items()}
            emg = augment(batch['emg'], training=True, generator=augmentation_rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                prediction = model(emg=emg, is_left=batch['is_left'],
                                   wtok=batch.get('wtok'), dino=batch.get('dino'))
                loss, terms = objective(model, prediction, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, step {steps}')
            with sdpa_kernel(SDPBackend.MATH):
                loss.backward()
            if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                       if parameter.grad is not None):
                raise FloatingPointError('Nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            steps += 1
            frames += len(batch['is_left'])
            loss_sum += float(loss.detach())
            if steps % 50 == 0:
                print(f'epoch {epoch} step {steps} frames {frames} loss {loss_sum / steps:.4f}',
                      flush=True)
        if frames != len(data):
            raise RuntimeError(f'Epoch covered {frames} of {len(data)} frames')
        save_checkpoint(args.out / 'checkpoint.pt', model, optimizer, epoch, metadata)
        (args.out / f'epoch_{epoch:03d}.json').write_text(json.dumps(dict(
            epoch=epoch, frames=frames, steps=steps, mean_loss=loss_sum / max(steps, 1),
            seconds=time.monotonic() - started,
            terms={key: float(value) for key, value in terms.items()}), indent=2))
        print(f'epoch {epoch}/{args.epochs} mean loss {loss_sum / max(steps, 1):.4f} '
              f'({time.monotonic() - started:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
