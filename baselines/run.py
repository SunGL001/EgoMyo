"""Train to a fixed final epoch, or evaluate that checkpoint with shared metrics."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from myobridge.mano import EmbeddedMANO
from myobridge.metrics import Metrics, geodesic_neighborhood
from myobridge.utils import load_partition, set_seed
from .cached import DIMS
from .data import BaselineDataset
from .models import LONG_EMG, METHODS, RGB, build


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def loss(prediction, target, partition):
    parts = []
    for name in DIMS:
        truth = target[name].float()
        if name in ('joints', 'mesh'):
            mask = target['joint_valid'][:, :, None] if name == 'joints' else target['mesh_valid'][:, None, None]
            error = (prediction[name].float() - truth).abs() * 100
        elif name.startswith('regional'):
            mask = target['contact_valid' if 'contact' in name else 'force_valid']
            error = (F.binary_cross_entropy(prediction[name].float().clamp(1e-6, 1 - 1e-6), truth, reduction='none')
                     if 'contact' in name else (prediction[name].float() - truth).abs())
        else:
            mask = target['surface_valid'][:, None] & (partition[None] >= 0)
            if name == 'vertex_force':
                mask = mask & target['vertex_contact'].bool()
            error = (F.binary_cross_entropy(prediction[name].float().clamp(1e-6, 1 - 1e-6), truth, reduction='none')
                     if 'contact' in name else (prediction[name].float() - truth).abs())
        mask = mask.bool().expand_as(error)
        if mask.any():
            parts.append(torch.where(mask, error, 0).sum() / mask.sum())
    return sum(parts, prediction['joints'].sum() * 0) / max(len(parts), 1)


def learned_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if not name.startswith('native.native.mano.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'evaluate'))
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--protocol', choices=('cross_trial', 'cross_session', 'cross_subject'), required=True)
    parser.add_argument('--normalization', type=Path)
    parser.add_argument('--assets', type=Path, help='External component/RGB source and checkpoint configuration JSON')
    parser.add_argument('--partition', type=Path, required=True)
    parser.add_argument('--mano', type=Path, help='Licensed geometry buffers, required only for evaluation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error('epochs/batch must be positive; workers must be nonnegative')
    if args.out.exists():
        raise FileExistsError('Use a new output directory/file; prior results are never overwritten')
    if args.phase == 'evaluate' and (args.checkpoint is None or args.mano is None):
        parser.error('Evaluation requires --checkpoint and licensed --mano geometry')
    if args.method in LONG_EMG + RGB and args.assets is None:
        parser.error('This method requires --assets; see baselines/README.md')
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    split = read(args.splits)['protocols'][args.protocol]
    if not split['train'] or not split['test'] or set(split['train']) & set(split['test']):
        raise ValueError('Empty or overlapping TRAIN/TEST split')
    normalization = read(args.normalization) if args.normalization else None
    if normalization is not None and normalization.get('protocol') != args.protocol:
        raise ValueError('Normalization must belong to this protocol TRAIN split')
    if args.method in ('mot', 'hbridge') and normalization is None:
        parser.error('MoT/HBridge require --normalization from the protocol TRAIN split')
    preparation = args.data / 'preparation.json'
    if not preparation.is_file() or not read(preparation).get('complete'):
        raise ValueError('A completed prepare_targets.py export is required')
    assets = read(args.assets) if args.assets else {}
    asset_hashes = {name: digest(assets[name]) for name in ('checkpoint', 'config') if name in assets}
    if 'components' in assets:
        asset_hashes['components_receipt'] = digest(Path(assets['components']) / 'receipt.json')
    if 'source_root' in assets:
        source_root = Path(assets['source_root'])
        asset_hashes['upstream_commit'] = subprocess.check_output(
            ['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip()
        asset_hashes['upstream_tracked_diff_sha256'] = hashlib.sha256(subprocess.check_output(
            ['git', '-C', str(source_root), 'diff', 'HEAD', '--'])).hexdigest()
    provenance = dict(method=args.method, protocol=args.protocol, seed=args.seed,
                      splits_sha256=digest(args.splits), preparation_sha256=digest(preparation),
                      membership_sha256=digest(args.data / 'membership.json'),
                      normalization_sha256=digest(args.normalization) if args.normalization else None,
                      assets_sha256=digest(args.assets) if args.assets else None,
                      partition_sha256=digest(args.partition), external_asset_hashes=asset_hashes,
                      baseline_source_sha256={path.name: digest(path) for path in Path(__file__).parent.glob('*.py')})
    model = build(args.method, assets, args.seed).to(device)
    partition = torch.as_tensor(load_partition(args.partition), device=device)
    ids = split['train' if args.phase == 'train' else 'test']
    data = BaselineDataset(args.data, ids, args.method, normalization)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(data, batch_size=args.batch, shuffle=args.phase == 'train',
                        generator=generator, num_workers=args.workers, pin_memory=device == 'cuda')
    if args.phase == 'train':
        args.out.mkdir(parents=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            rate = args.lr * min(1., epoch / 2) * .5 * (1 + math.cos(math.pi * (epoch - 1) / args.epochs))
            for group in optimizer.param_groups:
                group['lr'] = rate
            total = frames = steps = 0
            for host in loader:
                batch = {name: value.to(device) for name, value in host.items()}
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch)
                value = loss(prediction, batch, partition) + model.extra_loss(batch)
                if not torch.isfinite(value):
                    raise FloatingPointError('Nonfinite training objective')
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                total += float(value.detach())
                frames += len(batch['is_left'])
                steps += 1
            if frames != len(data):
                raise RuntimeError('TRAIN coverage mismatch')
            history.append(dict(epoch=epoch, frames=frames, mean_loss=total / max(1, steps)))
            print(json.dumps(history[-1]), flush=True)
        torch.save(dict(model=learned_state(model), epoch=args.epochs, provenance=provenance), args.out / 'final.pt')
        (args.out / 'training.json').write_text(json.dumps(dict(
            provenance=provenance, epochs=args.epochs, recipe=vars_for_json(args), history=history,
            test_accessed=False, checkpoint='final.pt'), indent=2) + '\n')
    else:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if saved['provenance'] != provenance or saved['epoch'] != args.epochs:
            raise ValueError('Checkpoint/data/method/protocol/seed/epoch provenance mismatch')
        if set(saved['model']) != set(learned_state(model)):
            raise ValueError('Checkpoint parameter set mismatch')
        current = model.state_dict()
        current.update(saved['model'])
        model.load_state_dict(current, strict=True)
        model.eval()
        mano = EmbeddedMANO.from_npz(args.mano)
        near = geodesic_neighborhood(mano.v_template.numpy(), mano.faces_tensor.numpy())
        metrics = Metrics(partition, near, device)
        frames = 0
        with torch.inference_mode():
            for host in loader:
                batch = {name: value.to(device) for name, value in host.items()}
                prediction = model(batch)
                if not all(torch.isfinite(value).all() for value in prediction.values()):
                    raise FloatingPointError('Nonfinite prediction')
                metrics.update(prediction, batch)
                frames += len(batch['is_left'])
        if frames != len(data):
            raise RuntimeError('TEST coverage mismatch')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(provenance=provenance, metrics=metrics.result(), frames=frames,
            checkpoint_epoch=saved['epoch'], checkpoint_sha256=digest(args.checkpoint), selection_by_test=False),
            indent=2, allow_nan=False) + '\n')


def vars_for_json(args):
    return {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}


if __name__ == '__main__':
    main()
