#!/usr/bin/env python3
"""Check a downloaded EgoMyo export against the MyoBridge training layout."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

PROTOCOLS = ('cross_trial', 'cross_session', 'cross_subject')
TARGET_FIELDS = {
    'emg', 'mesh', 'joints', 'regional_force', 'regional_contact', 'vertex_force',
    'vertex_contact', 'surface_regional_force', 'mesh_valid', 'joint_valid',
    'force_valid', 'contact_valid', 'surface_valid', 'distribution_valid', 'is_left',
}


def validate_dataset(root, visual='wilor', normalization_path=None):
    root = Path(root)
    normalization_path = Path(normalization_path) if normalization_path else root / 'normalization.json'
    errors = []
    required_files = ('membership.json', 'splits.json')
    for relative_path in required_files:
        if not (root / relative_path).is_file():
            errors.append(f'Missing {relative_path}')
    if not normalization_path.is_file():
        errors.append(f'Missing normalization: {normalization_path}')
    if errors:
        return errors

    try:
        membership = json.loads((root / 'membership.json').read_text())
        normalization = json.loads(normalization_path.read_text())
        splits = json.loads((root / 'splits.json').read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [f'Cannot read dataset metadata: {error}']

    trials = membership.get('trials')
    if not isinstance(trials, list) or not trials:
        return ['membership.json must contain a non-empty trials list']

    trial_rows = {}
    for row in trials:
        if not isinstance(row, dict):
            errors.append('Every trial entry must be an object')
            continue
        trial_id = row.get('trial_id')
        if (not isinstance(trial_id, str) or not trial_id or
                Path(trial_id).name != trial_id or '/' in trial_id or
                '\\' in trial_id or trial_id in {'.', '..'}):
            errors.append(f'Invalid trial_id: {trial_id!r}')
            continue
        if trial_id in trial_rows:
            errors.append(f'Duplicate trial_id: {trial_id}')
            continue
        frames = row.get('frames')
        if not isinstance(frames, int) or frames <= 0:
            errors.append(f'{trial_id}: frames must be a positive integer')
            continue
        trial_rows[trial_id] = frames

    means = normalization.get('mean')
    deviations = normalization.get('std')
    if (not isinstance(means, list) or not isinstance(deviations, list) or
            len(means) != 18 or len(deviations) != 18):
        errors.append('normalization.json must provide 18-element mean and std arrays')
    elif any(not isinstance(value, (int, float)) or not math.isfinite(value)
             for value in means + deviations) or any(value <= 0 for value in deviations):
        errors.append('Normalization values must be finite and std values must be positive')

    protocol_splits = splits.get('protocols')
    if not isinstance(protocol_splits, dict):
        errors.append('splits.json must contain a protocols object')
        protocol_splits = {}
    trial_ids = set(trial_rows)
    for protocol in PROTOCOLS:
        split = protocol_splits.get(protocol)
        if not isinstance(split, dict):
            errors.append(f'Missing split protocol: {protocol}')
            continue
        train = split.get('train')
        test = split.get('test')
        if not isinstance(train, list) or not isinstance(test, list):
            errors.append(f'{protocol}: train and test must be lists')
            continue
        train_ids, test_ids = set(train), set(test)
        if len(train_ids) != len(train) or len(test_ids) != len(test):
            errors.append(f'{protocol}: duplicate trial ids in split')
        if train_ids & test_ids:
            errors.append(f'{protocol}: train/test leakage')
        if train_ids | test_ids != trial_ids:
            errors.append(f'{protocol}: split does not cover exactly the membership trials')

    expected_shapes = {'wilor': (13, 1280), 'dino': (34, 768)}
    feature_root = root / 'features' / visual
    target_root = root / 'targets'
    for trial_id, frame_count in trial_rows.items():
        feature_path = feature_root / f'{trial_id}.npy'
        target_path = target_root / f'{trial_id}.npz'
        if not feature_path.is_file():
            errors.append(f'Missing {feature_path.relative_to(root)}')
        else:
            try:
                features = np.load(feature_path, mmap_mode='r', allow_pickle=False)
                if features.shape != (frame_count, *expected_shapes[visual]):
                    errors.append(f'{trial_id}: {visual} feature shape {features.shape} is invalid')
                del features
            except (OSError, ValueError) as error:
                errors.append(f'{trial_id}: cannot read {visual} features: {error}')
        if not target_path.is_file():
            errors.append(f'Missing {target_path.relative_to(root)}')
            continue
        try:
            with np.load(target_path, allow_pickle=False) as archive:
                missing_fields = TARGET_FIELDS - set(archive.files)
            if missing_fields:
                errors.append(f'{trial_id}: target archive lacks {sorted(missing_fields)}')
        except (OSError, ValueError) as error:
            errors.append(f'{trial_id}: cannot read targets: {error}')

    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='Downloaded model-ready dataset directory')
    parser.add_argument('--visual', choices=('wilor', 'dino'), default='wilor')
    parser.add_argument('--normalization', type=Path, help='Protocol-specific TRAIN-only statistics')
    args = parser.parse_args()
    errors = validate_dataset(args.data, args.visual, args.normalization)
    if errors:
        print(f'Dataset validation failed with {len(errors)} issue(s):')
        for error in errors[:50]:
            print(f'- {error}')
        if len(errors) > 50:
            print(f'- ... and {len(errors) - 50} more')
        raise SystemExit(1)
    print(f'Dataset layout is valid for {args.visual} features: {args.data.resolve()}')


if __name__ == '__main__':
    main()
