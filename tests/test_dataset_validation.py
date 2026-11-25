import json

import numpy as np

from scripts.validate_dataset import TARGET_FIELDS, validate_dataset


def make_dataset(root):
    trial_ids = ('trial-a', 'trial-b')
    (root / 'features' / 'wilor').mkdir(parents=True)
    (root / 'targets').mkdir()
    (root / 'membership.json').write_text(json.dumps({
        'trials': [
            {'trial_id': trial_id, 'frames': 1}
            for trial_id in trial_ids
        ],
    }))
    (root / 'normalization.json').write_text(json.dumps({
        'mean': [0] * 18,
        'std': [1] * 18,
    }))
    split = {'train': ['trial-a'], 'test': ['trial-b']}
    (root / 'splits.json').write_text(json.dumps({
        'protocols': {
            'cross_trial': split,
            'cross_session': split,
            'cross_subject': split,
        },
    }))
    for trial_id in trial_ids:
        np.save(root / 'features' / 'wilor' / f'{trial_id}.npy',
                np.zeros((1, 13, 1280), dtype=np.float16))
        np.savez(root / 'targets' / f'{trial_id}.npz', **{
            name: np.zeros((1,), dtype=np.float32)
            for name in TARGET_FIELDS
        })


def test_valid_model_ready_dataset(tmp_path):
    make_dataset(tmp_path)

    assert validate_dataset(tmp_path) == []


def test_protocol_specific_normalization_path(tmp_path):
    make_dataset(tmp_path)
    specific = tmp_path / 'normalization-cross_trial.json'
    (tmp_path / 'normalization.json').rename(specific)
    assert validate_dataset(tmp_path, normalization_path=specific) == []


def test_rejects_split_leakage(tmp_path):
    make_dataset(tmp_path)
    split_path = tmp_path / 'splits.json'
    splits = json.loads(split_path.read_text())
    splits['protocols']['cross_trial']['test'].append('trial-a')
    split_path.write_text(json.dumps(splits))

    errors = validate_dataset(tmp_path)

    assert any('cross_trial: train/test leakage' in error for error in errors)


def test_rejects_missing_trial_file(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / 'targets' / 'trial-b.npz').unlink()

    errors = validate_dataset(tmp_path)

    assert any('Missing targets/trial-b.npz' in error for error in errors)
