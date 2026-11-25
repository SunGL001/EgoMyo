"""Lazy baseline inputs sharing the released observation timeline and masks."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from myobridge.data import INTEGER_FIELDS, TARGET_FIELDS
from .models import EMG_ONLY, LONG_EMG, RGB


class BaselineDataset(Dataset):
    def __init__(self, root, trial_ids, method, normalization=None, cache_trials=2):
        self.root = Path(root)
        membership = json.loads((self.root / 'membership.json').read_text())
        rows = {row['trial_id']: row for row in membership['trials']}
        self.trials = [rows[uid] for uid in trial_ids]
        self.frames = [(trial, frame) for trial, row in enumerate(self.trials) for frame in range(row['frames'])]
        self.method = method
        self.cache = OrderedDict()
        self.cache_trials = cache_trials
        self.normalization = normalization
        if normalization is not None:
            mean, std = np.asarray(normalization['mean']), np.asarray(normalization['std'])
            if mean.shape != (18,) or std.shape != (18,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError('Expected finite TRAIN-only per-channel normalization')

    def __len__(self):
        return len(self.frames)

    def load_trial(self, trial):
        if trial in self.cache:
            self.cache.move_to_end(trial)
            return self.cache[trial]
        row = self.trials[trial]
        uid = row['trial_id']
        with np.load(self.root / 'targets' / f'{uid}.npz', allow_pickle=False) as archive:
            values = {name: archive[name] for name in TARGET_FIELDS}
            emg_name = 'emg_2000' if self.method in LONG_EMG else 'emg'
            valid_name = 'input_valid_2000' if self.method in LONG_EMG else 'input_valid'
            if emg_name not in archive or valid_name not in archive:
                raise ValueError(f'{uid}: missing {emg_name}/{valid_name}; export with --include-long-emg')
            values['emg'] = archive[emg_name].astype(np.float32)
            values['input_valid'] = archive[valid_name].astype(bool)
        samples = 2000 if self.method in LONG_EMG else 400
        if values['emg'].shape != (row['frames'], 18, samples):
            raise ValueError('Baseline EMG layout mismatch')
        if self.method in LONG_EMG:
            for name in ('mesh_valid', 'joint_valid', 'surface_valid', 'force_valid', 'contact_valid', 'distribution_valid'):
                mask = values[name]
                values[name] = mask & values['input_valid'].reshape((-1,) + (1,) * (mask.ndim - 1))
        elif self.normalization is not None:
            mean = np.asarray(self.normalization['mean'], np.float32)[None, :, None]
            std = np.asarray(self.normalization['std'], np.float32)[None, :, None]
            values['emg'] = (values['emg'] - mean) / std
        if self.method in RGB:
            with np.load(self.root / 'crops' / uid / 'rgb.npz', allow_pickle=False) as archive:
                values.update({name: archive[name] for name in ('img', 'right', 'box_center', 'box_size', 'img_size')})
        elif self.method not in EMG_ONLY:
            values['visual'] = np.load(self.root / 'features' / 'dino' / f'{uid}.npy', mmap_mode='r', allow_pickle=False)
            if values['visual'].shape != (row['frames'], 34, 768):
                raise ValueError('Expected frozen DINO [frames,34,768]')
        if any(len(value) != row['frames'] for value in values.values()):
            raise ValueError('Baseline inputs and targets are not frame-aligned')
        self.cache[trial] = values
        while len(self.cache) > self.cache_trials:
            self.cache.popitem(last=False)
        return values

    def __getitem__(self, index):
        trial, frame = self.frames[index]
        values = self.load_trial(trial)
        return {name: torch.tensor(np.array(value[frame], copy=True),
                                   dtype=torch.bool if name in INTEGER_FIELDS or name == 'input_valid' else torch.float32)
                for name, value in values.items()}
