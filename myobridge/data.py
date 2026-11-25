"""Trial datasets for cached visual tokens, EMG windows and supervision targets.

The dataset release provides, for every trial, one cached feature file and one
``targets.npz``. This module loads them lazily with a small trial cache so that
training never holds the full corpus in memory. EMG windows are standardized
with TRAIN-only statistics computed by ``scripts/compute_normalization.py``.
"""
from collections import OrderedDict
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

EMG_CHANNELS = 18
EMG_SAMPLES = 400
JOINTS = 21
VERTICES = 778
REGIONS = 7
TARGET_FIELDS = ('mesh', 'joints', 'regional_force', 'regional_contact', 'vertex_force',
                 'vertex_contact', 'surface_regional_force', 'mesh_valid', 'joint_valid',
                 'force_valid', 'contact_valid', 'surface_valid', 'distribution_valid',
                 'is_left')
INTEGER_FIELDS = {'regional_contact', 'vertex_contact', 'mesh_valid', 'joint_valid',
                  'force_valid', 'contact_valid', 'surface_valid', 'distribution_valid',
                  'is_left'}


def read_json(path):
    import json
    return json.loads(Path(path).read_text())


class TrialDataset(Dataset):
    """Frames of the split, grouped by trial and returned in shuffled batches."""

    def __init__(self, root, trial_ids, visual='wilor', normalization=None,
                 cache_trials=4, features=None, targets=None):
        self.root = Path(root)
        rows = {row['trial_id']: row for row in read_json(self.root / 'membership.json')['trials']}
        missing = [uid for uid in trial_ids if uid not in rows]
        if missing:
            raise KeyError(f'Unknown trial ids: {missing[:3]}')
        self.visual = visual
        self.trials = [rows[uid] for uid in trial_ids]
        self.lengths = [int(row['frames']) for row in self.trials]
        self.features = Path(features) if features is not None else self.root / 'features' / visual
        self.targets = Path(targets) if targets is not None else self.root / 'targets'
        self.frames = [(trial, frame) for trial, count in enumerate(self.lengths)
                       for frame in range(count)]
        if normalization is None:
            normalization = read_json(self.root / 'normalization.json')
        self.normalization = (torch.tensor(normalization['mean'], dtype=torch.float32),
                              torch.tensor(normalization['std'], dtype=torch.float32))
        if self.normalization[1].shape != (EMG_CHANNELS,) or (self.normalization[1] <= 0).any():
            raise ValueError('Normalization statistics must be positive per-channel values')
        self.cache_trials = cache_trials
        self.cache = OrderedDict()

    @property
    def sampler_lengths(self):
        return self.lengths

    def __len__(self):
        return len(self.frames)

    def _trial(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        uid = self.trials[index]['trial_id']
        feature_path = self.features / f'{uid}.npy'
        target_path = self.targets / f'{uid}.npz'
        if not feature_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f'Missing cached inputs for trial {uid}')
        feature = np.load(feature_path, mmap_mode='r', allow_pickle=False)
        expected = (13, 1280) if self.visual == 'wilor' else (34, 768)
        if feature.shape[1:] != expected or len(feature) != self.lengths[index]:
            raise ValueError(f'Feature shape {feature.shape} does not match trial {uid}')
        with np.load(target_path, allow_pickle=False) as archive:
            values = {name: archive[name] for name in TARGET_FIELDS if name in archive.files}
            if 'emg' not in archive.files:
                raise ValueError(f'Targets for trial {uid} have no EMG windows')
            emg = archive['emg']
        if emg.shape[1:] != (EMG_CHANNELS, EMG_SAMPLES):
            raise ValueError(f'EMG window shape {emg.shape} is not [n,18,400]')
        self.cache[index] = (values, emg, feature)
        while len(self.cache) > self.cache_trials:
            self.cache.popitem(last=False)
        return self.cache[index]

    def __getitems__(self, indices):
        count = len(indices)
        output = OrderedDict()
        groups = OrderedDict()
        for position, index in enumerate(indices):
            trial, frame = self.frames[index]
            groups.setdefault(trial, []).append((position, frame))
        trial_index = np.empty(count, np.int64)
        frame_index = np.empty(count, np.int64)
        for trial, pairs in groups.items():
            values, emg, feature = self._trial(trial)
            positions, frames = (np.asarray(part, np.int64) for part in zip(*pairs))
            for name, array in values.items():
                if name == 'emg':
                    continue
                if name not in output:
                    output[name] = np.empty((count, *array.shape[1:]),
                                            np.bool_ if name in INTEGER_FIELDS else array.dtype)
                output[name][positions] = array[frames]
            if 'emg' not in output:
                output['emg'] = np.empty((count, EMG_CHANNELS, EMG_SAMPLES), np.float32)
            output['emg'][positions] = emg[frames]
            key = 'wtok' if self.visual == 'wilor' else 'dino'
            if key not in output:
                output[key] = np.empty((count, *feature.shape[1:]), np.float16)
            output[key][positions] = feature[frames]
            trial_index[positions] = trial
            frame_index[positions] = frames
        mean, std = (value.numpy() for value in self.normalization)
        output['emg'] = (output['emg'] - mean[None, :, None]) / std[None, :, None]
        output['is_left'] = output['is_left'].astype(np.bool_)
        output['trial_index'] = trial_index
        output['frame_index'] = frame_index
        return {name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in output.items()}


def collate(batch):
    return batch


class TrialBlockBatches(Sampler):
    """Shuffle within-trial frame blocks and then the block order.

    Consecutive frames share the same trials and feature files, so blocks keep
    the per-trial cache warm while batches still mix trials.
    """

    def __init__(self, dataset, batch_size, seed=0, block_size=128):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.block_size = block_size

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        blocks = []
        offset = 0
        for count in self.dataset.lengths:
            indices = rng.permutation(count) + offset
            blocks.extend(indices[i:i + self.block_size].tolist()
                          for i in range(0, count, self.block_size))
            offset += count
        rng.shuffle(blocks)
        batch = []
        for block in blocks:
            for index in block:
                batch.append(index)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
