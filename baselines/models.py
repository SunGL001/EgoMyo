"""Registry with explicit modalities; no target tensor enters a model forward."""
import torch
from torch import nn

from .cached import FusionBaseline, TouchBaseline

METHODS = ('opentouch', 'egotouch', 'egotactile', 'egopressure', 'mot', 'hbridge',
           'neuropose', 'emgformer', 'emg2pose', 'hamer', 'wilor')
LONG_EMG = ('neuropose', 'emgformer', 'emg2pose')
RGB = ('hamer', 'wilor')
EMG_ONLY = ('neuropose', 'emg2pose')


class PredictionOnly(nn.Module):
    def __init__(self, method, native):
        super().__init__()
        self.method = method
        self.native = native

    def forward(self, batch):
        if self.method in LONG_EMG and batch['emg'].shape[1:] != (18, 2000):
            raise ValueError('This baseline requires real past-only 18x2000 EMG')
        if self.method not in RGB + EMG_ONLY and batch['visual'].shape[1:] != (34, 768):
            raise ValueError('This baseline requires DINO tokens [34,768]')
        allowed = ('emg', 'visual', 'input_valid', 'img', 'right', 'box_center', 'box_size', 'img_size')
        prediction = self.native({name: batch[name] for name in allowed if name in batch})
        sign = torch.ones((len(batch['is_left']), 1, 3), device=batch['is_left'].device)
        sign[:, 0, 0] = torch.where(batch['is_left'].bool(), -1., 1.)
        for name in ('joints', 'mesh'):
            prediction[name] = prediction[name] * sign
        return prediction

    def extra_loss(self, batch):
        if hasattr(self.native, 'semantic_loss'):
            return self.native.semantic_loss({name: batch[name] for name in ('visual', 'input_valid')})
        return batch['emg'].sum() * 0


def build(method, assets=None, seed=0):
    if method not in METHODS:
        raise ValueError('Unknown baseline method')
    assets = assets or {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if method in ('opentouch', 'egotouch', 'egotactile', 'egopressure'):
            native = TouchBaseline(method)
        elif method in ('mot', 'hbridge'):
            native = FusionBaseline(method)
        elif method in LONG_EMG:
            from .emg import EMGFormer, NeuroPose, VEMG2Pose
            native = {'neuropose': NeuroPose, 'emgformer': EMGFormer, 'emg2pose': VEMG2Pose}[method](assets['components'])
        else:
            from .visual import VisualBaseline
            native = VisualBaseline(method, assets)
    return PredictionOnly(method, native)
