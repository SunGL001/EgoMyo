"""Past-only EMG adaptations backed by externally acquired upstream components."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .cached import DirectHeads
from .prepare_sources import EGOEMG_COMMIT, FILES, META_COMMIT, META_SHA256


def source(root, name):
    root = Path(root)
    receipt = json.loads((root / 'receipt.json').read_text())
    if receipt['egoemg_commit'] != EGOEMG_COMMIT or receipt['meta_commit'] != META_COMMIT:
        raise ValueError('Unexpected upstream source revision')
    expected = next(row['sha256'] for row in receipt['files'] if row['file'] == name)
    if name in FILES and expected != FILES[name][1]:
        raise ValueError('Pinned source receipt changed')
    if name == 'meta_networks.py' and expected != META_SHA256:
        raise ValueError('Pinned Meta source receipt changed')
    path = root / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Upstream source bytes changed: ' + name)
    spec = importlib.util.spec_from_file_location('_egomyo_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NeuroPose(nn.Module):
    def __init__(self, root):
        super().__init__()
        upstream = source(root, 'neuropose.py')
        self.native = upstream.NeuroPose(
            encoder_blocks=[upstream.EncoderBlock(1, 32, (3, 2), (10, 2)),
                            upstream.EncoderBlock(32, 128, (3, 2), (8, 2)),
                            upstream.EncoderBlock(128, 256, (3, 2), (4, 4))],
            residual_blocks=[upstream.ResidualBlock(256, 256, (3, 2), 3) for _ in range(5)],
            decoder_blocks=[upstream.DecoderBlock(256, 128, (3, 2), (10, 4)),
                            upstream.DecoderBlock(128, 32, (3, 2), (8, 4)),
                            upstream.DecoderBlock(32, 1, (3, 2), (4, 2))],
            linear_in_channels=64, out_channels=63)
        self.native.network[-1].network[2] = nn.Identity()
        self.added = DirectHeads(64, ('mesh', 'regional_contact', 'regional_force', 'vertex_contact', 'vertex_force'))

    def forward(self, batch):
        values = F.pad(batch['emg'].float(), (0, 240, 0, 14))
        values = self.native.network(values[:, None].swapaxes(-1, -2))
        features = values.swapaxes(-2, -3).flatten(-2)[:, 1999]
        joints = self.native.linear(features).reshape(-1, 21, 3)
        return dict(joints=joints - joints[:, :1], **self.added(features))


class EMGFormer(nn.Module):
    def __init__(self, root):
        super().__init__()
        tds = source(root, 'tds.py')
        transformer = source(root, 'transformer.py')
        pool = source(root, 'pooling.py')
        self.featurizer = tds.TdsNetwork(
            conv_blocks=[tds.Conv1dBlock(18, 256, 11, 5), tds.Conv1dBlock(256, 256, 5, 2)],
            tds_stages=[tds.TdsStage(256, 9, 5, num_blocks=1, channels=8, feature_width=32, kernel_width=5),
                        tds.TdsStage(256, 3, 1, num_blocks=1, channels=8, feature_width=32, kernel_width=3)],
            se=dict(enable=True, reduction=4, residual=True, mode='global'))
        self.decoder = transformer.TransformerDecoder(256, 256, 4, 3, 512, dropout=.15,
                                                      causal=False, pos_encoding='rope')
        self.pool = pool.TemporalAttentionPool(256)
        self.vision = nn.Linear(768, 256)
        self.vision_head = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Dropout(.1), nn.Linear(512, 63))
        self.fusion = nn.Sequential(nn.Conv1d(512, 512, 1), nn.GELU(), nn.Dropout(.1), nn.Conv1d(512, 256, 1))
        self.residual = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Dropout(.1), nn.Linear(512, 63))
        nn.init.normal_(self.residual[-1].weight, mean=0., std=1e-5)
        nn.init.zeros_(self.residual[-1].bias)
        self.added = DirectHeads(256, ('mesh', 'regional_contact', 'regional_force', 'vertex_contact', 'vertex_force'))

    def forward(self, batch):
        emg = self.pool(self.decoder(self.featurizer(F.pad(batch['emg'].float(), (0, 4)))))
        visual = batch['visual'][:, 2:].float().mean(1)
        features = self.fusion(torch.cat((emg, self.vision(visual)), -1).unsqueeze(-1)).squeeze(-1)
        joints = (self.vision_head(visual) + self.residual(features)).reshape(-1, 21, 3)
        return dict(joints=joints - joints[:, :1], **self.added(features))


class VEMG2Pose(nn.Module):
    def __init__(self, root):
        super().__init__()
        upstream = source(root, 'meta_networks.py')
        self.network = upstream.TdsNetwork(
            [upstream.Conv1dBlock(18, 256, 11, 5), upstream.Conv1dBlock(256, 256, 5, 2)],
            [upstream.TdsStage(in_channels=256, in_conv_kernel_width=17, in_conv_stride=4,
                               num_blocks=2, channels=16, feature_width=16, kernel_width=9),
             upstream.TdsStage(in_channels=256, in_conv_kernel_width=9, in_conv_stride=2,
                               num_blocks=2, channels=16, feature_width=16, kernel_width=5, out_channels=64)])
        self.decoder = upstream.MLP(in_channels=127, layer_sizes=[512, 512], out_channels=63,
                                    layer_norm=True, scale=.01)
        self.mesh = nn.Sequential(nn.Linear(127, 512), nn.ReLU(), nn.Linear(512, 2334))
        self.added = DirectHeads(127, ('regional_contact', 'regional_force', 'vertex_contact', 'vertex_force'))

    def forward(self, batch):
        features = self.network(batch['emg'].float())
        state = torch.zeros(len(features), 63, device=features.device, dtype=features.dtype)
        for index in range(features.shape[-1]):
            state = self.decoder(torch.cat((features[:, :, index], state), -1))
        shared = torch.cat((features[:, :, -1], state), -1)
        joints = state.reshape(-1, 21, 3)
        return dict(joints=joints - joints[:, :1], mesh=self.mesh(shared).reshape(-1, 778, 3), **self.added(shared))
