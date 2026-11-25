"""Portable owner-adapted visual baselines and MoT/HBridge task architectures."""
import torch
from torch import nn
from torch.nn import functional as F

DIMS = dict(joints=63, mesh=2334, regional_contact=7, regional_force=7,
            vertex_contact=778, vertex_force=778)


def predictions(logits):
    output = {}
    for name, value in logits.items():
        output[name] = (value.sigmoid() if 'contact' in name else F.softplus(value)
                        if 'force' in name else value.reshape(len(value), -1, 3))
    if 'joints' in output:
        root = output['joints'][:, :1]
        output['joints'] = output['joints'] - root
        if 'mesh' in output:
            output['mesh'] = output['mesh'] - root
    return output


class DirectHeads(nn.Module):
    def __init__(self, width, names):
        super().__init__()
        self.readouts = nn.ModuleDict({name: nn.Linear(width, DIMS[name]) for name in names})

    def forward(self, features):
        return predictions({name: layer(features.float()) for name, layer in self.readouts.items()})


class TouchBaseline(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == 'egopressure':
            self.encoder = nn.Sequential(nn.Linear(1536, 512), nn.ReLU(),
                                         nn.Linear(512, 512), nn.ReLU())
            hidden = 512
        else:
            self.projection = nn.Linear(768, 256, bias=False)
            layer = nn.TransformerEncoderLayer(256, 8, 1024, dropout=.1,
                                               batch_first=True, norm_first=True, activation='gelu')
            self.encoder = nn.TransformerEncoder(layer, 1 if method == 'opentouch' else 2)
            self.norm = nn.LayerNorm(256)
            hidden = 384 if method == 'egotouch' else 512
            self.body = nn.Sequential(nn.Linear(256, hidden), nn.GELU(),
                                      nn.LayerNorm(hidden), nn.Dropout(.1))
        self.contact = nn.Linear(hidden, 7)
        self.force = nn.Linear(hidden, 7)
        self.added = DirectHeads(hidden, ('joints', 'mesh', 'vertex_contact', 'vertex_force'))

    def forward(self, batch):
        visual = batch['visual'].float()
        if self.method == 'egopressure':
            features = self.encoder(torch.cat((visual[:, 0], visual[:, 1]), -1))
        else:
            features = self.body(self.norm(self.encoder(self.projection(visual[:, 2:])).mean(1)))
        return dict(regional_contact=self.contact(features).sigmoid(),
                    regional_force=F.softplus(self.force(features)), **self.added(features))


class EMGTokens(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(18, 64, 9, stride=4, padding=4), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv1d(128, width, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(16))

    def forward(self, emg):
        return self.network(emg.float()).transpose(1, 2)


class Expert(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 768, bias=False)
        self.output = nn.Linear(256, width, bias=False)
        self.feedforward = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def project(self, values):
        return self.qkv(self.norm1(values)).chunk(3, -1)

    def finish(self, values, attended):
        values = values + self.output(attended)
        return values + self.feedforward(self.norm2(values))


class ExpertBlock(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.experts = nn.ModuleList([Expert(width) for width in widths])

    def attention(self, query, key, value):
        def split(values):
            return values.reshape(len(values), values.shape[1], 8, 32).transpose(1, 2)
        result = F.scaled_dot_product_attention(split(query), split(key), split(value), dropout_p=0.)
        return result.transpose(1, 2).flatten(2)

    def forward(self, streams, bridge):
        queries, keys, values = zip(*(expert.project(stream) for expert, stream in zip(self.experts, streams)))
        if bridge:
            attended = self.attention(torch.cat(queries, 1), torch.cat(keys, 1), torch.cat(values, 1))
            attended = attended.split([stream.shape[1] for stream in streams], 1)
        else:
            attended = [self.attention(query, key, value) for query, key, value in zip(queries, keys, values)]
        return [expert.finish(stream, result) for expert, stream, result in zip(self.experts, streams, attended)]


class FusionBaseline(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        emg_width = 192 if method == 'hbridge' else 256
        self.vision = nn.Linear(768, 256)
        self.emg = EMGTokens(emg_width)
        self.visual_position = nn.Parameter(torch.randn(1, 34, 256) * .02)
        self.emg_position = nn.Parameter(torch.randn(1, 16, emg_width) * .02)
        self.blocks = nn.ModuleList([ExpertBlock((256, emg_width)) for _ in range(6)])
        self.norms = nn.ModuleList([nn.LayerNorm(256), nn.LayerNorm(emg_width)])
        if method == 'hbridge':
            self.semantic = nn.Parameter(torch.randn(1, 16, emg_width) * .02)
            self.semantic_projection = nn.Linear(emg_width, 768)
        self.fusion = nn.Sequential(nn.Linear(256 + emg_width, 256), nn.GELU(), nn.LayerNorm(256))
        self.heads = DirectHeads(256, DIMS)
        self.last_semantic = None

    def forward(self, batch):
        streams = [self.vision(batch['visual'].float()) + self.visual_position,
                   self.emg(batch['emg']) + self.emg_position]
        if self.method == 'hbridge':
            streams[1] = torch.cat((streams[1], self.semantic.expand(len(streams[1]), -1, -1)), 1)
        for index, block in enumerate(self.blocks):
            streams = block(streams, self.method == 'mot' or index in (2, 3))
        streams = [norm(stream) for norm, stream in zip(self.norms, streams)]
        self.last_semantic = self.semantic_projection(streams[1][:, 16:]) if self.method == 'hbridge' else None
        features = self.fusion(torch.cat((streams[0].mean(1), streams[1][:, :16].mean(1)), -1))
        return self.heads(features)

    def semantic_loss(self, batch):
        if self.last_semantic is None:
            return batch['visual'].sum() * 0
        target = F.adaptive_avg_pool1d(batch['visual'][:, 2:].float().transpose(1, 2), 16).transpose(1, 2).detach()
        loss = 1 - F.cosine_similarity(self.last_semantic.float(), target, dim=-1).mean(1)
        valid = batch['input_valid'].bool()
        return torch.where(valid, loss, 0).sum() / valid.sum().clamp_min(1)
