"""MyoBridge: multimodal hand pose, contact and regional force prediction.

The visual stream consumes frozen WiLoR backbone tokens that are cached once
per trial, so no image backbone is trained here. An optional variant replaces
them with cached DINOv2 features passed through a small residual adapter, which
is the only case where an extra trainable block sits on the visual path. No
pose or shape prior enters the model, and MANO is a frozen differentiable layer.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .components import (ContactSupportedForceField, GranularityMatchedEncoder,
                         NoFeatureGate, RegionalVertexLifting,
                         VisionAnchoredRouting, validate_partition)
from .mano import forward_mano

VISUAL_KINDS = ('wilor', 'dino')
WILOR_TOKENS = 13
WILOR_DIM = 1280
DINO_TOKENS = 34
DINO_DIM = 768


def rotation6d(values):
    values = values.reshape(*values.shape[:-1], 2, 3)
    a, b = values[..., 0, :], values[..., 1, :]
    first = F.normalize(a, dim=-1)
    second = F.normalize(b - (first * b).sum(-1, keepdim=True) * first, dim=-1)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


class PoolRouting(nn.Module):
    """Pooled regional states with a gated EMG residual.

    Eight query states are formed once from the pooled visual and EMG streams,
    and the EMG stream enters again through a sigmoid gate, so vision anchors
    the pose while EMG refines it.
    """

    def __init__(self, model, dim=256):
        super().__init__()
        object.__setattr__(self, 'model_ref', model)
        self.evidence = nn.Linear(dim, 8 * dim)
        self.gate = nn.Linear(dim, 1)

    def forward(self, encoded):
        model = self.model_ref
        visual = encoded['visual_fused'].mean(1)
        emg = encoded['emg_fused'].mean(1)
        initial = model.global_fusion(torch.cat((visual, emg), -1)).reshape(-1, 8, model.dim)
        gates = torch.sigmoid(self.gate(initial))
        states = initial + gates * self.evidence(emg).reshape(-1, 8, model.dim)
        return dict(states=states, gates=gates, localized=None)


class PoolEncoder(nn.Module):
    """Pass-through wrapper that keeps the encoder under one named child.

    It exists so that state dict keys stay stable when the routing head is
    swapped during model construction.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, visual, emg):
        return self.base(visual, emg)


class FeatureAdapter(nn.Module):
    """Residual adapter for cached DINOv2 tokens, zero initialized at output."""

    def __init__(self, seed=0, dim=DINO_DIM, bottleneck=192):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(98000 + seed)
            self.norm = nn.LayerNorm(dim)
            self.down = nn.Linear(dim, bottleneck)
            self.up = nn.Linear(bottleneck, dim)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        self.activation = nn.GELU()

    def forward(self, features):
        if features.shape[1:] != (DINO_TOKENS, DINO_DIM):
            raise ValueError(f'Expected DINO tokens [B,{DINO_TOKENS},{DINO_DIM}]')
        return features + self.up(self.activation(self.down(self.norm(features))))


class MyoBridge(nn.Module):
    """Final architecture: frozen visual tokens, causal EMG, MANO geometry."""

    def __init__(self, mano, partition, dim=256, heads=8, layers=4, visual='wilor'):
        super().__init__()
        if visual not in VISUAL_KINDS:
            raise ValueError(f'Unknown visual stream: {visual}')
        validate_partition(partition)
        if len(partition) != 778:
            raise ValueError('MANO vertex partition must have 778 entries')
        self.dim = dim
        self.visual = visual
        self.mano = mano
        self.encoder = GranularityMatchedEncoder(dim=dim, heads=heads, layers=layers,
                                                 temporal_refinement=True)
        self.router = VisionAnchoredRouting(dim=dim, heads=heads, uptake_gate=False)
        self.global_fusion = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 8 * dim))
        self.pose = nn.Sequential(nn.LayerNorm(8 * dim), nn.Linear(8 * dim, dim * 2), nn.GELU(),
                                  nn.Linear(dim * 2, 106))
        nn.init.zeros_(self.pose[-1].weight)
        with torch.no_grad():
            self.pose[-1].bias.zero_()
            self.pose[-1].bias[:96] = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(16)
        self.lifting = RegionalVertexLifting(partition, dim=dim, heads=heads, regional_queries=True)
        self.field = ContactSupportedForceField(partition, dim=dim)
        self.direct_force = nn.Linear(dim, 1)
        nn.init.zeros_(self.direct_force.weight)
        nn.init.constant_(self.direct_force.bias, math.log(math.expm1(.01)))
        self.log_sigma = nn.Parameter(torch.zeros(4))

    def project_visual(self, wtok=None, dino=None):
        if self.visual == 'wilor':
            if wtok is None or wtok.shape[1:] != (WILOR_TOKENS, WILOR_DIM):
                raise ValueError(f'Expected WiLoR tokens [B,{WILOR_TOKENS},{WILOR_DIM}]')
            return self.wproj(wtok.float())
        if dino is None or dino.shape[1:] != (DINO_TOKENS, DINO_DIM):
            raise ValueError(f'Expected DINO features [B,{DINO_TOKENS},{DINO_DIM}]')
        return self.dino_adapter(dino.float())

    def forward(self, *, emg, is_left, wtok=None, dino=None, hard_contact_threshold=None):
        visual = self.project_visual(wtok=wtok, dino=dino)
        encoded = self.encoder(visual, emg)
        routing = self.router(encoded)
        states = routing['states']
        params = self.pose(states.flatten(1))
        with torch.autocast(device_type=emg.device.type, enabled=False):
            rotations = rotation6d(params[:, :96].float().reshape(-1, 16, 6))
            betas = params[:, 96:].float()
            geometry = forward_mano(self.mano, rotations, betas, is_left=is_left)
            wrist = geometry['joints'][:, :1]
            mesh = geometry['vertices'] - wrist
            joints = geometry['joints'] - wrist
        features = self.lifting(mesh, states, encoded['visual_fused'], encoded['emg_fused'])
        with torch.autocast(device_type=emg.device.type, enabled=False):
            field = self.field(states.float(), features.float(), hard_threshold=hard_contact_threshold)
        return dict(**field, mesh=mesh, joints=joints, rotations=rotations, betas=betas,
                    gates=routing['gates'])


def _encoder_without_gate(dim, temporal=True):
    return NoFeatureGate(channels=18, dim=dim, temporal_refinement=temporal)


def build_model(mano, partition, seed=0, dim=256, heads=8, layers=4, visual='wilor'):
    """Construct the final model with the initialization order used in training.

    The base network, the routing head and the token projection are drawn from
    separate seeded generator scopes so that a trained model seed reproduces
    the same initialization for every component.
    """
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = MyoBridge(mano, partition, dim=dim, heads=heads, layers=layers, visual='wilor')
        model.encoder.emg_encoder = _encoder_without_gate(dim)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(80000 + seed)
        model.router = PoolRouting(model, dim=dim)
        model.encoder = PoolEncoder(model.encoder)
    if visual == 'wilor':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(93000 + seed)
            model.wproj = nn.Sequential(nn.LayerNorm(WILOR_DIM), nn.Linear(WILOR_DIM, 768))
        model.visual = 'wilor'
    elif visual == 'dino':
        model.dino_adapter = FeatureAdapter(seed)
        model.visual = 'dino'
    else:
        raise ValueError(f'Unknown visual stream: {visual}')
    return model
