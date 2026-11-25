"""Building blocks of MyoBridge.

Every operator that touches EMG is causal. Electrode identity is preserved by
running shared temporal filters per channel, so no convolution runs across the
electrode axis. The anatomical partition maps MANO vertices to seven palmar
regions and stays fixed across training and evaluation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def forward(self, values):
        padding = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(F.pad(values, (padding, 0)))


class TemporalResidual(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.depthwise = CausalConv1d(dim, dim, 3, groups=dim, dilation=dilation)
        self.pointwise = nn.Conv1d(dim, dim, 1)

    def forward(self, values):
        return values + self.pointwise(F.gelu(self.depthwise(values)))


class CausalChannelEMGEncoder(nn.Module):
    """Causal per-electrode temporal encoder.

    Inputs are ``[batch, electrode, sample]`` windows ending at the observed
    frame. The shared temporal stack runs independently along each electrode's
    time axis, and the electrode embedding keeps channel identity explicit.
    """

    def __init__(self, channels=18, dim=256, temporal_refinement=True):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.base = CausalConv1d(1, dim, 15, stride=10)
        self.refinement = (nn.Sequential(TemporalResidual(dim, 2), TemporalResidual(dim, 4))
                           if temporal_refinement else nn.Identity())
        self.channel_embedding = nn.Parameter(torch.empty(channels, dim))
        nn.init.normal_(self.channel_embedding, std=.02)
        self.norm = nn.LayerNorm(dim)
        self.feature_gate = nn.Linear(dim, dim)

    def temporal_features(self, samples):
        if samples.ndim != 3 or samples.shape[1] != self.channels or samples.shape[2] < 1:
            raise ValueError('Expected nonempty [batch, electrode, sample] windows')
        batch, channels, length = samples.shape
        base = F.gelu(self.base(samples.reshape(batch * channels, 1, length)))
        encoded = self.refinement(base)
        return encoded.reshape(batch, channels, self.dim, encoded.shape[-1])

    def forward(self, samples):
        features = self.temporal_features(samples).mean(-1) + self.channel_embedding
        features = self.norm(features)
        return torch.sigmoid(self.feature_gate(features)) * features


class NoFeatureGate(CausalChannelEMGEncoder):
    """Encoder without the per-channel feature gate.

    The base constructor runs unchanged, so parameter names and initialization
    draws match the gated encoder apart from the removed gate.
    """

    def __init__(self, channels=18, dim=256, temporal_refinement=True):
        super().__init__(channels=channels, dim=dim, temporal_refinement=temporal_refinement)
        del self.feature_gate

    def forward(self, samples):
        return self.norm(self.temporal_features(samples).mean(-1) + self.channel_embedding)


class MixtureOfTransformerBlock(nn.Module):
    """Shared attention with separate visual and EMG feed-forward experts."""

    def __init__(self, dim=256, heads=8, expansion=4, dropout=0.):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        def expert():
            return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * expansion),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * expansion, dim))

        self.visual_expert = expert()
        self.emg_expert = expert()

    def forward(self, visual, emg):
        count = visual.shape[1]
        joined = torch.cat((visual, emg), dim=1)
        normalized = self.attention_norm(joined)
        joined = joined + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        visual, emg = joined[:, :count], joined[:, count:]
        return visual + self.visual_expert(visual), emg + self.emg_expert(emg)


class GranularityMatchedEncoder(nn.Module):
    def __init__(self, visual_dim=768, dim=256, channels=18, heads=8, layers=4,
                 temporal_refinement=True):
        super().__init__()
        self.visual_projection = nn.Sequential(nn.Linear(visual_dim, dim), nn.LayerNorm(dim))
        self.emg_encoder = CausalChannelEMGEncoder(channels, dim, temporal_refinement)
        self.blocks = nn.ModuleList(MixtureOfTransformerBlock(dim, heads) for _ in range(layers))

    def forward(self, visual_tokens, emg_samples):
        visual_anchor = self.visual_projection(visual_tokens)
        visual_fused = visual_anchor
        emg_fused = self.emg_encoder(emg_samples)
        for block in self.blocks:
            visual_fused, emg_fused = block(visual_fused, emg_fused)
        return dict(visual_anchor=visual_anchor, visual_fused=visual_fused, emg_fused=emg_fused)


class VisionAnchoredRouting(nn.Module):
    def __init__(self, dim=256, heads=8, uptake_gate=True):
        super().__init__()
        self.query = nn.Parameter(torch.empty(8, dim))
        nn.init.normal_(self.query, std=.02)
        self.vision_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.emg_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Linear(dim, 1) if uptake_gate else None
        self.uptake_gate = uptake_gate

    def forward(self, visual_anchor, emg_memory):
        query = self.query.unsqueeze(0).expand(visual_anchor.shape[0], -1, -1)
        localized = query + self.vision_attention(
            query, visual_anchor, visual_anchor, need_weights=False)[0]
        gates = (torch.sigmoid(self.gate(localized)) if self.uptake_gate
                 else torch.ones_like(localized[..., :1]))
        evidence = self.emg_attention(localized, emg_memory, emg_memory, need_weights=False)[0]
        return dict(states=localized + gates * evidence, localized=localized, gates=gates)


def validate_partition(vertex_region):
    if vertex_region.ndim != 1 or vertex_region.dtype != torch.long:
        raise ValueError('Vertex partition must be a one-dimensional int64 tensor')
    if not len(vertex_region) or torch.any((vertex_region < -1) | (vertex_region > 6)):
        raise ValueError('Vertex region IDs must be -1 (unassigned) or 0..6')
    if any(not torch.any(vertex_region == region) for region in range(7)):
        raise ValueError('Every region must have at least one explicitly assigned vertex')


class RegionalVertexLifting(nn.Module):
    def __init__(self, vertex_region, dim=256, heads=8, regional_queries=True):
        super().__init__()
        validate_partition(vertex_region)
        self.register_buffer('vertex_region', vertex_region.clone())
        self.embedding = nn.Parameter(torch.empty(len(vertex_region), dim))
        nn.init.normal_(self.embedding, std=.02)
        self.position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.regional_queries = regional_queries

    def forward(self, predicted_mesh, states, visual_memory, emg_memory):
        if predicted_mesh.shape != (states.shape[0], len(self.vertex_region), 3) or states.shape[1] != 8:
            raise ValueError('Expected a predicted [B,V,3] mesh and seven regional plus one global state')
        query = self.embedding + self.position(predicted_mesh)
        if self.regional_queries:
            owner = torch.where(self.vertex_region < 0, 7, self.vertex_region)
            query = query + states[:, owner]
        memory = torch.cat((visual_memory, emg_memory, states), dim=1)
        return query + self.attention(query, memory, memory, need_weights=False)[0]


def allocate_contact_supported_force(contact_logits, scores, magnitude, vertex_region,
                                     hard_threshold=None):
    """Allocate each regional magnitude over the vertices of its region.

    Soft training probabilities stay strictly positive, so they are not a claim
    of zero force below a reporting threshold. ``hard_threshold`` produces
    exactly zero force off the binary support for an inference view.
    """
    validate_partition(vertex_region)
    masks = torch.arange(7, device=vertex_region.device)[:, None] == vertex_region[None]
    if not torch.isfinite(magnitude).all() or torch.any(magnitude < 0):
        raise ValueError('Regional magnitude must be finite and nonnegative')
    return _allocate_from_masks(contact_logits, scores, magnitude, masks, hard_threshold)


def _allocate_from_masks(contact_logits, scores, magnitude, masks, hard_threshold):
    if (contact_logits.shape != scores.shape or contact_logits.ndim != 2
            or contact_logits.shape[1] != masks.shape[1]
            or magnitude.shape != (contact_logits.shape[0], 7)):
        raise ValueError('Inconsistent vertex/region allocation shapes')
    if hard_threshold is not None and not 0 < hard_threshold < 1:
        raise ValueError('Hard contact threshold must be strictly between zero and one')
    logits, scores, magnitude = contact_logits.float(), scores.float(), magnitude.float()
    probability = torch.sigmoid(logits)
    support = masks[None].expand(len(probability), -1, -1)
    if hard_threshold is not None:
        support = support & (probability[:, None] >= hard_threshold)
    log_weight = F.logsigmoid(logits) + scores
    weights = log_weight[:, None].expand_as(support).masked_fill(~support, -torch.inf)
    has_weight = torch.isfinite(weights).any(-1, keepdim=True)
    weights = torch.where(has_weight, weights, torch.zeros_like(weights))
    proportions = torch.softmax(weights, dim=-1) * support * has_weight
    regional_contact = (probability[:, None] * support).amax(-1)
    regional_force = regional_contact * magnitude
    return dict(vertex_contact=probability,
                vertex_distribution=proportions.sum(1),
                vertex_force=(regional_force[..., None] * proportions).sum(1),
                regional_contact=regional_contact, regional_force=regional_force)


class ContactSupportedForceField(nn.Module):
    def __init__(self, vertex_region, dim=256):
        super().__init__()
        validate_partition(vertex_region)
        self.register_buffer('vertex_region', vertex_region.clone())
        self.register_buffer('region_masks', torch.arange(7, device=vertex_region.device)[:, None]
                             == vertex_region[None])
        self.contact = nn.Linear(dim, 1)
        self.score = nn.Linear(dim, 1)
        self.magnitude = nn.Linear(dim, 1)
        nn.init.constant_(self.magnitude.bias, math.log(math.expm1(1.)))
        nn.init.zeros_(self.contact.bias)

    def forward(self, states, vertices, hard_threshold=None):
        if states.shape[1] != 8:
            raise ValueError('Expected seven regional states plus one global state')
        logits = self.contact(vertices).squeeze(-1)
        scores = self.score(vertices).squeeze(-1)
        magnitude = F.softplus(self.magnitude(states[:, :7]).squeeze(-1).float())
        result = _allocate_from_masks(logits, scores, magnitude, self.region_masks, hard_threshold)
        result.update(contact_logits=logits, allocation_scores=scores, magnitude=magnitude)
        return result
