"""Training objective and EMG augmentation.

Four prediction tasks are supervised: root-relative mesh and joints, surface
and regional contact, regional force, and the vertex-level force allocation.
Each task carries a learned log-variance, so an absent task contributes neither
a data loss nor an uncertainty reward. Geometry and allocation are evaluated in
FP32 because BF16 summation is inaccurate over 778 vertices.
"""
import torch
from torch.nn import functional as F


class Augmentation:
    """Channel-wise gain and offset noise with channel and modality dropout."""

    gain_std = .20
    offset_std = .10
    channel_dropout = .10
    modality_dropout = .10


def augment(samples, *, training, generator=None):
    if not training:
        return samples
    if samples.ndim != 3 or samples.shape[1] != 18:
        raise ValueError('Expected [batch, electrode, sample] EMG windows')
    shape = (*samples.shape[:2], 1)
    gain = torch.randn(shape, device=samples.device, generator=generator)
    offset = torch.randn(shape, device=samples.device, generator=generator)
    channels = torch.rand(shape, device=samples.device, generator=generator)
    modality = torch.rand((len(samples), 1, 1), device=samples.device, generator=generator)
    return ((samples * torch.exp(gain * Augmentation.gain_std) + offset * Augmentation.offset_std)
            * (channels >= Augmentation.channel_dropout)
            * (modality >= Augmentation.modality_dropout))


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (torch.where(mask.bool(), values, torch.zeros_like(values)) * mask).sum() / mask.sum().clamp_min(1)


def regional_contact_bce(probability, truth):
    # Region probabilities aggregate vertex support, so there is no native
    # region logit. CUDA prohibits probability BCE inside autocast, so the term
    # is computed explicitly in FP32 with autocast disabled.
    with torch.autocast(device_type=probability.device.type, enabled=False):
        return F.binary_cross_entropy(probability.float().clamp(1e-6, 1 - 1e-6),
                                      truth.float(), reduction='none')


def base_objective(model, prediction, target):
    """Masked mesh, contact, force and allocation losses with their variances."""
    mesh_mask = target['mesh_valid'][:, None, None].expand_as(target['mesh'])
    mesh = masked_mean((prediction['mesh'] - target['mesh']).abs(), mesh_mask)

    surface_mask = target['surface_valid'][:, None] & (model.lifting.vertex_region[None] >= 0)
    truth = target['vertex_contact'].float()
    probability = prediction['vertex_contact']
    bce = masked_mean(F.binary_cross_entropy_with_logits(
        prediction['contact_logits'], truth, reduction='none'), surface_mask)
    valid = surface_mask.to(probability.dtype)
    dice = 1 - (2 * (probability * truth * valid).sum(-1) + 1e-6) / (
        (probability * valid).sum(-1) + (truth * valid).sum(-1) + 1e-6)
    contact = bce + masked_mean(dice, target['surface_valid'])

    contact_valid = target['contact_valid']
    region_bce = regional_contact_bce(prediction['regional_contact'], target['regional_contact'])
    contact = contact + masked_mean(region_bce, contact_valid)

    force = masked_mean((prediction['regional_force'] - target['regional_force']).abs(),
                        target['force_valid'])

    owners = model.lifting.vertex_region.clamp_min(0)
    active = (target['distribution_valid'] & (target['surface_regional_force'] > 0)
              & target['regional_contact'].bool() & contact_valid)
    vertex_mask = active[:, owners] & (model.lifting.vertex_region[None] >= 0)
    denominator = target['surface_regional_force'][:, owners].clamp_min(1e-12)
    distribution = target['vertex_force'] / denominator
    cross_entropy = -distribution * prediction['vertex_distribution'].clamp_min(1e-12).log()
    # Normalize by the valid region count, not by the 778 vertices.
    allocation = torch.where(vertex_mask, cross_entropy,
                             torch.zeros_like(cross_entropy)).sum() / active.sum().clamp_min(1)

    losses = torch.stack((mesh, contact, force, allocation))
    present = torch.stack((target['mesh_valid'].any(),
                           target['surface_valid'].any() | contact_valid.any(),
                           target['force_valid'].any(), active.any())).to(losses.dtype)
    logs = model.log_sigma.clamp(-5, 5)
    total = ((.5 * torch.exp(-2 * logs) * losses + logs) * present).sum()
    return total, dict(mesh=mesh.detach(), contact=contact.detach(),
                       force=force.detach(), allocation=allocation.detach())


def objective(model, prediction, target):
    """Geometry term in root-relative metres plus the joint term.

    Absolute mesh error is replaced by a root-relative geometry loss measured
    in metres and scaled by a fixed factor, so the optimizer sees a comparable
    magnitude for mesh and joint errors.
    """
    total, terms = base_objective(model, prediction, target)
    mask = target['mesh_valid'][:, None, None].expand_as(target['mesh'])
    absolute_mesh = masked_mean((prediction['mesh'] - target['mesh']).abs(), mask)

    predicted_joints = prediction['joints'] - prediction['joints'][:, :1]
    target_joints = target['joints'] - target['joints'][:, :1]
    predicted_mesh = prediction['mesh'] - prediction['joints'][:, :1]
    target_mesh = target['mesh'] - target['joints'][:, :1]
    mesh = masked_mean((predicted_mesh - target_mesh).abs(), mask)

    joint_mask = target['joint_valid'][..., None].expand_as(target_joints)
    joint = masked_mean((predicted_joints - target_joints).abs(), joint_mask)

    geometry = 100. * (mesh + joint)
    absolute_present = target['mesh_valid'].any().to(total.dtype)
    present = (target['mesh_valid'].any() | target['joint_valid'].any()).to(total.dtype)
    log = model.log_sigma[0].clamp(-5, 5)
    total = total - (.5 * torch.exp(-2 * log) * absolute_mesh + log) * absolute_present
    total = total + (.5 * torch.exp(-2 * log) * geometry + log) * present
    return total, dict(terms, joint=joint.detach(), geometry=geometry.detach())
