import torch

from conftest import synthetic_mano_buffers, synthetic_partition
from myobridge.losses import objective
from myobridge.mano import EmbeddedMANO
from myobridge.model import build_model


def make_batch(batch=2):
    generator = torch.Generator().manual_seed(1)
    return dict(
        emg=torch.randn(batch, 18, 400, generator=generator),
        wtok=torch.randn(batch, 13, 1280, generator=generator).half(),
        is_left=torch.zeros(batch, dtype=torch.bool),
        mesh=torch.randn(batch, 778, 3, generator=generator) * .01,
        joints=torch.randn(batch, 21, 3, generator=generator) * .01,
        regional_force=torch.rand(batch, 7, generator=generator) * 30,
        regional_contact=torch.rand(batch, 7, generator=generator) > .5,
        vertex_force=torch.rand(batch, 778, generator=generator),
        vertex_contact=torch.rand(batch, 778, generator=generator) > .5,
        surface_regional_force=torch.rand(batch, 7, generator=generator) * 20,
        mesh_valid=torch.ones(batch, dtype=torch.bool),
        joint_valid=torch.ones(batch, 21, dtype=torch.bool),
        force_valid=torch.ones(batch, 7, dtype=torch.bool),
        contact_valid=torch.ones(batch, 7, dtype=torch.bool),
        surface_valid=torch.ones(batch, dtype=torch.bool),
        distribution_valid=torch.ones(batch, 7, dtype=torch.bool),
    )


def test_forward_shapes():
    mano = EmbeddedMANO(synthetic_mano_buffers())
    partition = synthetic_partition()
    model = build_model(mano, partition, seed=0, visual='wilor')
    batch = make_batch()
    output = model(emg=batch['emg'], is_left=batch['is_left'], wtok=batch['wtok'])
    assert output['mesh'].shape == (2, 778, 3)
    assert output['joints'].shape == (2, 21, 3)
    assert output['regional_force'].shape == (2, 7)
    assert output['vertex_force'].shape == (2, 778)
    assert output['gates'].shape == (2, 8, 1)
    for key in ('mesh', 'joints', 'regional_force', 'vertex_force'):
        assert torch.isfinite(output[key]).all()


def test_loss_is_finite_and_backward():
    mano = EmbeddedMANO(synthetic_mano_buffers())
    model = build_model(mano, synthetic_partition(), seed=0, visual='wilor')
    batch = make_batch()
    output = model(emg=batch['emg'], is_left=batch['is_left'], wtok=batch['wtok'])
    loss, terms = objective(model, output, batch)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert not any(name.startswith('mano.') for name, p in model.named_parameters())


def test_dino_variant_shapes():
    mano = EmbeddedMANO(synthetic_mano_buffers())
    model = build_model(mano, synthetic_partition(), seed=0, visual='dino')
    dino = torch.randn(2, 34, 768)
    output = model(emg=torch.randn(2, 18, 400), is_left=torch.zeros(2, dtype=torch.bool), dino=dino)
    assert output['mesh'].shape == (2, 778, 3)
