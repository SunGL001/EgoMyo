import torch

from conftest import synthetic_mano_buffers
from myobridge.mano import EmbeddedMANO
from myobridge.metrics import Metrics, geodesic_neighborhood


def test_perfect_prediction_scores_zero_error():
    mano = EmbeddedMANO(synthetic_mano_buffers())
    near = geodesic_neighborhood(mano.v_template.numpy(), mano.faces_tensor.numpy())
    partition = torch.arange(778, dtype=torch.int64) % 7
    metrics = Metrics(partition, near, 'cpu')
    target = dict(
        joints=torch.randn(2, 21, 3), mesh=torch.randn(2, 778, 3),
        joint_valid=torch.ones(2, 21, dtype=torch.bool), mesh_valid=torch.ones(2, dtype=torch.bool),
        regional_contact=torch.ones(2, 7, dtype=torch.bool),
        regional_force=torch.rand(2, 7) * 10, force_valid=torch.ones(2, 7, dtype=torch.bool),
        contact_valid=torch.ones(2, 7, dtype=torch.bool),
        vertex_contact=torch.ones(2, 778, dtype=torch.bool),
        vertex_force=torch.rand(2, 778), surface_valid=torch.ones(2, dtype=torch.bool))
    prediction = dict(joints=target['joints'].clone(), mesh=target['mesh'].clone(),
                      regional_contact=torch.ones(2, 7), regional_force=target['regional_force'].clone(),
                      vertex_contact=torch.ones(2, 778), vertex_force=target['vertex_force'].clone())
    metrics.update(prediction, target)
    result = metrics.result()
    assert result['mpjpe_root_relative_mm'] == 0.
    assert result['pve_root_relative_mm'] == 0.
    assert result['region_contact_macro_f1'] == 1.
    assert result['region_force_macro_mae_estimated_n'] == 0.
    assert result['contact_vertex_force_mae_estimated_n'] == 0.
