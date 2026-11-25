import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from baselines.cached import DIMS
from baselines.data import BaselineDataset
from baselines.models import build
from baselines.run import learned_state, loss
from conftest import synthetic_mano_buffers
from test_prepare_targets import add_pressure2, make_raw_trial
from scripts.prepare_targets import prepare_trial


@pytest.mark.parametrize('method', ['opentouch', 'egotouch', 'egotactile', 'egopressure', 'mot', 'hbridge'])
def test_cached_baselines_forward_backward_and_no_label_inputs(method):
    torch.set_num_threads(2)
    model = build(method, seed=0)
    batch = dict(visual=torch.randn(2, 34, 768), emg=torch.randn(2, 18, 400),
                 input_valid=torch.ones(2, dtype=torch.bool), is_left=torch.tensor([False, True]))
    prediction = model(batch)
    assert set(prediction) == set(DIMS)
    for name, width in DIMS.items():
        expected = (2, width // 3, 3) if name in ('mesh', 'joints') else (2, width)
        assert prediction[name].shape == expected
        assert torch.isfinite(prediction[name]).all()
    sum(value.square().mean() for value in prediction.values()).add(model.extra_loss(batch)).backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        first = model(batch)
        second = model(dict(batch, joints=torch.randn(2, 21, 3) * 100, regional_force=torch.ones(2, 7) * 999))
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def make_prepared(root):
    source = root / 'source'
    members = [make_raw_trial(source, side) for side in ('L', 'R')]
    data = root / 'prepared'
    (data / 'targets').mkdir(parents=True)
    (data / 'features/dino').mkdir(parents=True)
    for member in members:
        add_pressure2(source, member)
        values, atlas, _, _ = prepare_trial(source, member)
        values['emg_2000'] = np.ones((3, 18, 2000), np.float32)
        values['input_valid_2000'] = np.array([False, True, True])
        np.savez(data / 'targets' / f'{member["trial_id"]}.npz', **values)
        np.save(data / 'features/dino' / f'{member["trial_id"]}.npy', np.zeros((3, 34, 768), np.float16))
    np.savez(data / 'anatomical_partition.npz', **atlas)
    (data / 'membership.json').write_text(json.dumps(dict(trials=[dict(member, frames=3) for member in members])))
    (data / 'preparation.json').write_text(json.dumps(dict(complete=True, pressure_version='pressure2')))
    (data / 'splits.json').write_text(json.dumps(dict(protocols=dict(
        cross_trial=dict(train=[members[0]['trial_id']], test=[members[1]['trial_id']])))))
    return data


def test_long_window_baseline_masks_history_without_fake_padding(tmp_path):
    data = make_prepared(tmp_path)
    dataset = BaselineDataset(data, ['trial-R'], 'neuropose')
    first = dataset[0]
    assert first['emg'].shape == (18, 2000)
    assert not first['mesh_valid'] and not first['contact_valid'].any()
    assert not first['force_valid'].any() and not first['surface_valid']
    assert dataset[2]['force_valid'].all()


def test_masked_baseline_loss_ignores_invalid_labels():
    prediction = dict(joints=torch.zeros(1, 21, 3), mesh=torch.zeros(1, 778, 3),
                      regional_contact=torch.full((1, 7), .5), regional_force=torch.zeros(1, 7),
                      vertex_contact=torch.full((1, 778), .5), vertex_force=torch.zeros(1, 778))
    target = dict(prediction, joint_valid=torch.zeros(1, 21, dtype=torch.bool),
                  mesh_valid=torch.zeros(1, dtype=torch.bool), surface_valid=torch.zeros(1, dtype=torch.bool),
                  force_valid=torch.zeros(1, 7, dtype=torch.bool), contact_valid=torch.zeros(1, 7, dtype=torch.bool))
    assert loss(prediction, target, torch.arange(778) % 7) == 0


def test_fixed_epoch_runner_train_evaluate_and_provenance(tmp_path):
    data = make_prepared(tmp_path)
    mano = tmp_path / 'synthetic-mano.npz'
    np.savez(mano, **{name: value.numpy() for name, value in synthetic_mano_buffers().items()})
    common = ['--method', 'egopressure', '--data', str(data), '--splits', str(data / 'splits.json'),
              '--protocol', 'cross_trial', '--partition', str(data / 'anatomical_partition.npz'),
              '--seed', '0', '--epochs', '1', '--batch', '2', '--workers', '0']
    run = tmp_path / 'run'
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, '-m', 'baselines.run', 'train', *common, '--out', str(run)], cwd=root, check=True)
    record = json.loads((run / 'training.json').read_text())
    assert record['test_accessed'] is False and record['history'][0]['frames'] == 3
    command = [sys.executable, '-m', 'baselines.run', 'evaluate', *common,
               '--mano', str(mano), '--checkpoint', str(run / 'final.pt')]
    output = tmp_path / 'test.json'
    subprocess.run([*command, '--out', str(output)], cwd=root, check=True)
    result = json.loads(output.read_text())
    assert result['frames'] == 3 and not result['selection_by_test']
    (data / 'preparation.json').write_text(json.dumps(dict(complete=True, pressure_version='changed')))
    failed = subprocess.run([*command, '--out', str(tmp_path / 'mismatch.json')], cwd=root, capture_output=True)
    assert failed.returncode != 0 and b'provenance mismatch' in failed.stderr


@pytest.mark.parametrize('method', ['neuropose', 'emgformer', 'emg2pose'])
def test_pinned_emg_components_forward_backward(method):
    components = os.environ.get('EGOMYO_COMPONENTS')
    if not components:
        pytest.skip('Set EGOMYO_COMPONENTS to separately acquired pinned external source files')
    torch.set_num_threads(2)
    model = build(method, dict(components=components), seed=0)
    batch = dict(emg=torch.randn(2, 18, 2000), visual=torch.randn(2, 34, 768),
                 input_valid=torch.ones(2, dtype=torch.bool), is_left=torch.zeros(2, dtype=torch.bool))
    prediction = model(batch)
    assert set(prediction) == set(DIMS)
    assert all(torch.isfinite(value).all() for value in prediction.values())
    sum(value.square().mean() for value in prediction.values()).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    with pytest.raises(ValueError, match='18x2000'):
        model(dict(batch, emg=torch.zeros(2, 18, 400)))


@pytest.mark.parametrize('method', ['hamer', 'wilor'])
def test_external_rgb_interface_with_synthetic_native_model(method, monkeypatch, tmp_path):
    class SyntheticNative(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mano = torch.nn.Linear(1, 1)
            readout = torch.nn.Linear(3, 63)
            if method == 'wilor':
                self.refine_net = torch.nn.ModuleDict(dict(dec_pose=readout))
            else:
                self.mano_head = torch.nn.ModuleDict(dict(decpose=readout))

        def forward(self, batch):
            feature = batch['img'].mean((2, 3))
            readout = self.refine_net['dec_pose'] if method == 'wilor' else self.mano_head['decpose']
            joints = readout(feature).reshape(-1, 21, 3)
            return dict(pred_keypoints_3d=joints, pred_vertices=joints[:, :1].expand(-1, 778, -1))

    config = SimpleNamespace(MODEL=SimpleNamespace(
        IMAGE_MEAN=[.485, .456, .406], IMAGE_STD=[.229, .224, .225], IMAGE_SIZE=256))
    package = ModuleType(method)
    module = ModuleType(method + '.models')
    setattr(module, 'load_' + method, lambda *args: (SyntheticNative(), config))
    monkeypatch.setitem(sys.modules, method, package)
    monkeypatch.setitem(sys.modules, method + '.models', module)
    model = build(method, dict(source_root=str(tmp_path), checkpoint='synthetic', config='synthetic'))
    batch = dict(img=torch.rand(2, 3, 256, 256), right=torch.ones(2), box_center=torch.zeros(2, 2),
                 box_size=torch.ones(2), img_size=torch.ones(2, 2), is_left=torch.zeros(2, dtype=torch.bool))
    prediction = model(batch)
    assert set(prediction) == set(DIMS)
    assert prediction['mesh'].shape == (2, 778, 3)
    assert not any(name.startswith('native.native.mano.') for name in learned_state(model))
