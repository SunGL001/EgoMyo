import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.prepare_targets import prepare_trial
from scripts.raw_target_contract import (REGIONS, causal_emg_windows, explicit_contact_masks,
                                         nearest_indices, sparse_vertex_force)


def test_causal_windows_never_read_future_and_mask_short_history():
    times = np.arange(500, dtype=np.int64) * 500_000
    adc = np.repeat(np.arange(500)[:, None], 18, axis=1)
    windows, valid = causal_emg_windows(adc, times, np.arange(500), np.array([times[399], times[10]]))
    assert valid.tolist() == [True, False]
    np.testing.assert_array_equal(windows[0, 0], np.arange(400))
    assert windows[1, 0, -1] == 10
    sequence = np.arange(500)
    sequence[200:] += 1
    _, valid = causal_emg_windows(adc, times, sequence, times[[399]])
    assert not valid[0]


def test_nearest_tie_prefers_earlier_and_rejects_stale():
    indices, close = nearest_indices([0, 100], [50, 1000], 60)
    assert indices.tolist() == [0, 1]
    assert close.tolist() == [True, False]
    with pytest.raises(ValueError, match='strictly increasing'):
        nearest_indices([1, 1], [1], 10)


def test_sparse_lift_preserves_force_and_masks_unsupported_region():
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    partition = np.array([0, 0, 1, 1])
    force, regional, valid = sparse_vertex_force(
        faces, partition, [0, 1], [0, 6], [[.2, .3, .5], [.2, .3, .5]], [5, 7])
    assert force.sum() == pytest.approx(5)
    assert regional[0] == pytest.approx(5)
    assert not valid[6]


def test_unknown_contacts_are_not_valid_negatives():
    contact = np.array([[1, -1, 0, 0, 0, 0, 0]])
    valid, support, surface = explicit_contact_masks(
        contact, np.array([1]), np.array([True]), np.array([0, 1]),
        np.array([[1., 1.]]), np.array([True]))
    assert not valid[0, 1]
    assert support.tolist() == [[True, False]]
    assert not surface[0]


def make_raw_trial(root, side='R'):
    package = 'synthetic-' + side
    trial_id = 'trial-' + side
    relative = package + '/dataset/trial'
    trial = root / relative
    for folder in ('ego', 'emg', 'contact', 'force'):
        (trial / folder).mkdir(parents=True)
    metadata = dict(trial_id=trial_id, source_run_id='run', subject_id=package,
                    session_number=1, action_index=1, repetition=1)
    (trial / 'trial_info.json').write_text(json.dumps(metadata))
    (root / package / 'DATASET.json').write_text(json.dumps(dict(
        formal_accepted=False, cache_release_allowed=False, product_version='synthetic')))
    stamps = np.array([250_000_000, 280_000_000, 310_000_000], np.int64)
    (trial / 'ego/ego_timestamps.csv').write_text(
        'output_frame_index,host_receive_timestamp_ns,trial_id,source_run_id\n' +
        ''.join(f'{index},{time},{trial_id},run\n' for index, time in enumerate(stamps)))
    with h5py.File(trial / 'emg/emg_raw.h5', 'w') as handle:
        handle['adc'] = np.ones((700, 18), np.int32)
        handle['host_monotonic_ns'] = np.arange(700, dtype=np.int64) * 500_000
        handle['sample_index'] = np.arange(700)
    regions = np.arange(1538) % 7
    faces = regions[:, None] * 3 + np.arange(3)[None]
    if side == 'L':
        faces = faces[:, [0, 2, 1]]
    np.savez(trial / 'contact/mano.npz', timestamp_ns=stamps, faces=faces,
             trial_id=np.full(3, trial_id), source_run_id=np.full(3, 'run'),
             vertices_ego_m=np.ones((3, 778, 3), np.float32),
             joints_3d_ego_m=np.ones((3, 21, 3), np.float32),
             mesh_valid=np.ones(3, bool), original_valid=np.ones((3, 21), bool),
             ego_coordinate_status=np.full(3, 'frozen_calibration_unvalidated'))
    force = np.zeros((3, 7))
    force[:, 0] = 2
    states = np.array([1, -1, 1])
    quality = np.array([0, 0, 4])
    with h5py.File(trial / 'force/pressure_postprocessed.h5', 'w') as handle:
        handle['host_monotonic_ns'] = stamps
        handle['sample_index'] = np.arange(3)
        handle['quality_flags'] = quality
        handle['contact_state'] = states
        handle['force_7_estimated_n'] = force
        handle['unassigned_estimated_n'] = np.full(3, 999.)
    columns = ['source_run_id', 'host_monotonic_ns', 'sample_index', 'contact_state', 'quality_flags']
    columns += [region + '_estimated_n' for region in REGIONS] + [region + '_contact' for region in REGIONS]
    contents = ','.join(columns) + '\n'
    for index, stamp in enumerate(stamps):
        values = ['run', stamp, index, states[index], quality[index], *force[index], 1, 0, 0, 0, 0, 0, 0]
        contents += ','.join(map(str, values)) + '\n'
    csv_path = trial / 'force/region_force_postprocessed.csv'
    csv_path.write_text(contents)
    report = dict(trial_id=trial_id, source_run_id='run', outputs={
        'force/region_force_postprocessed.csv': dict(bytes=csv_path.stat().st_size,
                                                   sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest())})
    (trial / 'force/pressure_postprocess.json').write_text(json.dumps(report))
    with h5py.File(trial / 'contact/contact_postprocessed.h5', 'w') as handle:
        handle.attrs['atlas_sha256'] = 'synthetic-atlas'
        for key, value in dict(face_region=regions, host_monotonic_ns=stamps, contact_state=states,
                               mapping_valid=np.ones(3, bool), sparse_frame_offsets=np.arange(4),
                               sparse_face_index=np.zeros(3, int), sparse_channel=np.zeros(3, int),
                               sparse_barycentric=np.full((3, 3), 1 / 3),
                               sparse_force_matrix_estimated_n=np.full(3, 2.)).items():
            handle[key] = value
    return dict(trial_id=trial_id, source_run_id='run', package=package, side=side,
                relative_path=relative, session=1, action=1, repetition=1, participant='synthetic')


def test_prepare_real_layout_preserves_v5_semantics(tmp_path):
    member = make_raw_trial(tmp_path)
    values, atlas, receipts, indices = prepare_trial(tmp_path, member, pressure_version='pressure1')
    assert values['emg'].shape == (3, 18, 400)
    assert values['input_valid'].all()
    assert values['force_valid'][:, 0].tolist() == [True, True, False]
    assert values['contact_valid'][:, 0].tolist() == [True, False, False]
    assert values['surface_valid'].tolist() == [True, False, False]
    assert values['timing_unknown'].all()
    assert values['vertex_force'][0].sum() == pytest.approx(2)
    assert indices == [0, 1, 2]
    assert 'emg/emg_raw.h5' in receipts
    left = make_raw_trial(tmp_path, 'L')
    left_values, left_atlas, _, _ = prepare_trial(tmp_path, left, atlas, pressure_version='pressure1')
    assert left_values['is_left'].all()
    np.testing.assert_array_equal(atlas['partition'], left_atlas['partition'])


def test_rejects_bad_input_identity_and_unsafe_paths(tmp_path):
    member = make_raw_trial(tmp_path)
    with pytest.raises(ValueError, match='Unsafe relative path'):
        prepare_trial(tmp_path, dict(member, relative_path='../escape'))
    member['inputs'] = {'emg/emg_raw.h5': dict(bytes=1, sha256='invalid')}
    with pytest.raises(ValueError, match='Frozen membership input changed'):
        prepare_trial(tmp_path, member)


def test_cli_writes_model_ready_targets_and_frame_map(tmp_path):
    source = tmp_path / 'source'
    member = make_raw_trial(source)
    manifest = tmp_path / 'membership.json'
    manifest.write_text(json.dumps(dict(trials=[member])))
    output = tmp_path / 'prepared'
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts/prepare_targets.py'),
               '--source-root', str(source), '--membership', str(manifest), '--out', str(output)]
    command += ['--pressure-version', 'pressure1']
    subprocess.run(command, check=True)
    assert (output / 'targets/trial-R.npz').is_file()
    assert json.loads((output / 'frame-map.json').read_text())['trials']['trial-R']['frame_indices'] == [0, 1, 2]
    record = json.loads((output / 'preparation.json').read_text())
    assert record['source_package_status'][0]['cache_release_allowed'] is False
    assert subprocess.run(command, capture_output=True).returncode != 0


def add_pressure2(root, member):
    trial = root / member['relative_path']
    stamps = np.array([250_000_000, 265_000_000, 280_000_000, 310_000_000])
    force = np.zeros((4, 7))
    force[:, 0] = [2., 3., 4., 5.]
    quality = np.array([0, 4 | 1, 8, 128])
    with h5py.File(trial / 'force/pressure_completed.h5', 'w') as handle:
        handle.attrs['version'] = 'pressure2'
        handle.attrs['channel_order_json'] = json.dumps(REGIONS)
        for key, value in dict(host_monotonic_ns=stamps, force_7_estimated_n=force,
                               quality_flags=quality, source_quality_flags=np.zeros(4, int),
                               source_row_index=np.array([0, -1, 1, 2])).items():
            handle[key] = value
    columns = ['host_monotonic_ns', 'quality_flags'] + [name + '_estimated_n' for name in REGIONS]
    with (trial / 'force/region_force_completed.csv').open('w', newline='') as stream:
        writer = __import__('csv').writer(stream)
        writer.writerow(columns)
        writer.writerows([[stamp, flag, *values] for stamp, flag, values in zip(stamps, quality, force)])
    (trial / 'force/pressure_completion.json').write_text(json.dumps(dict(
        version='pressure2', trial_id=member['trial_id'], source_run_id=member['source_run_id'])))
    with h5py.File(trial / 'contact/contact_surface_completed.h5', 'w') as handle:
        handle.attrs['pressure_version'] = 'pressure2'
        dense = np.zeros((4, 1538))
        dense[:, 0] = force[:, 0]
        for key, value in dict(host_monotonic_ns=stamps, pressure_quality_flags=quality,
                               face_region=np.arange(1538) % 7, face_force_matrix_estimated_n=dense,
                               spatial_mapping_method=np.array([1, 2, 0, 1])).items():
            handle[key] = value


def test_pressure2_default_uses_completed_axis_and_own_flags(tmp_path):
    member = make_raw_trial(tmp_path)
    add_pressure2(tmp_path, member)
    values, _, _, _ = prepare_trial(tmp_path, member)
    assert values['regional_force'][:, 0].tolist() == [2., 4., 5.]
    assert values['force_valid'][:, 0].tolist() == [True, False, True]
    assert values['surface_valid'].tolist() == [True, False, True]
    assert not values['contact_valid'][1].any()
    assert values['vertex_force'][2].sum() == pytest.approx(5)
    np.testing.assert_allclose(values['surface_regional_force'][:, 0], [2, 0, 5])


def test_pressure2_inserted_force_is_usable_and_bad_surface_axis_rejected(tmp_path):
    member = make_raw_trial(tmp_path)
    add_pressure2(tmp_path, member)
    trial = tmp_path / member['relative_path']
    path = trial / 'ego/ego_timestamps.csv'
    path.write_text(path.read_text().replace('280000000', '265000000'))
    values, _, _, _ = prepare_trial(tmp_path, member)
    assert values['pressure_source_row'][1] == -1
    assert values['force_valid'][1].all()
    assert values['regional_force'][1, 0] == 3
    assert values['vertex_force'][1].sum() == pytest.approx(3)
    with h5py.File(trial / 'contact/contact_surface_completed.h5', 'r+') as handle:
        handle['host_monotonic_ns'][1] += 1
    with pytest.raises(ValueError, match='surface axis'):
        prepare_trial(tmp_path, member)
