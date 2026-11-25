#!/usr/bin/env python3
"""Convert delivered EgoMyo trials to model-ready targets (pressure2 by default).

Requires completed pressure2 and MANO/surface annotations, not just sensor-only
ADC/matrices. Source files are read-only; output must be a new directory.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_target_contract import (PRESSURE_REJECT_BITS, REGIONS, causal_emg_windows,
                                 explicit_contact_masks, mesh_masks, nearest_indices,
                                 sparse_vertex_force, vertex_partition)

REQUIRED = ('trial_info.json', 'ego/ego_timestamps.csv', 'emg/emg_raw.h5',
            'contact/mano.npz')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def safe_relative(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes source root: {relative}')
    return result


def read_region_csv(path, source_run_id):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'source_run_id', 'host_monotonic_ns', 'sample_index', 'contact_state', 'quality_flags'}
        required.update(region + suffix for region in REGIONS for suffix in ('_contact', '_estimated_n'))
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing explicit regional contact/force columns')
        rows = list(reader)
    if not rows or any(row['source_run_id'] != source_run_id for row in rows):
        raise ValueError('Empty CSV or source run mismatch')
    contact = np.array([[int(row[region + '_contact']) for region in REGIONS] for row in rows], np.int8)
    state = np.array([int(row['contact_state']) for row in rows], np.int8)
    if not np.isin(contact, [-1, 0, 1]).all() or not np.isin(state, [-1, 0, 1]).all():
        raise ValueError('Illegal contact state; unknown -1 must be masked, not coerced')
    return dict(timestamp_ns=np.array([int(row['host_monotonic_ns']) for row in rows], np.int64),
                sample_index=np.array([int(row['sample_index']) for row in rows], np.int64),
                quality_flags=np.array([int(row['quality_flags']) for row in rows], np.int64),
                regional_force=np.array([[float(row[region + '_estimated_n']) for region in REGIONS] for row in rows]),
                regional_contact=contact, contact_state=state)


def completed_pressure(trial, member, times, input_valid, mesh_valid, faces, atlas, threshold):
    report = read_json(trial / 'force/pressure_completion.json')
    if (report.get('version') != 'pressure2' or report.get('trial_id') != member['trial_id'] or
            report.get('source_run_id') != member['source_run_id']):
        raise ValueError('Pressure2 report identity/version mismatch')
    with (trial / 'force/region_force_completed.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    csv_times = np.array([int(row['host_monotonic_ns']) for row in rows], np.int64)
    csv_force = np.array([[float(row[name + '_estimated_n']) for name in REGIONS] for row in rows])
    with h5py.File(trial / 'force/pressure_completed.h5', 'r') as pressure:
        if pressure.attrs.get('version') != 'pressure2':
            raise ValueError('Expected pressure2 H5 version')
        pressure_times = pressure['host_monotonic_ns'][:]
        all_force = pressure['force_7_estimated_n'][:]
        all_quality = pressure['quality_flags'][:]
        if (not np.array_equal(csv_times, pressure_times) or
                not np.allclose(csv_force, all_force, rtol=1e-6, atol=1e-7, equal_nan=True) or
                not np.array_equal([int(row['quality_flags']) for row in rows], all_quality)):
            raise ValueError('Pressure2 CSV/H5 mismatch')
        if list(json.loads(pressure.attrs['channel_order_json'])) != list(REGIONS):
            raise ValueError('Pressure2 region order mismatch')
        index, close = nearest_indices(pressure_times, times, 40_000_000)
        force = all_force[index].astype(np.float32)
        quality = all_quality[index].astype(np.int64)
        source_quality = pressure['source_quality_flags'][:][index]
        source_row = pressure['source_row_index'][:][index]
    good = close & input_valid & ((quality & (8 | 64)) == 0)
    force_valid = good[:, None] & np.isfinite(force) & (force >= 0)
    contact_valid = force_valid.copy()
    contact_labels = np.where(contact_valid, (force > threshold).astype(np.int8), -1)
    state = np.where(contact_valid.all(1), (contact_labels == 1).any(1).astype(np.int8), -1)
    with h5py.File(trial / 'contact/contact_surface_completed.h5', 'r') as surface:
        original_regions = surface['face_region'][:]
        atlas_hash = surface.attrs.get('atlas_sha256')
        original_path = trial / 'contact/contact_postprocessed.h5'
        if original_path.is_file():
            with h5py.File(original_path, 'r') as original:
                if not np.array_equal(original['face_region'][:], original_regions):
                    raise ValueError('Pressure2 surface atlas mismatch')
                if atlas_hash is not None and str(atlas_hash) != str(original.attrs['atlas_sha256']):
                    raise ValueError('Pressure2 surface atlas hash mismatch')
                atlas_hash = original.attrs['atlas_sha256']
        if atlas_hash is None:
            raise ValueError('Pressure2 surface is missing its anatomical atlas hash')
        atlas_hash = str(atlas_hash)
        if (surface.attrs.get('pressure_version') != 'pressure2' or
                not np.array_equal(surface['host_monotonic_ns'][:], pressure_times) or
                not np.array_equal(surface['pressure_quality_flags'][:], all_quality)):
            raise ValueError('Pressure2 surface axis/quality/atlas mismatch')
        canonical_faces = faces if member['side'] == 'R' else faces[:, [0, 2, 1]]
        partition = vertex_partition(canonical_faces, original_regions)
        current_atlas = dict(faces=canonical_faces, face_region=original_regions,
                             partition=partition, atlas_sha256=np.array(atlas_hash))
        if atlas is not None and any(not np.array_equal(atlas[key], value) for key, value in current_atlas.items()):
            raise ValueError('Cross-trial anatomical atlas/topology changed')
        dense = surface['face_force_matrix_estimated_n'][:]
        if dense.shape != (len(pressure_times), len(faces)):
            raise ValueError('Pressure2 per-face force layout mismatch')
        mapping = surface['spatial_mapping_method'][:]
    vertex_force = np.zeros((len(times), 778), np.float32)
    regional = np.zeros((len(times), 7), np.float32)
    distribution_valid = np.zeros((len(times), 7), bool)
    base_valid = np.zeros(len(times), bool)
    for frame, row in enumerate(index):
        if not (good[frame] and mesh_valid[frame] and mapping[row] in (1, 2)):
            continue
        face_force = dense[row]
        assigned = (original_regions >= 0) & (original_regions < 7)
        if not np.isfinite(face_force).all() or (face_force < 0).any():
            continue
        selected = np.flatnonzero(assigned & (face_force > 0))
        values, sums, valid = sparse_vertex_force(
            faces, partition, selected, original_regions[selected],
            np.full((len(selected), 3), 1 / 3), face_force[selected])
        vertex_force[frame], regional[frame], distribution_valid[frame] = values, sums, valid
        base_valid[frame] = bool(valid.all())
    _, vertex_contact, surface_valid = explicit_contact_masks(
        contact_labels, state, good, partition, vertex_force, base_valid)
    return dict(regional_force=np.nan_to_num(force), force_valid=force_valid,
                regional_contact=contact_labels == 1, contact_valid=contact_valid,
                global_contact_state=state, vertex_force=vertex_force, vertex_contact=vertex_contact,
                surface_regional_force=regional, surface_valid=surface_valid,
                distribution_valid=distribution_valid & surface_valid[:, None],
                pressure_quality=quality, pressure_source_quality=source_quality,
                pressure_source_row=source_row), current_atlas


def prepare_trial(source, member, atlas=None, pressure_version='pressure2', contact_threshold=0.2):
    trial = safe_relative(source, member['relative_path'])
    receipts = {}
    required = REQUIRED
    if pressure_version == 'pressure2':
        required += ('force/pressure_completed.h5', 'force/region_force_completed.csv',
                     'force/pressure_completion.json', 'contact/contact_surface_completed.h5')
        if (trial / 'contact/contact_postprocessed.h5').is_file():
            required += ('contact/contact_postprocessed.h5',)
    elif pressure_version == 'pressure1':
        required += ('force/pressure_postprocessed.h5', 'force/region_force_postprocessed.csv',
                     'force/pressure_postprocess.json', 'contact/contact_postprocessed.h5')
    elif pressure_version != 'pressure1':
        raise ValueError('Unknown pressure version')
    if not np.isfinite(contact_threshold) or contact_threshold < 0:
        raise ValueError('Contact threshold must be finite and nonnegative')
    for name in required:
        path = trial / name
        receipt = dict(bytes=path.stat().st_size, sha256=sha256(path))
        expected = member.get('inputs', {}).get(name)
        if expected is not None and any(receipt[key] != expected[key] for key in ('bytes', 'sha256')):
            raise ValueError(f'Frozen membership input changed: {path}')
        receipts[name] = receipt
    info = read_json(trial / 'trial_info.json')
    if info['trial_id'] != member['trial_id'] or info['source_run_id'] != member['source_run_id']:
        raise ValueError('Trial/run identity mismatch')
    if member['side'] not in ('L', 'R') or info['subject_id'] != member['package']:
        raise ValueError('Package identity or hand side mismatch')
    if not member['package'].upper().endswith('-' + member['side']):
        raise ValueError('Hand side disagrees with package metadata')
    if (info['session_number'] != member['session'] or info['action_index'] != member['action'] or
            info['repetition'] != member['repetition']):
        raise ValueError('Session/action/repetition metadata mismatch')
    with (trial / 'ego/ego_timestamps.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row['trial_id'] != member['trial_id'] or
                       row['source_run_id'] != member['source_run_id'] for row in rows):
        raise ValueError('Video timestamp identity mismatch')
    frame_indices = [int(row['output_frame_index']) for row in rows]
    if frame_indices != list(range(len(rows))):
        raise ValueError('Video output frame indices must be consecutive from zero')
    times = np.array([int(row['host_receive_timestamp_ns']) for row in rows], np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError('Video timestamps must strictly increase')
    if 'frames' in member and member['frames'] != len(times):
        raise ValueError('Membership/video frame count mismatch')
    with h5py.File(trial / 'emg/emg_raw.h5', 'r') as handle:
        emg, input_valid = causal_emg_windows(handle['adc'][:], handle['host_monotonic_ns'][:],
                                             handle['sample_index'][:], times)
    with np.load(trial / 'contact/mano.npz', allow_pickle=False) as geometry:
        if (np.any(geometry['trial_id'] != member['trial_id']) or
                np.any(geometry['source_run_id'] != member['source_run_id'])):
            raise ValueError('MANO trial/run identity mismatch')
        faces = geometry['faces'].astype(np.int64)
        pose_index, pose_close = nearest_indices(geometry['timestamp_ns'], times, 50_000_000)
        masks = mesh_masks(geometry['vertices_ego_m'], geometry['joints_3d_ego_m'],
                           geometry['mesh_valid'], geometry['original_valid'],
                           rejected=geometry['source_timing_rejected'] if 'source_timing_rejected' in geometry else None,
                           coordinate_status=geometry['ego_coordinate_status'])
        mesh = geometry['vertices_ego_m'][pose_index] - geometry['joints_3d_ego_m'][pose_index, :1]
        joints = geometry['joints_3d_ego_m'][pose_index] - geometry['joints_3d_ego_m'][pose_index, :1]
        mesh_valid = masks['mesh_valid'][pose_index] & pose_close & input_valid
        joint_valid = masks['joint_valid'][pose_index] & pose_close[:, None] & input_valid[:, None]
        timing_unknown = masks['timing_unknown'][pose_index]
    if pressure_version == 'pressure2':
        pressure, current_atlas = completed_pressure(
            trial, member, times, input_valid, mesh_valid, faces, atlas, contact_threshold)
        values = dict(timestamp_ns=times, emg=emg, input_valid=input_valid,
                      mesh=np.nan_to_num(mesh).astype(np.float32), joints=np.nan_to_num(joints).astype(np.float32),
                      mesh_valid=mesh_valid, joint_valid=joint_valid, timing_unknown=timing_unknown,
                      is_left=np.full(len(times), member['side'] == 'L', bool), **pressure)
        return values, current_atlas, receipts, frame_indices
    labels = read_region_csv(trial / 'force/region_force_postprocessed.csv', member['source_run_id'])
    report = read_json(trial / 'force/pressure_postprocess.json')
    expected_csv = report['outputs']['force/region_force_postprocessed.csv']
    if (report['trial_id'] != member['trial_id'] or report['source_run_id'] != member['source_run_id'] or
            any(expected_csv[key] != receipts['force/region_force_postprocessed.csv'][key]
                for key in ('bytes', 'sha256'))):
        raise ValueError('Pressure report/CSV identity mismatch')
    with h5py.File(trial / 'force/pressure_postprocessed.h5', 'r') as pressure:
        pressure_times = pressure['host_monotonic_ns'][:]
        for key, field in (('timestamp_ns', 'host_monotonic_ns'), ('sample_index', 'sample_index'),
                           ('quality_flags', 'quality_flags'), ('contact_state', 'contact_state')):
            if not np.array_equal(labels[key], pressure[field][:]):
                raise ValueError(f'CSV/pressure H5 mismatch: {field}')
        if not np.allclose(labels['regional_force'], pressure['force_7_estimated_n'][:],
                           rtol=1e-6, atol=1e-7, equal_nan=True):
            raise ValueError('CSV/pressure force mismatch')
        pressure_index, pressure_close = nearest_indices(pressure_times, times, 40_000_000)
        force = pressure['force_7_estimated_n'][:][pressure_index].astype(np.float32)
        quality = pressure['quality_flags'][:][pressure_index].astype(np.int64)
    frame_good = pressure_close & input_valid & ((quality & sum(PRESSURE_REJECT_BITS)) == 0)
    force_valid = frame_good[:, None] & np.isfinite(force) & (force >= 0)
    vertex_force = np.zeros((len(times), 778), np.float32)
    surface_regional = np.zeros((len(times), 7), np.float32)
    distribution_valid = np.zeros((len(times), 7), bool)
    base_surface_valid = np.zeros(len(times), bool)
    with h5py.File(trial / 'contact/contact_postprocessed.h5', 'r') as contact:
        face_region = contact['face_region'][:]
        atlas_hash = str(contact.attrs['atlas_sha256'])
        canonical_faces = faces if member['side'] == 'R' else faces[:, [0, 2, 1]]
        partition = vertex_partition(canonical_faces, face_region)
        current_atlas = dict(faces=canonical_faces, face_region=face_region,
                             partition=partition, atlas_sha256=np.array(atlas_hash))
        if atlas is not None and any(not np.array_equal(atlas[key], value) for key, value in current_atlas.items()):
            raise ValueError('Cross-trial anatomical atlas/topology changed')
        if (not np.array_equal(contact['host_monotonic_ns'][:], pressure_times) or
                not np.array_equal(contact['contact_state'][:], labels['contact_state'])):
            raise ValueError('Contact/pressure time axis or state mismatch')
        mapping_valid = contact['mapping_valid'][:]
        offsets = contact['sparse_frame_offsets'][:]
        sparse_faces = contact['sparse_face_index'][:]
        sparse_channel = contact['sparse_channel'][:]
        sparse_barycentric = contact['sparse_barycentric'][:]
        sparse_force = contact['sparse_force_matrix_estimated_n'][:]
        if (offsets.shape != (len(pressure_times) + 1,) or offsets[0] != 0 or
                np.any(np.diff(offsets) < 0) or offsets[-1] != len(sparse_force)):
            raise ValueError('Invalid sparse frame offsets')
        for frame, index in enumerate(pressure_index):
            if not (frame_good[frame] and mapping_valid[index] and mesh_valid[frame]):
                continue
            begin, end = int(offsets[index]), int(offsets[index + 1])
            values, regional, valid = sparse_vertex_force(
                faces, partition, sparse_faces[begin:end], sparse_channel[begin:end],
                sparse_barycentric[begin:end], sparse_force[begin:end])
            vertex_force[frame], surface_regional[frame], distribution_valid[frame] = values, regional, valid
            base_surface_valid[frame] = bool(valid.all())
    contact_labels = labels['regional_contact'][pressure_index]
    state = labels['contact_state'][pressure_index]
    contact_valid, vertex_contact, surface_valid = explicit_contact_masks(
        contact_labels, state, frame_good, partition, vertex_force, base_surface_valid)
    values = dict(timestamp_ns=times, emg=emg, input_valid=input_valid,
                  mesh=np.nan_to_num(mesh).astype(np.float32), joints=np.nan_to_num(joints).astype(np.float32),
                  mesh_valid=mesh_valid, joint_valid=joint_valid, timing_unknown=timing_unknown,
                  regional_force=np.nan_to_num(labels['regional_force'][pressure_index]).astype(np.float32),
                  force_valid=force_valid, regional_contact=contact_labels == 1, contact_valid=contact_valid,
                  global_contact_state=state, vertex_force=vertex_force, vertex_contact=vertex_contact,
                  surface_regional_force=surface_regional, surface_valid=surface_valid,
                  distribution_valid=distribution_valid & surface_valid[:, None], pressure_quality=quality,
                  is_left=np.full(len(times), member['side'] == 'L', bool))
    return values, current_atlas, receipts, frame_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True,
                        help='Read-only root containing delivered package directories')
    parser.add_argument('--membership', type=Path, required=True,
                        help='Frozen trial membership with relative_path, identities and split metadata')
    parser.add_argument('--out', type=Path, required=True, help='New model-ready output directory')
    parser.add_argument('--pressure-version', choices=('pressure2', 'pressure1'), default='pressure2')
    parser.add_argument('--contact-threshold-n', type=float, default=0.2,
                        help='Pressure2 derived contact: unsmoothed completed regional force > threshold')
    parser.add_argument('--include-long-emg', action='store_true',
                        help='Also export causal 2000-sample raw windows for EMG baselines')
    args = parser.parse_args()
    membership = read_json(args.membership)
    trials = membership['trials']
    identifiers = [member['trial_id'] for member in trials]
    if not trials or len(set(identifiers)) != len(identifiers):
        raise ValueError('Membership must contain nonempty, unique trial IDs')
    for identifier in identifiers:
        if not isinstance(identifier, str) or not identifier or '/' in identifier or '\\' in identifier or identifier in ('.', '..'):
            raise ValueError('Unsafe trial ID')
    if args.out.exists():
        raise FileExistsError('Use a new output directory; existing caches are never overwritten')
    args.out.mkdir(parents=True)
    (args.out / 'targets').mkdir()
    (args.out / 'provenance').mkdir()
    atlas = None
    exported = []
    frame_map = {}
    package_status = {}
    for member in trials:
        if member['package'] not in package_status:
            dataset_path = safe_relative(args.source_root, member['package']) / 'DATASET.json'
            dataset = read_json(dataset_path)
            package_status[member['package']] = dict(
                package=member['package'], dataset_sha256=sha256(dataset_path),
                formal_accepted=dataset.get('formal_accepted'),
                cache_release_allowed=dataset.get('cache_release_allowed'),
                product_version=dataset.get('product_version'),
            )
        values, atlas, receipts, indices = prepare_trial(
            args.source_root, member, atlas, args.pressure_version, args.contact_threshold_n)
        if args.include_long_emg:
            with h5py.File(safe_relative(args.source_root, member['relative_path']) / 'emg/emg_raw.h5', 'r') as raw:
                values['emg_2000'], values['input_valid_2000'] = causal_emg_windows(
                    raw['adc'][:], raw['host_monotonic_ns'][:], raw['sample_index'][:],
                    values['timestamp_ns'], samples=2000, history_ns=1_100_000_000)
        destination = args.out / 'targets' / f'{member["trial_id"]}.npz'
        with destination.open('wb') as stream:
            np.savez_compressed(stream, **values)
        row = dict(member, frames=len(indices))
        exported.append(row)
        video = safe_relative(args.source_root, member['relative_path']) / 'ego/ego.mp4'
        frame_map[member['trial_id']] = dict(video=str(video), side=member['side'], frame_indices=indices)
        write_json(args.out / 'provenance' / f'{member["trial_id"]}.json', dict(
            inputs=receipts, output_sha256=sha256(destination),
            source_capture_usable=member.get('source_capture_usable'),
            source_capture_status=member.get('source_capture_status'),
            coverage={key: int(values[key].sum()) for key in
                      ('input_valid', 'mesh_valid', 'joint_valid', 'force_valid', 'contact_valid', 'surface_valid')},
        ))
        print(member['trial_id'], 'frames', len(indices), 'input_valid', int(values['input_valid'].sum()), flush=True)
    np.savez_compressed(args.out / 'anatomical_partition.npz', **atlas)
    write_json(args.out / 'membership.json', dict(membership, trials=exported))
    write_json(args.out / 'frame-map.json', dict(trials=frame_map))
    write_json(args.out / 'preparation.json', dict(
        schema='egomyo.release.targets.' + args.pressure_version + '-v1', complete=True, trials=len(trials),
        membership_sha256=sha256(args.membership),
        algorithm_sha256=sha256(Path(__file__)),
        contract_sha256=sha256(Path(__file__).with_name('raw_target_contract.py')),
        pressure_version=args.pressure_version,
        pressure_reject_quality_bits=[8, 64] if args.pressure_version == 'pressure2' else PRESSURE_REJECT_BITS,
        contact_policy=dict(kind='regional_force_threshold', threshold_n=args.contact_threshold_n)
            if args.pressure_version == 'pressure2' else dict(kind='explicit_source_states'),
        force_variant='unsmoothed',
        long_emg_exported=args.include_long_emg,
        surface_lift='owner-renormalized equal face corners' if args.pressure_version == 'pressure2'
            else 'source sparse barycentric',
        pose_tolerance_ms=50, pressure_tolerance_ms=40,
        source_root=str(args.source_root.resolve()),
        annotation_status='experimental derived annotations; source acceptance/release flags unchanged',
        physical_sync_verified=False, absolute_force_calibrated=False,
        unassigned_force_screening=False,
        source_package_status=list(package_status.values()),
    ))


if __name__ == '__main__':
    main()
