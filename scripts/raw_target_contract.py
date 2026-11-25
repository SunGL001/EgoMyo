"""Array rules ported from the original EgoMyo target/v5 mask builders.

These rules preserve experimental annotation semantics, not hardware calibration.
"""
import numpy as np


REGIONS = ('T', 'I', 'M', 'R', 'P', 'PALM_UPPER', 'PALM_LOWER')
PRESSURE_REJECT_BITS = (4, 8, 16, 64, 128)


def nearest_indices(source_ns, target_ns, tolerance_ns):
    source_ns = np.asarray(source_ns, dtype=np.int64)
    target_ns = np.asarray(target_ns, dtype=np.int64)
    if source_ns.ndim != 1 or not len(source_ns) or np.any(np.diff(source_ns) <= 0):
        raise ValueError('Annotation timestamps must be nonempty and strictly increasing')
    right = np.searchsorted(source_ns, target_ns).clip(0, len(source_ns) - 1)
    left = (right - 1).clip(0, len(source_ns) - 1)
    index = np.where(np.abs(source_ns[left] - target_ns) <=
                     np.abs(source_ns[right] - target_ns), left, right)
    return index, np.abs(source_ns[index] - target_ns) <= tolerance_ns


def causal_emg_windows(adc, receive_ns, sample_index, observation_ns, samples=400, history_ns=250_000_000):
    """Last acquired samples only, with explicit invalidity for padding or gaps."""
    adc = np.asarray(adc)
    receive_ns = np.asarray(receive_ns, dtype=np.int64)
    sample_index = np.asarray(sample_index, dtype=np.int64)
    observation_ns = np.asarray(observation_ns, dtype=np.int64)
    if adc.shape != (len(receive_ns), 18) or sample_index.shape != receive_ns.shape:
        raise ValueError('EMG layout mismatch')
    if np.any(np.diff(receive_ns) < 0):
        raise ValueError('Decreasing raw receive timestamps; do not silently sort')
    if not len(adc):
        return np.zeros((len(observation_ns), 18, samples), np.float32), np.zeros(len(observation_ns), bool)
    stop = np.searchsorted(receive_ns, observation_ns, side='right')
    indices = stop[:, None] - samples + np.arange(samples)[None]
    available = indices >= 0
    safe = indices.clip(0, len(adc) - 1)
    window = adc[safe].astype(np.float32)
    window[~available] = 0
    times = receive_ns[safe]
    sequence = sample_index[safe]
    valid = available.all(1) & np.isfinite(window).all((1, 2))
    valid &= (np.diff(sequence, axis=1) == 1).all(1)
    valid &= (np.diff(times, axis=1) <= 50_000_000).all(1)
    valid &= ((observation_ns - times[:, -1] <= 20_000_000) &
              (times[:, 0] >= observation_ns - history_ns))
    valid &= (times <= observation_ns[:, None]).all(1)
    return window.transpose(0, 2, 1), valid


def mesh_masks(vertices, joints, mesh_valid, original_valid, rejected=None, coordinate_status=None):
    count = len(vertices)
    if (vertices.shape != (count, 778, 3) or joints.shape != (count, 21, 3) or
            original_valid.shape != (count, 21) or mesh_valid.shape != (count,)):
        raise ValueError('Bad mesh/joint layout')
    valid = (np.asarray(mesh_valid, bool) & np.isfinite(vertices).all((1, 2)) &
             np.isfinite(joints[:, 0]).all(1))
    unknown = np.full(count, rejected is None, bool)
    if rejected is not None:
        if rejected.shape != (count,):
            raise ValueError('Bad source timing mask')
        valid &= ~np.asarray(rejected, bool)
    if coordinate_status is not None:
        valid &= np.isin(coordinate_status, ['frozen_calibration_unvalidated', 'frozen_calibration_validated'])
    joint_valid = np.asarray(original_valid, bool) & np.isfinite(joints).all(-1) & valid[:, None]
    valid &= joint_valid[:, 0]
    joint_valid &= valid[:, None]
    return dict(mesh_valid=valid, joint_valid=joint_valid, timing_unknown=unknown)


def vertex_partition(faces, face_region, vertices=778):
    faces = np.asarray(faces, dtype=np.int64)
    face_region = np.asarray(face_region, dtype=np.int64)
    if faces.shape != (1538, 3) or face_region.shape != (1538,) or np.any((faces < 0) | (faces >= vertices)):
        raise ValueError('Unexpected anatomical topology')
    if np.any((face_region < -1) | (face_region > 6)):
        raise ValueError('Unknown face region')
    votes = np.zeros((vertices, 7), dtype=np.int64)
    assigned = face_region >= 0
    for corner in range(3):
        np.add.at(votes, (faces[assigned, corner], face_region[assigned]), 1)
    partition = np.where(votes.sum(1) > 0, votes.argmax(1), -1)
    if any(not np.any(partition == region) for region in range(7)):
        raise ValueError('Empty anatomical region')
    return partition


def sparse_vertex_force(faces, partition, face_index, channel, barycentric, force):
    """Lift pressure-cell forces to owner-matching vertices without duplication."""
    face_index = np.asarray(face_index, dtype=np.int64)
    channel = np.asarray(channel, dtype=np.int64)
    barycentric = np.asarray(barycentric, dtype=np.float64)
    force = np.asarray(force, dtype=np.float64)
    if barycentric.shape != (len(force), 3) or face_index.shape != force.shape or channel.shape != force.shape:
        raise ValueError('Sparse contact layout mismatch')
    if np.any((face_index < 0) | (face_index >= len(faces))) or np.any((channel < 0) | (channel > 6)):
        raise ValueError('Bad sparse topology/region ID')
    if (not np.isfinite(barycentric).all() or not np.isfinite(force).all() or
            np.any(force < 0) or np.any(barycentric < -1e-6)):
        raise ValueError('Nonfinite or negative sparse force/support')
    if not np.allclose(barycentric.sum(1), 1, atol=1e-4):
        raise ValueError('Barycentric weights do not sum to one')
    vertices = faces[face_index]
    weights = np.maximum(barycentric, 0) * (partition[vertices] == channel[:, None])
    totals = weights.sum(1)
    supported = totals > 1e-12
    weights = np.divide(weights, totals[:, None], out=np.zeros_like(weights), where=supported[:, None])
    output = np.zeros(len(partition), dtype=np.float64)
    np.add.at(output, vertices.reshape(-1), (weights * force[:, None]).reshape(-1))
    regional = np.array([output[partition == region].sum() for region in range(7)])
    valid = np.ones(7, bool)
    for region in np.unique(channel[(~supported) & (force > 0)]):
        valid[region] = False
    return output.astype(np.float32), regional.astype(np.float32), valid


def explicit_contact_masks(contact, state, frame_good, partition, vertex_force, base_surface_valid):
    known = np.isin(contact, [0, 1])
    global_known = np.isin(state, [0, 1])
    contact_valid = frame_good[:, None] & known & global_known[:, None]
    owners = partition.clip(0)
    support = ((vertex_force > 0) & (partition[None] >= 0) &
               (contact[:, owners] == 1) & (state == 1)[:, None])
    surface_valid = base_surface_valid & known.all(1) & global_known
    return contact_valid, support, surface_valid
