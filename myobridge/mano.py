"""In-memory MANO buffers and native linear blend skinning.

The MANO model itself is licensed and is not distributed with this release.
``prepare_mano.py`` converts an official MANO model file into the buffer
archive consumed here. Buffers are validated on construction, so a wrong or
partially updated model file fails loudly instead of silently degrading.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from smplx.lbs import lbs

SHAPES = dict(v_template=(778, 3), shapedirs=(778, 3, 10), posedirs=(135, 2334),
              J_regressor=(16, 778), parents=(16,), lbs_weights=(778, 16),
              faces_tensor=(1538, 3), hand_mean=(45,), pose_mean=(48,),
              extra_joints_idxs=(5,), joint_map=(21,))
TIP_IDS = [744, 320, 443, 554, 671]
JOINT_MAP = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
INTEGER_BUFFERS = {'parents', 'faces_tensor', 'extra_joints_idxs', 'joint_map'}


def strict_buffers(state, prefix='mano.'):
    result = {}
    for name, shape in SHAPES.items():
        key = prefix + name
        if key not in state:
            raise ValueError(f'Missing MANO buffer: {key}')
        value = state[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f'Bad MANO layout: {key}, expected {shape}')
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite MANO buffer: {key}')
        if name in INTEGER_BUFFERS and value.dtype != torch.int64:
            raise ValueError(f'Noninteger MANO buffer: {key}')
        if name not in INTEGER_BUFFERS and not value.is_floating_point():
            raise ValueError(f'Nonfloating MANO buffer: {key}')
        result[name] = value.detach().clone()
    if result['parents'].tolist() != PARENTS:
        raise ValueError('Unexpected MANO kinematic tree')
    if result['extra_joints_idxs'].tolist() != TIP_IDS or result['joint_map'].tolist() != JOINT_MAP:
        raise ValueError('Unexpected fingertip or joint convention')
    for name in ('lbs_weights', 'J_regressor'):
        values = result[name]
        if (values < -1e-6).any() or not torch.allclose(
                values.sum(-1), torch.ones(len(values)), atol=1e-4):
            raise ValueError(f'Bad normalization in {name}')
    faces = result['faces_tensor']
    if (faces < 0).any() or (faces >= 778).any() or any(
            (faces[:, i] == faces[:, j]).any() for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Invalid MANO face topology')
    if result['v_template'].abs().max() < .01 or result['v_template'].abs().max() > 1:
        raise ValueError('Unexpected MANO template scale, expected metres')
    if not torch.allclose(result['pose_mean'][3:], result['hand_mean']) or \
            result['pose_mean'][:3].abs().sum() != 0:
        raise ValueError('Inconsistent MANO pose mean')
    return result


class EmbeddedMANO(nn.Module):
    """Right-hand MANO with an explicit left-hand mirror adapter.

    Rotation matrices represent the complete pose, so the MANO hand mean is not
    added here. Mirroring applies S = diag(-1, 1, 1) to the vertices and
    reverses the face winding while keeping vertex indices unchanged. The
    returned mesh is in metres with no hidden root centering.
    """

    def __init__(self, buffers):
        super().__init__()
        for name, value in strict_buffers({'mano.' + k: v for k, v in buffers.items()}).items():
            self.register_buffer(name, value)

    @classmethod
    def from_npz(cls, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'MANO buffer archive not found: {path}')
        with np.load(path, allow_pickle=False) as archive:
            return cls({name: torch.from_numpy(archive[name].copy()) for name in SHAPES})

    @classmethod
    def from_checkpoint(cls, path, prefix='mano.'):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        checkpoint = torch.load(str(path), map_location='cpu', mmap=True, weights_only=False)
        state = checkpoint.get('state_dict', checkpoint)
        return cls({name: state[prefix + name] for name in SHAPES})

    def forward(self, rotations, betas, *, is_left=None, translation=None, scale=None):
        return forward_mano(self, rotations, betas, is_left=is_left,
                            translation=translation, scale=scale)


def forward_mano(mano, rotations, betas, is_left=None, translation=None, scale=None):
    batch = len(rotations)
    if rotations.shape != (batch, 16, 3, 3) or betas.shape != (batch, 10):
        raise ValueError('Expected rotations [B,16,3,3] and betas [B,10]')
    vertices, joints16 = lbs(betas, rotations, mano.v_template, mano.shapedirs,
                             mano.posedirs, mano.J_regressor, mano.parents,
                             mano.lbs_weights, pose2rot=False)
    joints = torch.cat((joints16, vertices[:, mano.extra_joints_idxs]), dim=1)[:, mano.joint_map]
    if scale is not None:
        if scale.shape not in {(batch,), (batch, 1)}:
            raise ValueError('Scale must be one value per sample')
        vertices = vertices * scale.reshape(batch, 1, 1)
        joints = joints * scale.reshape(batch, 1, 1)
    faces = mano.faces_tensor[None].expand(batch, -1, -1)
    if is_left is not None:
        if is_left.shape != (batch,) or is_left.dtype != torch.bool:
            raise ValueError('is_left must be [B] boolean')
        sign = torch.ones((batch, 3), dtype=vertices.dtype, device=vertices.device)
        sign[:, 0] = torch.where(is_left, -1., 1.)
        vertices, joints = vertices * sign[:, None], joints * sign[:, None]
        faces = torch.where(is_left[:, None, None], faces[:, :, [0, 2, 1]], faces)
    if translation is not None:
        if translation.shape != (batch, 3):
            raise ValueError('Translation must be [B,3]')
        vertices, joints = vertices + translation[:, None], joints + translation[:, None]
    return dict(vertices=vertices, joints=joints, faces=faces)
