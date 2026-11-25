import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myobridge.mano import JOINT_MAP, PARENTS, TIP_IDS  # noqa: E402


def synthetic_mano_buffers(seed=0):
    """Shape-valid MANO buffers for tests, not a substitute for the real model."""
    generator = torch.Generator().manual_seed(seed)
    vertices = 778
    faces = torch.tensor([[i, i + 1, i + 2] for i in range(0, 3 * 1538, 3)], dtype=torch.int64)
    faces = faces % vertices
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])
    faces = faces[keep][:1538]
    weights = torch.rand(vertices, 16, generator=generator)
    weights = weights / weights.sum(-1, keepdim=True)
    regressor = torch.rand(16, vertices, generator=generator)
    regressor = regressor / regressor.sum(-1, keepdim=True)
    hand_mean = torch.randn(45, generator=generator) * .05
    return dict(
        v_template=torch.randn(vertices, 3, generator=generator) * .05 + .1,
        shapedirs=torch.randn(vertices, 3, 10, generator=generator) * .01,
        posedirs=torch.randn(135, vertices * 3, generator=generator) * .01,
        J_regressor=regressor,
        parents=torch.tensor(PARENTS, dtype=torch.int64),
        lbs_weights=weights,
        faces_tensor=faces,
        hand_mean=hand_mean,
        pose_mean=torch.cat((torch.zeros(3), hand_mean)),
        extra_joints_idxs=torch.tensor(TIP_IDS, dtype=torch.int64),
        joint_map=torch.tensor(JOINT_MAP, dtype=torch.int64),
    )


def synthetic_partition():
    return (torch.arange(778, dtype=torch.int64) % 7)
