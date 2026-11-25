#!/usr/bin/env python3
"""Convert an official MANO model file into the buffer archive used here.

The MANO model is licensed and is not redistributed. Pass the path to the
official right-hand model file you obtained from the license holder. The output
archive holds exactly the buffers validated by ``myobridge.mano``.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myobridge.mano import JOINT_MAP, SHAPES, TIP_IDS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mano-model', type=Path, required=True,
                        help='Official MANO right-hand model file')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    import smplx

    model = smplx.create(str(args.mano_model.parent), model_type='mano', is_rhand=True,
                         use_pca=False, flat_hand_mean=False)
    hand_mean = model.hand_mean.detach().clone()
    pose_mean = torch.cat((torch.zeros(3), hand_mean))
    buffers = dict(
        v_template=model.v_template.detach(),
        shapedirs=model.shapedirs.detach(),
        posedirs=model.posedirs.detach(),
        J_regressor=model.J_regressor.detach(),
        parents=model.parents.detach().to(torch.int64),
        lbs_weights=model.lbs_weights.detach(),
        faces_tensor=model.faces.detach().to(torch.int64),
        hand_mean=hand_mean,
        pose_mean=pose_mean,
        extra_joints_idxs=torch.tensor(TIP_IDS, dtype=torch.int64),
        joint_map=torch.tensor(JOINT_MAP, dtype=torch.int64),
    )
    for name, shape in SHAPES.items():
        if tuple(buffers[name].shape) != shape:
            raise ValueError(f'{name} has shape {tuple(buffers[name].shape)}, expected {shape}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{name: value.numpy() for name, value in buffers.items()})
    print('wrote', args.out)


if __name__ == '__main__':
    main()
