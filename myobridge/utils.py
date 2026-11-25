"""Utilities shared by the command line scripts."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def trainable_state_dict(model):
    """Parameters of the model without the frozen MANO buffers.

    The MANO buffers are licensed geometry, not learned weights, so they are
    kept out of every checkpoint written by this release.
    """
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if not name.startswith('mano.')}


def save_checkpoint(path, model, optimizer, epoch, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(dict(model=trainable_state_dict(model),
                    optimizer=optimizer.state_dict(),
                    epoch=epoch,
                    metadata=metadata), temporary)
    temporary.replace(path)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_partition(path):
    with np.load(path, allow_pickle=False) as archive:
        partition = torch.from_numpy(archive['partition'].copy()).long()
    return partition
