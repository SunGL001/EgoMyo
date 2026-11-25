"""Interface between this release and an official WiLoR installation.

WiLoR is third party software and is not redistributed here. Install it from
its official repository and implement ``load`` so that it returns an object
with a ``tokens(crops)`` method:

    tokens(crops: torch.FloatTensor[B, 3, 256, 256]) -> torch.FloatTensor[B, C, H, W]

``crops`` are RGB crops in [0, 1] and the returned feature map is the backbone
output used for the refinement stage. The default implementation below follows
the public WiLoR entry points. Adjust only the import names if your checkout
differs.
"""
import torch


def load(checkpoint, device='cuda'):
    from wilor.models import load_wilor  # provided by the official repository

    model = load_wilor(checkpoint_path=str(checkpoint), cfg_path=None)
    model = model.to(device).eval()
    model.requires_grad_(False)
    backbone = model.backbone

    def tokens(crops):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            _, _, _, feature = backbone(crops[:, :, :, 32:-32])
        return feature.float()

    return tokens
