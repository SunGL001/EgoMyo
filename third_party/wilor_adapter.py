"""Interface between this release and an official WiLoR installation.

WiLoR is third party software and is not redistributed here. Install it from
its official repository. This adapter accepts cached RGB crops in [0, 1],
applies the official WiLoR ImageNet normalization, removes the 32-pixel side
strips used by the official inference path, and returns the backbone map.

    tokens(crops: torch.FloatTensor[B, 3, 256, 256]) -> torch.FloatTensor[B, C, H, W]

The model config is required because it defines the crop normalization values.
"""
import torch


def load(checkpoint, config, device='cuda'):
    from wilor.models import load_wilor  # provided by the official repository

    model, model_cfg = load_wilor(checkpoint_path=str(checkpoint), cfg_path=str(config))
    model = model.to(device).eval()
    model.requires_grad_(False)
    backbone = model.backbone
    mean = torch.tensor(model_cfg.MODEL.IMAGE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(model_cfg.MODEL.IMAGE_STD, device=device).view(1, 3, 1, 1)

    def tokens(crops):
        crops = (crops - mean) / std
        device_type = torch.device(device).type
        with torch.no_grad(), torch.autocast(device_type, dtype=torch.float16,
                                             enabled=device_type == 'cuda'):
            _, _, _, feature = backbone(crops[:, :, :, 32:-32])
        return feature.float()

    return tokens
