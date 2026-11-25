"""External HaMeR/WiLoR native geometry plus prediction-only added task heads."""
import sys
from pathlib import Path

from torch import nn

from .cached import DirectHeads


class VisualBaseline(nn.Module):
    def __init__(self, method, assets):
        super().__init__()
        sys.path.insert(0, str(Path(assets['source_root']).resolve()))
        if method == 'wilor':
            from wilor.models import load_wilor
            self.native, config = load_wilor(assets['checkpoint'], assets['config'])
            readout = self.native.refine_net.dec_pose
        else:
            from hamer.models import load_hamer
            self.native, config = load_hamer(assets['checkpoint'])
            readout = self.native.mano_head.decpose
        if list(config.MODEL.IMAGE_MEAN) != [.485, .456, .406] or list(config.MODEL.IMAGE_STD) != [.229, .224, .225]:
            raise ValueError('RGB crops require the documented ImageNet normalization')
        if int(config.MODEL.IMAGE_SIZE) != 256:
            raise ValueError('RGB crop size must match the external model configuration')
        self.native.mano.requires_grad_(False)
        self.features = None
        self.handle = readout.register_forward_pre_hook(self.capture)
        self.added = DirectHeads(readout.in_features, ('regional_contact', 'regional_force', 'vertex_contact', 'vertex_force'))

    def capture(self, module, inputs):
        self.features = inputs[0]

    def forward(self, batch):
        self.features = None
        native = self.native({name: batch[name] for name in ('img', 'right', 'box_center', 'box_size', 'img_size')})
        if self.features is None or self.features.shape[0] != len(batch['img']) or self.features.ndim != 2:
            raise ValueError('External model shared readout changed; review upstream version')
        joints = native['pred_keypoints_3d']
        root = joints[:, :1]
        return dict(joints=joints - root, mesh=native['pred_vertices'] - root, **self.added(self.features))
