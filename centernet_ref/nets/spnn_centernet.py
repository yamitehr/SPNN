"""
SPNN wrapper for CenterNet (Objects as Points).

Outputs the same format as CenterNet's ResNet/Hourglass models:
  [[hmap, regs, w_h_]]

The SPNN produces a raw [B, C+4, H/4, W/4] tensor which is split into:
  hmap: [B, num_classes, H/4, W/4]  — class heatmaps
  regs: [B, 2, H/4, W/4]           — sub-pixel offset
  w_h_: [B, 2, H/4, W/4]           — width and height
"""

import sys
import os
import torch
import torch.nn as nn

# Add project root to path so we can import models.py
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models import SPNN, ConvPINNBlock, PixelUnshuffleBlock


def transfer_backbone_from_classifier(cls_checkpoint_path, spnn_model):
    """Copy backbone weights from an ImageNet classification checkpoint.

    The classifier's blocks 0-1 (PixelUnshuffle + ConvPINNBlock(48->24))
    are identical to the CenterNet SPNN architecture.
    Returns the number of transferred parameter tensors.
    """
    import torch
    raw = torch.load(cls_checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        cls_state = raw["state_dict"]
    else:
        cls_state = raw

    # Normalize keys: strip common prefixes from various checkpoint formats
    normalized = {}
    for key, val in cls_state.items():
        k = key
        for prefix in ["module.", "fcos_body.backbone.spnn.", "fcos_body.backbone.",
                        "spnn."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        normalized[k] = val

    det_state = spnn_model.state_dict()
    transferred = 0
    for key, val in normalized.items():
        # Blocks 0-1: PixelUnshuffle (no params) + ConvPINNBlock(48->24)
        if any(key.startswith(f"pinn.blocks.{i}.") for i in range(2)):
            if key in det_state and det_state[key].shape == val.shape:
                det_state[key] = val
                transferred += 1

    spnn_model.load_state_dict(det_state)
    print(f"[transfer] Copied {transferred} backbone parameter tensors from classification checkpoint")
    return transferred


class SPNNCenterNet(nn.Module):
    """Wraps SPNN to match CenterNet's output format.

    The SPNN is the full end-to-end invertible model.
    This wrapper only splits the output channels — adds zero parameters.
    Access the raw SPNN via model.spnn for pinv() / DDNM.
    """

    def __init__(self, spnn, num_classes=20):
        super().__init__()
        self.spnn = spnn
        self.num_classes = num_classes
        # Scale + bias for heatmap channels.
        # SPNN output is in [-4, 4] range. Without scaling, sigmoid ≈ 0.5 everywhere,
        # which causes focal loss to explode. We start with small scale (0.01) so
        # initial output is dominated by bias (-2.19), giving sigmoid ≈ 0.1.
        # The scale grows during training as the SPNN learns meaningful heatmaps.
        self.hmap_scale = nn.Parameter(torch.ones(1, num_classes, 1, 1) * 0.01)
        self.hmap_bias = nn.Parameter(torch.full((1, num_classes, 1, 1), -2.19))

    def forward(self, x):
        raw = self.spnn(x)  # [B, num_classes+4, H/4, W/4]
        hmap = raw[:, :self.num_classes] * self.hmap_scale + self.hmap_bias  # [B, 20, H/4, W/4]
        regs = raw[:, self.num_classes:self.num_classes + 2]  # [B, 2, H/4, W/4]
        w_h_ = raw[:, self.num_classes + 2:]  # [B, 2, H/4, W/4]
        return [[hmap, regs, w_h_]]


def build_spnn_centernet(num_classes=20, hidden=256, mix_type="householder",
                         scale_bound=1.0, pretrained_backbone=None):
    """Build end-to-end invertible SPNN for CenterNet detection.

    Architecture:
      [3, 256, 256] -> PixelUnshuffle(4) -> [48, 64, 64]
                    -> ConvPINNBlock(48 -> num_classes+4) -> [24, 64, 64]

    Output: [num_classes+4, 64, 64] at stride 4
      channels 0:num_classes  = class heatmaps
      channels num_classes:+2 = sub-pixel offset (x, y)
      channels +2:+4          = width, height
    """
    out_ch = num_classes + 4  # 20 + 4 = 24 for VOC

    layer_channels = [
        (PixelUnshuffleBlock, {"r": 4}),
        (ConvPINNBlock, {"in_ch": 48, "out_ch": out_ch, "hidden": hidden,
                         "scale_bound": scale_bound, "feat_size": 64,
                         "mix_type": mix_type}),
    ]

    spnn = SPNN(
        img_ch=3,
        num_classes=out_ch,
        img_size=256,
        layer_channels=layer_channels,
        output_spatial_size=(64, 64),
    )

    if pretrained_backbone is not None:
        transfer_backbone_from_classifier(pretrained_backbone, spnn)

    return SPNNCenterNet(spnn, num_classes=num_classes)


def get_spnn_centernet(num_classes=20, pretrained_backbone=None):
    """Entry point matching CenterNet's model creation pattern."""
    return build_spnn_centernet(num_classes=num_classes,
                                pretrained_backbone=pretrained_backbone)
