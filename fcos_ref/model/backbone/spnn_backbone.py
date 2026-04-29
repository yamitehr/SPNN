"""
SPNN backbone for FCOS detection.

Outputs a single feature map [B, 48, 16, 16] at stride 16 for 256x256 input.
The FPN neck then builds multi-scale features from this single output.

Architecture (blocks 0-5, shared with ImageNet classifier):
  PixelUnshuffle(4)           # [3, 256, 256] -> [48, 64, 64]
  ConvPINNBlock(48 -> 24)     # [24, 64, 64]
  ConvPINNBlock(24 -> 12)     # [12, 64, 64]
  PixelUnshuffle(4)           # [12, 64, 64] -> [192, 16, 16]
  ConvPINNBlock(192 -> 96)    # [96, 16, 16]
  ConvPINNBlock(96 -> 48)     # [48, 16, 16]  <- output
"""

import sys
import os
import torch
import torch.nn as nn

# Add project root to path so we can import models.py
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models import SPNN, ConvPINNBlock, PixelUnshuffleBlock


def build_spnn_backbone(hidden=256, mix_type="cayley", scale_bound=2.0):
    """Build the SPNN backbone (blocks 0-5) that outputs [B, 48, 16, 16].

    This is the same architecture as the ImageNet classifier backbone,
    so pretrained weights can be transferred directly.
    """
    layer_channels = [
        (PixelUnshuffleBlock, {"r": 4}),
        (ConvPINNBlock, {"in_ch": 48, "out_ch": 24, "hidden": hidden,
                         "scale_bound": scale_bound, "feat_size": 64, "mix_type": mix_type}),
        (ConvPINNBlock, {"in_ch": 24, "out_ch": 12, "hidden": hidden,
                         "scale_bound": scale_bound, "feat_size": 64, "mix_type": mix_type}),
        (PixelUnshuffleBlock, {"r": 4}),
        (ConvPINNBlock, {"in_ch": 192, "out_ch": 96, "hidden": hidden,
                         "scale_bound": scale_bound, "feat_size": 16, "mix_type": mix_type}),
        (ConvPINNBlock, {"in_ch": 96, "out_ch": 48, "hidden": hidden,
                         "scale_bound": scale_bound, "feat_size": 16, "mix_type": mix_type}),
    ]

    model = SPNN(
        img_ch=3,
        num_classes=48,
        img_size=256,
        layer_channels=layer_channels,
        output_spatial_size=(16, 16),
    )
    return model


def transfer_backbone_from_classifier(cls_checkpoint_path, spnn_backbone):
    """Copy backbone weights from an ImageNet classification checkpoint.

    The classifier has blocks 0-5 identical to this backbone.
    Returns the number of transferred parameter tensors.
    """
    raw = torch.load(cls_checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        cls_state = raw["state_dict"]
    else:
        cls_state = raw

    backbone_state = spnn_backbone.state_dict()
    transferred = 0

    # Normalize keys: strip common prefixes from FCOS/DataParallel checkpoints
    normalized = {}
    for key, val in cls_state.items():
        k = key
        for prefix in ["module.", "fcos_body.backbone.spnn.", "fcos_body.backbone."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        normalized[k] = val

    for key, val in normalized.items():
        # Blocks 0-5 are the backbone in both classifier and detector
        if any(key.startswith(f"pinn.blocks.{i}.") for i in range(6)):
            if key in backbone_state and backbone_state[key].shape == val.shape:
                backbone_state[key] = val
                transferred += 1

    spnn_backbone.load_state_dict(backbone_state)
    print(f"[transfer] Copied {transferred} backbone parameter tensors from classification checkpoint")
    return transferred


def build_spnn_e2e(hidden=256, mix_type="cayley", scale_bound=2.0):
    """Build end-to-end invertible SPNN: image [3,256,256] -> detection grid [25,16,16].

    Backbone (blocks 0-5) + single detection block (block 6), all invertible.
    Output: 25 channels = 4 (ltrb) + 1 (objectness) + 20 (classes)
    Single detection block has x1=23 channels — rich conditioning signal.
    """
    layer_channels = [
        # Backbone (shared with classifier, blocks 0-5)
        (PixelUnshuffleBlock, {"r": 4}),
        (ConvPINNBlock, {"in_ch": 48, "out_ch": 24, "hidden": hidden,
                         "scale_bound": scale_bound, "feat_size": 64, "mix_type": mix_type}),
        (ConvPINNBlock, {"in_ch": 24, "out_ch": 12, "hidden": hidden,
                         "scale_bound": scale_bound, "feat_size": 64, "mix_type": mix_type}),
        (PixelUnshuffleBlock, {"r": 4}),
        (ConvPINNBlock, {"in_ch": 192, "out_ch": 96, "hidden": hidden,
                         "scale_bound": scale_bound, "feat_size": 16, "mix_type": mix_type}),
        (ConvPINNBlock, {"in_ch": 96, "out_ch": 48, "hidden": hidden,
                         "scale_bound": scale_bound, "feat_size": 16, "mix_type": mix_type}),
        # Detection head (block 6, invertible) — x1=23ch, x0=25ch
        (ConvPINNBlock, {"in_ch": 48, "out_ch": 25, "hidden": hidden,
                         "scale_bound": scale_bound, "feat_size": 16, "mix_type": mix_type}),
    ]

    return SPNN(
        img_ch=3,
        num_classes=25,
        img_size=256,
        layer_channels=layer_channels,
        output_spatial_size=(16, 16),
    )


class SPNNBackbone(nn.Module):
    """Wraps the SPNN backbone for use in FCOS.

    Input:  [B, 3, 256, 256]
    Output: single feature map [B, 48, 16, 16] (stride 16)

    Unlike ResNet which returns (C3, C4, C5), this returns a single tensor.
    The FPN neck handles creating multi-scale features.
    """

    def __init__(self, hidden=256, mix_type="cayley", scale_bound=2.0, pretrained_path=None):
        super().__init__()
        self.spnn = build_spnn_backbone(hidden=hidden, mix_type=mix_type, scale_bound=scale_bound)
        self.out_channels = 48

        if pretrained_path is not None:
            transfer_backbone_from_classifier(pretrained_path, self.spnn)

    def forward(self, x):
        """Returns single feature map [B, 48, 16, 16]."""
        return self.spnn(x)

    def freeze_stages(self, stage):
        """Freeze early blocks for fine-tuning.

        stage=1: freeze blocks 0-1 (PixelUnshuffle + first ConvPINNBlock)
        stage=2: freeze blocks 0-3 (first two ConvPINNBlocks + PixelUnshuffles)
        """
        blocks = list(self.spnn.pinn.blocks)
        freeze_up_to = min(stage * 2, len(blocks))
        for i in range(freeze_up_to):
            for param in blocks[i].parameters():
                param.requires_grad = False
        print(f"INFO===>frozen SPNN backbone blocks 0-{freeze_up_to - 1}")
