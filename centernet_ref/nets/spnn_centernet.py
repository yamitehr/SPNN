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

from models import (SPNN, ConvPINNBlock, PixelUnshuffleBlock,
                    Cayley1x1Conv, Householder1x1Conv)
from models_deeper import ConvMLP as _DeeperConvMLP


class _DeepHeadConvPINNBlock(ConvPINNBlock):
    """ConvPINNBlock that uses the deeper 3-level U-net ConvMLP from
    models_deeper.py for its t/s/r networks.

    Used ONLY for the detector head (block 4) so that the backbone (blocks
    0-3) remains bit-compatible with checkpoints trained against models.py.

    The mixer (Householder1x1Conv) is identical between models.py and
    models_deeper.py, so it's left untouched.
    """

    def __init__(self, in_ch, out_ch, hidden=64, scale_bound=2.,
                 img_size=32, mix_type="householder", feat_size=None,
                 mlp_tail_hidden=0):
        super().__init__(in_ch=in_ch, out_ch=out_ch, hidden=hidden,
                         scale_bound=scale_bound, img_size=img_size,
                         mix_type=mix_type, feat_size=feat_size)
        # Replace shallow t/s/r with the deeper variant.  When
        # mlp_tail_hidden > 0, each ConvMLP appends a per-pixel MLP residual
        # after its U-net (zero-init last conv → identity at step 0), adding
        # nonlinear channel-direction capacity inside s/t/r.  Bijectivity is
        # unchanged: the coupling y = x0 * s(x1) + t(x1) is invertible for any
        # deterministic s, t — the tail just makes them deeper.
        self.t = _DeeperConvMLP(in_ch - out_ch, out_ch, None, hidden,
                                img_size=img_size, feat_size=feat_size,
                                mlp_tail_hidden=mlp_tail_hidden)
        self.s = _DeeperConvMLP(in_ch - out_ch, out_ch, scale_bound, hidden,
                                img_size=img_size, feat_size=feat_size,
                                mlp_tail_hidden=mlp_tail_hidden)
        self.r = _DeeperConvMLP(out_ch, in_ch - out_ch, None, hidden,
                                img_size=img_size, feat_size=feat_size,
                                mlp_tail_hidden=mlp_tail_hidden)


def transfer_backbone_from_classifier(cls_checkpoint_path, spnn_model):
    """Copy backbone weights from an ImageNet classification checkpoint.

    Copies block indices 0-3 (PU + ConvPINN(12->8) + PU + ConvPINN(32->28)).
    Block 4 (ConvPINN(28->24), the detector head) keeps random init since
    its output channels carry detection-specific semantics.

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
        # Blocks 0-3: shared backbone. Block 4 is the detector head (random init).
        if any(key.startswith(f"pinn.blocks.{i}.") for i in range(4)):
            if key in det_state and det_state[key].shape == val.shape:
                det_state[key] = val
                transferred += 1

    spnn_model.load_state_dict(det_state)
    print(f"[transfer] Copied {transferred} backbone parameter tensors from classification checkpoint")
    return transferred


class SPNNCenterNet(nn.Module):
    """Wraps SPNN to match CenterNet's output format.

    The SPNN is the full end-to-end invertible model.
    Access the raw SPNN via model.spnn for pinv() / DDNM.

    Heatmap head modes (all preserve full bijectivity on the hmap channels,
    so the composition head ∘ SPNN remains surjective and DDNM-invertible):

      head_mode='affine' (default, original):
          y = scale ⊙ raw + bias
          raw = (y - bias) / scale
        Per-class scalar scale + per-class bias. 40 learnable scalars total.

      head_mode='orthogonal_mix':
          y = W (scale ⊙ raw) + bias
          raw = (W^T (y - bias)) / scale
        Adds a learnable C×C orthogonal channel mixer W. Two parameterizations,
        selectable via head_mix_type:
          'cayley'      W = matrix_exp(A − Aᵀ); A inits at 0 so W = I exactly.
                        Continuity with the 'affine' head at init. Full SO(C).
          'householder' W = H_K · ... · H_1, each H = I − 2·v·vᵀ. Random V at
                        init → random orthogonal W. Bit-exact orthogonality
                        (better for DDNM round-trips). With K = num_reflections
                        ≥ C reflections covers all of O(C); default K = C.
        Adds ~C² learnable params (e.g. 400 for C=20).
    """

    def __init__(self, spnn, num_classes=20, hmap_init_scale=0.01,
                 hmap_init_bias=-2.19,
                 head_mode='affine', head_mix_type='householder',
                 head_mix_reflections=None):
        super().__init__()
        self.spnn = spnn
        self.num_classes = num_classes
        if head_mode not in ('affine', 'orthogonal_mix'):
            raise ValueError(f"Unknown head_mode={head_mode!r}; "
                             f"expected 'affine' or 'orthogonal_mix'")
        self.head_mode = head_mode
        self.hmap_scale = nn.Parameter(torch.ones(1, num_classes, 1, 1) * hmap_init_scale)
        self.hmap_bias = nn.Parameter(torch.full((1, num_classes, 1, 1), float(hmap_init_bias)))
        if head_mode == 'orthogonal_mix':
            if head_mix_type == 'cayley':
                self.hmap_mix = Cayley1x1Conv(num_classes)
            elif head_mix_type == 'householder':
                K = head_mix_reflections if head_mix_reflections is not None \
                    else num_classes
                self.hmap_mix = Householder1x1Conv(num_classes, num_reflections=K)
            else:
                raise ValueError(f"Unknown head_mix_type={head_mix_type!r}; "
                                 f"expected 'cayley' or 'householder'")
            self.head_mix_type = head_mix_type

    def forward(self, x):
        raw = self.spnn(x)  # [B, num_classes+4, H/4, W/4]
        hmap_raw = raw[:, :self.num_classes]
        hmap = hmap_raw * self.hmap_scale  # per-channel rescale
        if self.head_mode == 'orthogonal_mix':
            hmap = self.hmap_mix(hmap)     # 20×20 orthogonal channel mix
        hmap = hmap + self.hmap_bias       # per-channel offset
        regs = raw[:, self.num_classes:self.num_classes + 2]  # [B, 2, H/4, W/4]
        w_h_ = raw[:, self.num_classes + 2:]  # [B, 2, H/4, W/4]
        return [[hmap, regs, w_h_]]

    def hmap_to_raw(self, hmap):
        """Invert the head: recover SPNN's raw hmap channels from final logits.

        Used by the pinv / DDNM chain — apply this first, then SPNN.pinv.
        """
        y = hmap - self.hmap_bias.to(hmap.device)
        if self.head_mode == 'orthogonal_mix':
            y = self.hmap_mix.inverse(y)  # apply W^T
        return y / self.hmap_scale.to(hmap.device)


def build_spnn_centernet(num_classes=20, hidden=256, mix_type="householder",
                         scale_bound=1.0, pretrained_backbone=None,
                         hmap_init_scale=0.01, hmap_init_bias=-2.19,
                         head_mode='affine',
                         head_mix_type='householder',
                         head_mix_reflections=None,
                         deep_det_head=False, deep_head_hidden=128,
                         mlp_tail_hidden=0,
                         two_block_head=False):
    """Build end-to-end invertible SPNN for CenterNet detection.

    Architecture (Option C: multi-scale 128+64):
      [3, 256, 256] -> PixelUnshuffle(2) -> [12, 128, 128]
                    -> ConvPINNBlock(12 -> 8)  -> [8, 128, 128]    x1=4
                    -> PixelUnshuffle(2) -> [32, 64, 64]
                    -> ConvPINNBlock(32 -> 28) -> [28, 64, 64]     x1=4

    Detector head (random init, not transferred from classifier):
      two_block_head=False (default):
                    -> ConvPINNBlock(28 -> 24) -> [24, 64, 64]     x1=4
      two_block_head=True:
                    -> ConvPINNBlock(28 -> 26) -> [26, 64, 64]     x1=2
                    -> ConvPINNBlock(26 -> 24) -> [24, 64, 64]     x1=2

    Output: [num_classes+4, 64, 64] at stride 4
      channels 0:num_classes  = class heatmaps
      channels num_classes:+2 = sub-pixel offset (x, y)
      channels +2:+4          = width, height
    """
    out_ch = num_classes + 4  # 20 + 4 = 24 for VOC

    def _head_block(in_ch_b, out_ch_b):
        """Build one detector-head ConvPINN block, deep variant if requested."""
        cls = _DeepHeadConvPINNBlock if deep_det_head else ConvPINNBlock
        kwargs = {"in_ch": in_ch_b, "out_ch": out_ch_b,
                  "hidden": deep_head_hidden if deep_det_head else hidden,
                  "scale_bound": scale_bound, "feat_size": 64,
                  "mix_type": mix_type}
        if deep_det_head:
            kwargs["mlp_tail_hidden"] = mlp_tail_hidden
        return (cls, kwargs)

    layer_channels = [
        # 128x128 processing (fine spatial detail)
        (PixelUnshuffleBlock, {"r": 2}),
        (ConvPINNBlock, {"in_ch": 12, "out_ch": 8, "hidden": hidden,
                         "scale_bound": scale_bound, "feat_size": 128,
                         "mix_type": mix_type}),
        # 64x64 processing
        (PixelUnshuffleBlock, {"r": 2}),
        (ConvPINNBlock, {"in_ch": 32, "out_ch": 28, "hidden": hidden,
                         "scale_bound": scale_bound, "feat_size": 64,
                         "mix_type": mix_type}),
    ]
    # Detector head — random init (not transferred from classifier).
    # When deep_det_head=True, head blocks use the deeper U-net t/s/r nets
    # from models_deeper.py.
    if two_block_head:
        layer_channels.append(_head_block(28, 26))
        layer_channels.append(_head_block(26, out_ch))
    else:
        layer_channels.append(_head_block(28, out_ch))

    spnn = SPNN(
        img_ch=3,
        num_classes=out_ch,
        img_size=256,
        layer_channels=layer_channels,
        output_spatial_size=(64, 64),
    )

    if pretrained_backbone is not None:
        transfer_backbone_from_classifier(pretrained_backbone, spnn)

    return SPNNCenterNet(spnn, num_classes=num_classes,
                         hmap_init_scale=hmap_init_scale,
                         hmap_init_bias=hmap_init_bias,
                         head_mode=head_mode,
                         head_mix_type=head_mix_type,
                         head_mix_reflections=head_mix_reflections)


def get_spnn_centernet(num_classes=20, pretrained_backbone=None,
                       hmap_init_scale=0.01, hmap_init_bias=-2.19,
                       head_mode='affine',
                       head_mix_type='householder',
                       head_mix_reflections=None,
                       deep_det_head=False, deep_head_hidden=128,
                       mlp_tail_hidden=0,
                       two_block_head=False):
    """Entry point matching CenterNet's model creation pattern."""
    return build_spnn_centernet(num_classes=num_classes,
                                pretrained_backbone=pretrained_backbone,
                                hmap_init_scale=hmap_init_scale,
                                hmap_init_bias=hmap_init_bias,
                                head_mode=head_mode,
                                head_mix_type=head_mix_type,
                                head_mix_reflections=head_mix_reflections,
                                deep_det_head=deep_det_head,
                                deep_head_hidden=deep_head_hidden,
                                mlp_tail_hidden=mlp_tail_hidden,
                                two_block_head=two_block_head)
