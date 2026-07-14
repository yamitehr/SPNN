"""
SPNN wrapper for CenterNet (Objects as Points).

Outputs the same format as CenterNet's ResNet/Hourglass models:
  [[hmap, regs, w_h_]]

The SPNN produces a raw [B, C+4, H/4, W/4] tensor which is split into:
  hmap: [B, num_classes, H/4, W/4]  — class heatmaps
  regs: [B, 2, H/4, W/4]           — sub-pixel offset
  w_h_: [B, 2, H/4, W/4]           — width and height

The per-class detection-logit affine (the scale ~0.5 and bias ~-2.19 that
CenterNet conventionally applies after the feature trunk) lives INSIDE the
last head block's s and t networks via models_deeper.ConvMLP's
`final_bias_init` / `final_log_scale_init` parameters. Both are learnable
and bijectivity-preserving (the s inverse uses sign-flipped log_scale via
the neg flag). No external head adapter is constructed here.

Internal-affine support requires --deep_det_head, since only the deeper
ConvMLP (models_deeper.py) carries the final_bias / final_log_scale
parameters. Without --deep_det_head the SPNN raw output is used directly,
with no per-class affine applied.
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
from models_deeper import ConvMLP as _DeeperConvMLP


class _DeepHeadConvPINNBlock(ConvPINNBlock):
    """ConvPINNBlock that uses the deeper 3-level U-net ConvMLP from
    models_deeper.py for its t/s/r networks.

    Used ONLY for the detector head so the backbone (ImageNet-pretrained
    blocks) stays bit-compatible with checkpoints trained against models.py.

    When `t_bias_init` / `s_log_scale_init` are passed (intended for the
    LAST head block — see build_spnn_centernet), each ConvMLP constructs a
    learnable per-output-channel affine inside itself (self.t.final_bias
    and self.s.final_log_scale). These give the detection logits the right
    per-class baseline and scale, with bijectivity preserved by the neg
    flag in the s inverse.
    """

    def __init__(self, in_ch, out_ch, hidden=64, scale_bound=2.,
                 img_size=32, mix_type="householder", feat_size=None,
                 t_bias_init=None, s_log_scale_init=None):
        super().__init__(in_ch=in_ch, out_ch=out_ch, hidden=hidden,
                         scale_bound=scale_bound, img_size=img_size,
                         mix_type=mix_type, feat_size=feat_size)
        # Replace shallow t/s/r with the deeper variant. r is intentionally
        # left without affine — it's only used by pinv() and doesn't see
        # detection-logit dynamic range.
        self.t = _DeeperConvMLP(in_ch - out_ch, out_ch, None, hidden,
                                img_size=img_size, feat_size=feat_size,
                                final_bias_init=t_bias_init)
        self.s = _DeeperConvMLP(in_ch - out_ch, out_ch, scale_bound, hidden,
                                img_size=img_size, feat_size=feat_size,
                                final_log_scale_init=s_log_scale_init)
        self.r = _DeeperConvMLP(out_ch, in_ch - out_ch, None, hidden,
                                img_size=img_size, feat_size=feat_size)


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

    The SPNN is the full end-to-end invertible model; access the raw SPNN
    via `model.spnn` for pinv() / DDNM. The per-class detection-logit
    affine (when --deep_det_head is set) lives inside the last head block's
    s/t networks — see module docstring and build_spnn_centernet. No
    external head adapter is constructed here.
    """

    def __init__(self, spnn, num_classes=20,
                 freeze_backbone=False, num_backbone_blocks=4):
        super().__init__()
        self.spnn = spnn
        self.num_classes = num_classes
        # Backbone-freeze config: when True, forward iterates blocks manually
        # and detaches the tensor between block index `num_backbone_blocks - 1`
        # and the head, so gradients flow only into head blocks.
        self.freeze_backbone = freeze_backbone
        self.num_backbone_blocks = num_backbone_blocks

    def _spnn_forward_with_freeze(self, x):
        """Manual block iteration that detaches between backbone and head.

        Equivalent to self.spnn(x) but with .detach() inserted after the last
        backbone block. Backbone params still build the autograd graph but
        receive no gradient (their .grad stays None after backward); head
        blocks still train normally.
        """
        h = x
        blocks = self.spnn.pinn.blocks
        for block in blocks[:self.num_backbone_blocks]:
            h, _ = block(h, return_latent=False)
        h = h.detach()
        for block in blocks[self.num_backbone_blocks:]:
            h, _ = block(h, return_latent=False)
        return h  # output_spatial_size is set, so SPNN skips the .view step

    def forward(self, x, return_latents=False):
        if self.freeze_backbone:
            if return_latents:
                raise NotImplementedError(
                    "return_latents not supported with freeze_backbone")
            raw = self._spnn_forward_with_freeze(x)
            z_list = None
        else:
            if return_latents:
                raw, z_list = self.spnn(x, return_latents=True)
            else:
                raw = self.spnn(x)  # [B, num_classes+4, H/4, W/4]
                z_list = None
        hmap = raw[:, :self.num_classes]
        regs = raw[:, self.num_classes:self.num_classes + 2]
        w_h_ = raw[:, self.num_classes + 2:]
        if return_latents:
            return [[hmap, regs, w_h_]], z_list
        return [[hmap, regs, w_h_]]

    def pinv(self, hmap, regs, w_h_, latents=None):
        """Right-inverse of forward: y = [hmap, regs, w_h_]  →  reconstructed image.

        Concatenates the three tensors and runs SPNN.pinv directly. There is
        no external head adapter to invert; the per-class affine lives inside
        the last head block's s/t and is inverted automatically by spnn.pinv
        (s uses sign-flipped log_scale via the neg flag).
        """
        raw = torch.cat([hmap, regs, w_h_], dim=1)
        return self.spnn.pinv(raw, latents=latents)


def build_spnn_centernet(num_classes=20, hidden=256, mix_type="householder",
                         scale_bound=1.0, pretrained_backbone=None,
                         hmap_init_scale=0.01, hmap_init_bias=-2.19,
                         deep_det_head=False, deep_head_hidden=128,
                         two_block_head=False,
                         freeze_backbone=False):
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

    Per-class detection-logit affine: when `deep_det_head=True`, the LAST
    head block carries a learnable per-output-channel affine inside its s
    and t (s.final_log_scale, t.final_bias). Init vectors:
      hmap channels (first num_classes): t init = hmap_init_bias,
                                         s init = log(hmap_init_scale)
      regs/wh channels (last 4):         0 (identity).
    Without --deep_det_head, no per-class affine is applied (raw spnn
    output is used as detection logits directly).
    """
    out_ch = num_classes + 4  # 20 + 4 = 24 for VOC

    # When deep_det_head is set, the last head block carries learnable
    # per-output-channel affine inside its s and t. Earlier head blocks (if
    # two_block_head) are NOT class-aligned at their output, so giving them
    # a class-shaped bias would be meaningless — only the last block gets
    # these inits.
    if deep_det_head:
        import math
        t_bias_init = ([float(hmap_init_bias)] * num_classes
                       + [0.0] * 4)
        s_log_scale_init = ([math.log(float(hmap_init_scale))] * num_classes
                            + [0.0] * 4)
    else:
        t_bias_init = None
        s_log_scale_init = None

    def _head_block(in_ch_b, out_ch_b, is_last=False):
        """Build one detector-head ConvPINN block, deep variant if requested."""
        cls = _DeepHeadConvPINNBlock if deep_det_head else ConvPINNBlock
        kwargs = {"in_ch": in_ch_b, "out_ch": out_ch_b,
                  "hidden": deep_head_hidden if deep_det_head else hidden,
                  "scale_bound": scale_bound, "feat_size": 64,
                  "mix_type": mix_type}
        if deep_det_head and is_last:
            kwargs["t_bias_init"] = t_bias_init
            kwargs["s_log_scale_init"] = s_log_scale_init
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
        layer_channels.append(_head_block(28, 26, is_last=False))
        layer_channels.append(_head_block(26, out_ch, is_last=True))
    else:
        layer_channels.append(_head_block(28, out_ch, is_last=True))

    spnn = SPNN(
        img_ch=3,
        num_classes=out_ch,
        img_size=256,
        layer_channels=layer_channels,
        output_spatial_size=(64, 64),
    )

    if pretrained_backbone is not None:
        transfer_backbone_from_classifier(pretrained_backbone, spnn)

    # Backbone is always indices 0-3 (PU + ConvPINN + PU + ConvPINN); head is
    # everything after. The two_block_head flag only adds head blocks past
    # this boundary, so num_backbone_blocks=4 covers both head variants.
    return SPNNCenterNet(spnn, num_classes=num_classes,
                         freeze_backbone=freeze_backbone,
                         num_backbone_blocks=4)


def get_spnn_centernet(num_classes=20, pretrained_backbone=None,
                       hmap_init_scale=0.01, hmap_init_bias=-2.19,
                       deep_det_head=False, deep_head_hidden=128,
                       two_block_head=False,
                       freeze_backbone=False):
    """Entry point matching CenterNet's model creation pattern."""
    return build_spnn_centernet(num_classes=num_classes,
                                pretrained_backbone=pretrained_backbone,
                                hmap_init_scale=hmap_init_scale,
                                hmap_init_bias=hmap_init_bias,
                                deep_det_head=deep_det_head,
                                deep_head_hidden=deep_head_hidden,
                                two_block_head=two_block_head,
                                freeze_backbone=freeze_backbone)
