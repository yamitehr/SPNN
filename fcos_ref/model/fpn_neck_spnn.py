"""
Single-scale FPN neck for SPNN backbone.

Takes a single feature map [B, 48, 16, 16] from the SPNN backbone
and builds 5-level multi-scale features [P3, P4, P5, P6, P7] at 256 channels,
matching the standard FCOS FPN output format.

For 256x256 input:
  P3: [B, 256, 32, 32]  stride 8
  P4: [B, 256, 16, 16]  stride 16  (backbone output level)
  P5: [B, 256, 8, 8]    stride 32
  P6: [B, 256, 4, 4]    stride 64
  P7: [B, 256, 2, 2]    stride 128
"""

import torch.nn as nn
import torch.nn.functional as F
import math


class FPN_SPNN(nn.Module):
    """FPN neck that creates multi-scale features from SPNN's single output."""

    def __init__(self, in_channels=48, features=256):
        super().__init__()

        # Project SPNN channels to FPN feature dim
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=1),
            nn.GroupNorm(32, features),
            nn.ReLU(inplace=True),
        )

        # P3: upsample 2x from P4 + refine
        self.upsample_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GroupNorm(32, features),
            nn.ReLU(inplace=True),
        )

        # P4: refine projected feature
        self.refine_p4 = nn.Conv2d(features, features, kernel_size=3, padding=1)

        # P5: downsample from P4
        self.down_p5 = nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1)

        # P6: downsample from P5
        self.down_p6 = nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1)

        # P7: downsample from P6
        self.down_p7 = nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, feat):
        """
        Args:
            feat: [B, 48, 16, 16] from SPNN backbone

        Returns:
            [P3, P4, P5, P6, P7] — list of 5 feature maps at 256 channels
        """
        # Project to 256 channels
        P4 = self.project(feat)  # [B, 256, 16, 16]

        # Upsample for P3
        P3_up = F.interpolate(P4, scale_factor=2, mode='nearest')  # [B, 256, 32, 32]
        P3 = self.upsample_conv(P3_up)

        # Refine P4
        P4 = self.refine_p4(P4)

        # Downsample for P5, P6, P7
        P5 = self.down_p5(F.relu(P4))    # [B, 256, 8, 8]
        P6 = self.down_p6(F.relu(P5))    # [B, 256, 4, 4]
        P7 = self.down_p7(F.relu(P6))    # [B, 256, 2, 2]

        return [P3, P4, P5, P6, P7]
