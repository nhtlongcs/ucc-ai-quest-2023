from typing import List

import timm
import torch
from segmentation_models_pytorch.encoders._base import EncoderMixin
from torch import nn


class RegNetY_SEER(nn.Module, EncoderMixin):
    def __init__(self, **kwargs):
        super().__init__()

        self.backbone = timm.models.create_model("regnety_640.seer", pretrained=True, features_only=True)

        self._in_channels = 3
        self._out_channels = self.backbone.feature_info.channels()  # [128, 256, 512, 1024]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
        """

        """
        torch.Size([1, 128, 16, 16])
        torch.Size([1, 256, 8, 8])
        torch.Size([1, 512, 4, 4])
        torch.Size([1, 1024, 2, 2])
        """
        features = self.backbone(x)
        return features
