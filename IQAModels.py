import torch
import torch.nn as nn
from open_clip import create_model


# SigLIP2_384_Image: Image Quality Assessment version of SigLIP2_384.
# Input : single image  batch x 3 x H x W
# Output: quality score batch
class SigLIP2_384_Image(nn.Module):
    def __init__(self):
        super(SigLIP2_384_Image, self).__init__()

        model = create_model('ViT-B-16-SigLIP2-384')

        # spatial quality analyzer
        self.feature_extraction = model.visual

        # quality regressor
        self.quality = self.quality_regression(768, 128, 1)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x):
        # x: batch x 3 x H x W
        x = self.feature_extraction(x)   # batch x 768
        x = torch.flatten(x, 1)
        x = self.quality(x)              # batch x 1
        return x.squeeze(1)              # batch


class SigLIP2_ViTG_384_Image_MT(nn.Module):
    """Multi-task SigLIP2 ViT-SO400M/14 378px — predicts mos + 6 quality attributes."""

    TASK_NAMES = ['mos', 'light', 'color', 'noise',
                  'exposure', 'nature', 'content_recovery']

    def __init__(self):
        super(SigLIP2_ViTG_384_Image_MT, self).__init__()

        model = create_model('ViT-SO400M-14-SigLIP2-378')
        self.feature_extraction = model.visual

        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(1152, 128),
                nn.Linear(128, 1),
            )
            for name in self.TASK_NAMES
        })

    def forward(self, x):
        feat = self.feature_extraction(x)   # batch x 1152
        feat = torch.flatten(feat, 1)
        return {name: head(feat).squeeze(1) for name, head in self.heads.items()}
