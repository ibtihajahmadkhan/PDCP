import torch
import torch.nn as nn
from torchvision import models


class ResNet18_Gray(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Change first conv to accept 1 channel instead of 3
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # If pretrained, initialize new conv1 using averaged RGB weights
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # Replace classifier head for binary logit
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)
