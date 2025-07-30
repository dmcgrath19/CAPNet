import torch
import torch.nn as nn
import torchvision.models as models

class RGBEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # remove FC layers
        self.project = nn.Conv2d(512, out_dim, kernel_size=1)  # project to transformer dim

    def forward(self, x):
        feat = self.encoder(x)        # (B, 512, H/32, W/32)
        proj = self.project(feat)     # (B, out_dim, H/32, W/32)
        B, C, H, W = proj.shape
        return proj.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
