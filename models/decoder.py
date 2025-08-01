import torch
import torch.nn as nn
import torch.nn.functional as F

class SegDecoder(nn.Module):
    def __init__(self, dim=64, num_classes=40, upsample_size=(224, 224)):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.upsample_size = upsample_size

    def forward(self, x):
        # x: (B, N, D)
        logits = self.head(x)  # (B, N, C)
        
        B, N, C = logits.shape
        H, W = self.upsample_size  # 224,224

        # Compute spatial size of token grid:
        spatial_size = int(N ** 0.5)
        assert spatial_size * spatial_size == N, "N must be a perfect square"

        logit_map = logits.permute(0, 2, 1).reshape(B, C, spatial_size, spatial_size)  # (B, C, h, w)

        # Upsample to target H, W
        logit_map = F.interpolate(logit_map, size=(H, W), mode='bilinear', align_corners=False)

        return logit_map
