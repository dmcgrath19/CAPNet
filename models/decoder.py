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

        # Assuming first tokens align with RGB (after prompts)
        B, N, C = logits.shape
        H, W = self.upsample_size
        logit_map = logits[:, -H * W:, :]  # (B, H*W, C)
        logit_map = logit_map.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
        return logit_map
