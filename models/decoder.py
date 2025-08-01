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

        # Calculate spatial size of tokens (assumes perfect square)
        spatial_size = int((N - 5) ** 0.5)  # subtract prompts count if needed

        # If you included prompts, remove them here for reshaping spatial tokens only
        # Example assumes prompts are at the start:
        spatial_logits = logits[:, 5:, :]  # (B, N - 5, C)

        logit_map = spatial_logits.permute(0, 2, 1).reshape(B, C, spatial_size, spatial_size)  # (B, C, h, w)

        # Upsample to full image resolution
        logit_map = F.interpolate(logit_map, size=self.upsample_size, mode='bilinear', align_corners=False)

        return logit_map
