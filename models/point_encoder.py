import torch
import torch.nn as nn

class PointEncoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):  # x: (B, N, 3)
        return self.net(x)  # (B, N, out_dim)
