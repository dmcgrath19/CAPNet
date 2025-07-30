import torch
import torch.nn as nn

class PromptTokens(nn.Module):
    def __init__(self, num_prompts=5, dim=64):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(1, num_prompts, dim))

    def forward(self, B):
        return self.prompts.expand(B, -1, -1)  # (B, num_prompts, dim)
