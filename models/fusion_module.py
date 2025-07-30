import torch
import torch.nn as nn
from .attention import CrossAttention
from .prompts import PromptTokens

class FusionModule(nn.Module):
    def __init__(self, dim=64, num_prompts=5):
        super().__init__()
        self.prompt_rgb = PromptTokens(num_prompts, dim)
        self.prompt_3d = PromptTokens(num_prompts, dim)
        self.cross_attn = CrossAttention(dim)

    def forward(self, rgb_tokens, pc_tokens):
        B = rgb_tokens.size(0)
        rgb_full = torch.cat([self.prompt_rgb(B), rgb_tokens], dim=1)
        pc_full = torch.cat([self.prompt_3d(B), pc_tokens], dim=1)

        fused = self.cross_attn(rgb_full, pc_full, pc_full)  # RGB queries 3D
        return fused  # (B, prompt + N_rgb, dim)
