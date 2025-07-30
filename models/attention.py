import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.heads = heads
        self.scale = dim ** -0.5

    def forward(self, q, k, v):
        B, N, D = q.shape
        q = self.q_proj(q).view(B, N, self.heads, -1).transpose(1, 2)
        k = self.k_proj(k).view(B, k.size(1), self.heads, -1).transpose(1, 2)
        v = self.v_proj(v).view(B, v.size(1), self.heads, -1).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)
