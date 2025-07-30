from models.attention import CrossAttention
import torch

rgb = torch.randn(2, 100, 64)  # dummy RGB tokens
pc = torch.randn(2, 300, 64)   # dummy 3D tokens

cross_attn = CrossAttention(64)
out = cross_attn(rgb, pc, pc)  # RGB queries, 3D keys/values
print(out.shape)  # should be (2, 100, 64)
