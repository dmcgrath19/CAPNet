from models.rgb_encoder import RGBEncoder
from models.point_encoder import PointEncoder
from models.fusion_module import FusionModule
import torch

rgb = torch.randn(2, 3, 224, 224)
pc = torch.randn(2, 300, 3)

rgb_enc = RGBEncoder()
pc_enc = PointEncoder()
fusion = FusionModule()

rgb_tokens = rgb_enc(rgb)        # (B, N1, D)
pc_tokens = pc_enc(pc)           # (B, N2, D)
fused = fusion(rgb_tokens, pc_tokens)

print("Fused shape:", fused.shape)  # e.g., (2, N1 + prompt, D)
