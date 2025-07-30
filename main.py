from models.rgb_encoder import RGBEncoder
from models.point_encoder import PointEncoder
from models.fusion_module import FusionModule
from models.decoder import SegDecoder
import torch

rgb = torch.randn(2, 3, 224, 224)
pc = torch.randn(2, 300, 3)

rgb_enc = RGBEncoder()
pc_enc = PointEncoder()
fusion = FusionModule()
decoder = SegDecoder()

rgb_tokens = rgb_enc(rgb)
pc_tokens = pc_enc(pc)
fused = fusion(rgb_tokens, pc_tokens)
out = decoder(fused)

print("Output shape:", out.shape)  # should be (2, num_classes, 224, 224)
