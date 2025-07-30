import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.nyu_loader import NYUDataset
from models.rgb_encoder import RGBEncoder
from models.point_encoder import PointEncoder
from models.fusion_module import FusionModule
from models.decoder import SegDecoder  # You’ll build this

def train_model(config):
    device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')

    dataset = NYUDataset(config['dataset_path'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    rgb_enc = RGBEncoder().to(device)
    pc_enc = PointEncoder().to(device)
    fusion = FusionModule().to(device)
    decoder = SegDecoder(num_classes=config['num_classes']).to(device)

    model = nn.Sequential(rgb_enc, pc_enc, fusion, decoder)  # OR combine manually
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['epochs']):
        for rgb, pc, label in loader:
            rgb, pc, label = rgb.to(device), pc.to(device), label.to(device)

            rgb_feat = rgb_enc(rgb)
            pc_feat = pc_enc(pc)
            fused = fusion(rgb_feat, pc_feat)
            out = decoder(fused)

            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
