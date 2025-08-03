import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.nyu_data import NYUDataset 
from models.rgb_encoder import RGBEncoder
from models.point_encoder import PointEncoder
from models.fusion_module import FusionModule
from models.decoder import SegDecoder

def train_model(config):
    device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')

    dataset = NYUDataset(split='train')
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    rgb_enc = RGBEncoder().to(device)
    pc_enc = PointEncoder().to(device)
    fusion = FusionModule().to(device)
    decoder = SegDecoder(num_classes=config['num_classes']).to(device)

    optimizer = torch.optim.Adam(
        list(rgb_enc.parameters()) +
        list(pc_enc.parameters()) +
        list(fusion.parameters()) +
        list(decoder.parameters()),
        lr=config['lr']
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['epochs']):
        rgb_enc.train()
        pc_enc.train()
        fusion.train()
        decoder.train()

        total_loss = 0
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

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
