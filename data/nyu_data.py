from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import torch
from torchvision import transforms

class NYUDataset(Dataset):
    def __init__(self, split='train', max_points=300, downsample=50):
        self.dataset = load_dataset('tanganke/nyuv2', split=split)
        self.dataset = self.dataset.with_format('torch')
        self.max_points = max_points
        self.downsample = downsample

        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        rgb = sample['image']  # PIL.Image.Image
        depth = sample['depth'].squeeze(0)  # (H, W)
        label = sample['segmentation']      # (H, W)

        rgb = self.rgb_transform(rgb)  # (3, 224, 224)
        depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze()
        label = torch.nn.functional.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=(224, 224), mode='nearest').long().squeeze()

        pc = self.depth_to_point_cloud(depth)  # (N, 3)

        return rgb, pc, label

    def depth_to_point_cloud(self, depth):
        H, W = depth.shape
        x = torch.arange(0, W).view(1, -1).expand(H, W)
        y = torch.arange(0, H).view(-1, 1).expand(H, W)
        z = depth
        xyz = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        return xyz[::self.downsample][:self.max_points]  # limit number of points
