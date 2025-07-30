from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch

class NYUDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.rgb_files = sorted(os.listdir(os.path.join(root, 'images')))
        self.depth_files = sorted(os.listdir(os.path.join(root, 'depths')))
        self.label_files = sorted(os.listdir(os.path.join(root, 'labels')))
        self.transform = transform

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb = Image.open(os.path.join(self.root, 'images', self.rgb_files[idx])).convert('RGB')
        depth = np.load(os.path.join(self.root, 'depths', self.depth_files[idx]))  # (H, W)
        label = np.array(Image.open(os.path.join(self.root, 'labels', self.label_files[idx])))

        rgb = rgb.resize((224, 224))
        depth = torch.tensor(depth).float().unsqueeze(0) / 1000.0
        label = torch.tensor(label).long()

        rgb = torch.tensor(np.array(rgb)).permute(2, 0, 1).float() / 255.0
        pc = self.depth_to_point_cloud(depth.squeeze())  # (N, 3)

        return rgb, pc, label

    def depth_to_point_cloud(self, depth):
        # Simple conversion to 3D (placeholder: use camera intrinsics if you want it accurate)
        H, W = depth.shape
        x = torch.arange(0, W).view(1, -1).expand(H, W)
        y = torch.arange(0, H).view(-1, 1).expand(H, W)
        z = depth
        xyz = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        return xyz[::50]  # downsample
