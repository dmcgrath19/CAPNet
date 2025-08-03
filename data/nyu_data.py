from torchvision.transforms import ToPILImage

class NYUDataset(Dataset):
    def __init__(self, split='train', max_points=300, downsample=50):
        self.dataset = load_dataset('tanganke/nyuv2', split=split)
        self.dataset = self.dataset.with_format('torch')
        self.max_points = max_points
        self.downsample = downsample

        self.to_pil = ToPILImage()  # convert tensor to PIL
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        rgb_tensor = sample['image']  # This is a torch.Tensor now

        rgb = self.to_pil(rgb_tensor)  # Convert tensor to PIL Image
        rgb = self.rgb_transform(rgb)  # Now works fine

        depth = sample['depth'].squeeze(0)  # (H, W)
        label = sample['segmentation']      # (H, W)

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0).unsqueeze(0), size=(224, 224),
            mode='bilinear', align_corners=False
        ).squeeze()

        label = torch.nn.functional.interpolate(
            label.unsqueeze(0).unsqueeze(0).float(), size=(224, 224),
            mode='nearest'
        ).long().squeeze()

        pc = self.depth_to_point_cloud(depth)

        return rgb, pc, label
