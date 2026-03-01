from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA
from torchvision import transforms
import os
from PIL import Image
import torch

def get_celeba_loaders(batch_size, img_size):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = CelebA(root="~/datasets", split="train", target_type="attr", download=True, transform=transform)
    val_set   = CelebA(root="~/datasets", split="valid", target_type="attr", download=False, transform=transform)
    test_set  = CelebA(root="~/datasets", split="test",  target_type="attr", download=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

class CelebAHQLocalDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of (image_path, attribute_tensor) tuples
        """
        self.data = samples
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, attrs = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, attrs

def get_local_celebahq_loaders(root, batch_size, img_size=256):
    # 1. Parse metadata once
    image_dir = os.path.join(root, "CelebA-HQ-img")
    attr_file = os.path.join(root, "CelebAMask-HQ-attribute-anno.txt")
    
    if not os.path.exists(image_dir) or not os.path.exists(attr_file):
        raise FileNotFoundError(f"Could not find CelebA-HQ-img or CelebAMask-HQ-attribute-anno.txt in {root}")

    print(f"Parsing annotations from {attr_file}...")
    with open(attr_file, "r") as f:
        lines = f.readlines()
        
    all_items = []
    # Lines start from index 2
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 2: continue
        
        img_name = parts[0]
        full_img_path = os.path.join(image_dir, img_name)

        # Parse attributes
        attrs = [int(x) for x in parts[1:]]
        attrs = torch.tensor(attrs, dtype=torch.float32)
        # Map -1 to 0, 1 remains 1. (x + 1) // 2
        attrs = (attrs + 1) // 2 
        
        all_items.append((full_img_path, attrs))
        
    # 2. Sort numerically (e.g., 0.jpg, 1.jpg ... 10.jpg)
    try:
        all_items.sort(key=lambda x: int(os.path.basename(x[0]).split('.')[0]))
    except ValueError:
        all_items.sort(key=lambda x: x[0])

    # 3. Split the data (80% / 10% / 10%)
    n = len(all_items)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    train_data = all_items[:n_train]
    val_data   = all_items[n_train:n_train+n_val]
    test_data  = all_items[n_train+n_val:]
    
    print(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # 4. Create Datasets and Loaders
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = CelebAHQLocalDataset(train_data, transform=train_transform)
    val_set   = CelebAHQLocalDataset(val_data,   transform=eval_transform)
    test_set  = CelebAHQLocalDataset(test_data,  transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
