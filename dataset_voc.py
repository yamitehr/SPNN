"""
Pascal VOC dataset for SPNN detection (S=8, B=1, C=20).

Two dataset classes:
  1. VOCDetectionDataset — uses torchvision.datasets.VOCDetection directly.
     Handles download, XML parsing, and grid encoding automatically.
     No manual conversion needed.

  2. VOCYoloDataset — reads pre-converted YOLO .txt labels + CSV index.
     For use with the aladdinpersson YOLO repo data format.

Usage (recommended):
    from dataset_voc import get_voc_loaders
    train_loader, val_loader = get_voc_loaders(root="/path/to/voc", batch_size=16)
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image


# VOC class names in order (index = class_id)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
VOC_CLASS_TO_IDX = {name: idx for idx, name in enumerate(VOC_CLASSES)}


class VOCDetectionDataset(torch.utils.data.Dataset):
    """
    Wraps torchvision.datasets.VOCDetection and encodes targets into
    a fixed [S, S, C+5] grid tensor for YOLOv1-style training.

    Args:
        root: Path to store/find VOC data (e.g. "/data/voc")
        year: "2007" or "2012"
        image_set: "train", "val", or "trainval"
        download: Whether to download the dataset
        S: Grid size (default 8)
        C: Number of classes (default 20)
        img_size: Resize images to this size (default 256)
    """

    def __init__(self, root, year="2012", image_set="train", download=False,
                 S=8, C=20, img_size=256, augment=False):
        self.voc = torchvision.datasets.VOCDetection(
            root=root, year=year, image_set=image_set, download=download,
        )
        self.S = S
        self.C = C
        self.img_size = img_size
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, index):
        img, target = self.voc[index]

        # Parse XML annotation
        annotation = target["annotation"]
        img_w = int(annotation["size"]["width"])
        img_h = int(annotation["size"]["height"])

        objects = annotation.get("object", [])
        if not isinstance(objects, list):
            objects = [objects]

        # Extract boxes as normalized [class_id, cx, cy, w, h]
        boxes = []
        for obj in objects:
            class_name = obj["name"]
            if class_name not in VOC_CLASS_TO_IDX:
                continue
            class_id = VOC_CLASS_TO_IDX[class_name]

            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"]) / img_w
            ymin = float(bbox["ymin"]) / img_h
            xmax = float(bbox["xmax"]) / img_w
            ymax = float(bbox["ymax"]) / img_h

            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin

            boxes.append([class_id, cx, cy, w, h])

        # Data augmentation: random horizontal flip
        if self.augment and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [[cls, 1 - cx, cy, w, h] for cls, cx, cy, w, h in boxes]

        # Apply image transform
        img = self.transform(img)

        # Encode into grid: [S, S, C + 5*B] = [S, S, 30]
        # B=2 boxes in the target tensor, but only the first box (channels 20-24) is filled.
        # The loss compares both predicted boxes against this single GT box.
        B = 2
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * B))
        for box in boxes:
            class_id, cx, cy, w, h = box
            class_id = int(class_id)

            # Cell row and column
            j = int(self.S * cx)
            i = int(self.S * cy)
            j = min(j, self.S - 1)
            i = min(i, self.S - 1)

            # Offset within cell
            x_cell = self.S * cx - j
            y_cell = self.S * cy - i
            w_cell = w * self.S
            h_cell = h * self.S

            # One object per cell
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1  # objectness
                label_matrix[i, j, 21:25] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                label_matrix[i, j, class_id] = 1  # one-hot class

        return img, label_matrix


def get_voc_loaders(root, batch_size=16, img_size=256, S=8, C=20,
                    year="2012", download=False, num_workers=2):
    """
    Create train and val data loaders for Pascal VOC.

    Args:
        root: Path to VOC data root
        batch_size: Batch size
        img_size: Image resize target (default 256)
        S: Grid size (default 8)
        C: Number of classes (default 20)
        year: VOC year — "2007" or "2012"
        download: Whether to download dataset
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = VOCDetectionDataset(
        root=root, year=year, image_set="train", download=download,
        S=S, C=C, img_size=img_size, augment=True,
    )
    val_dataset = VOCDetectionDataset(
        root=root, year=year, image_set="val", download=download,
        S=S, C=C, img_size=img_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    return train_loader, val_loader


# ─── Legacy format support (aladdinpersson repo) ───

import os
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


class VOCYoloDataset(torch.utils.data.Dataset):
    """
    Reads pre-converted YOLO-format labels (.txt files with class x y w h)
    and a CSV index file. For use with the aladdinpersson YOLO repo data format.

    Expected label format per line: class_id x_center y_center width height (normalized 0-1).
    CSV format: image_filename,label_filename
    """
    def __init__(self, csv_file, img_dir, label_dir, S=8, B=1, C=20, transform=None):
        if not _HAS_PANDAS:
            raise ImportError("pandas is required for VOCYoloDataset. Install with: pip install pandas")
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        from PIL import Image

        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)
            i = min(i, self.S - 1)
            j = min(j, self.S - 1)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                label_matrix[i, j, 21:25] = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


class Compose:
    """Apply transforms to image only (boxes unchanged)."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes
