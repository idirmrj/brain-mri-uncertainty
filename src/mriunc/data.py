"""
data.py - dataset, transforms and dataloaders for the brain MRI classifier.

Decisions baked in (from 01_explore):
  - Images resized to 224x224 (ImageNet-pretrained encoder standard).
  - Every image converted to RGB: the dataset mixes grayscale ('L') and RGB,
    and a pretrained encoder expects 3 channels. convert("RGB") fixes both.
  - Dataset is balanced (1400/class), so NO class weighting is needed.
  - A validation split is carved OUT OF Training (the official Testing set is
    kept untouched as the final hold-out).
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet normalization stats (the pretrained encoder was trained with these)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def _loader_rgb(path):
    """Force every image to RGB (handles the mixed L / RGB modes)."""
    from PIL import Image
    return Image.open(path).convert("RGB")


def build_transforms():
    """Train transforms include light augmentation; eval transforms don't.

    Augmentation is deliberately gentle for MRI: horizontal flip is fine, but we
    avoid vertical flips / heavy rotations that would create anatomically
    implausible brains. Small rotations + affine mimic real acquisition variance.
    """
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_dataloaders(data_dir, batch_size=32, val_split=0.15,
                      num_workers=2, seed=42):
    """Return (train_loader, val_loader, test_loader, class_names).

    - train/val come from data/Training (val carved out with a fixed seed)
    - test comes from data/Testing (the official hold-out, eval transforms)
    """
    data_dir = Path(data_dir)
    train_tf, eval_tf = build_transforms()

    # Two views of the SAME Training folder: one with augmentation (train),
    # one without (val). We then split indices so val never sees augmentation.
    train_full = datasets.ImageFolder(data_dir / "Training", transform=train_tf,
                                      loader=_loader_rgb)
    val_full = datasets.ImageFolder(data_dir / "Training", transform=eval_tf,
                                    loader=_loader_rgb)

    n_total = len(train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = torch.utils.data.Subset(train_full, train_idx)
    val_ds = torch.utils.data.Subset(val_full, val_idx)
    test_ds = datasets.ImageFolder(data_dir / "Testing", transform=eval_tf,
                                   loader=_loader_rgb)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_full.classes


if __name__ == "__main__":
    # quick self-check: run `python src/mriunc/data.py <path-to-data>`
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    tr, va, te, classes = build_dataloaders(root, batch_size=8, num_workers=0)
    print("Classes:", classes)
    print("Batches - train:", len(tr), "val:", len(va), "test:", len(te))
    xb, yb = next(iter(tr))
    print("Batch tensor:", xb.shape, "| labels:", yb.tolist())
    print("Value range:", round(xb.min().item(), 2), "to", round(xb.max().item(), 2),
          "(normalized, so negatives are expected)")
