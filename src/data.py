from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_loaders(cfg: dict):
    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["training"]
    root = Path(ds_cfg["prepared_dir"])
    size = int(ds_cfg["image_size"])
    batch = int(tr_cfg["batch_size"])
    workers = int(ds_cfg.get("num_workers", 4))

    for split in ("train", "val", "test"):
        if not (root / split).exists():
            raise FileNotFoundError(
                f"Missing dataset directory: {root / split}\n"
                "Copy or prepare the dataset before training."
            )

    train_tf = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(size, scale=(0.80, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(root / "test", transform=eval_tf)

    pin = bool(ds_cfg.get("pin_memory", True)) and torch.cuda.is_available()
    persistent = bool(ds_cfg.get("persistent_workers", True)) and workers > 0
    common = dict(num_workers=workers, pin_memory=pin, persistent_workers=persistent)
    if workers > 0:
        common["prefetch_factor"] = int(ds_cfg.get("prefetch_factor", 4))

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, drop_last=False, **common)

    return train_loader, val_loader, test_loader, train_ds.classes
