import os
import random
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess a numpy image to tensor format expected by the model."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    return transform(image)


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


def _split_indices(n: int, seed: int) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)

    train_size = int(0.8 * n)
    val_size = int(0.1 * n)

    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size + val_size]
    test_idx = idx[train_size + val_size:]
    return train_idx, val_idx, test_idx


def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads dataset from DATA_DIR and returns train/val/test loaders.

    Split:
      - 80/10/10 (train/val/test), deterministic via SEED env var.

    Optional:
      - MAX_SAMPLES env var to subsample for faster local runs.
    """
    seed = int(os.getenv("SEED", "42"))

    base = datasets.ImageFolder(DATA_DIR)
    n = len(base)
    indices = list(range(n))

    # Optional subsample for faster local runs
    max_samples_env = os.getenv("MAX_SAMPLES", "").strip()
    if max_samples_env:
        max_samples = min(int(max_samples_env), n)
        rnd = random.Random(seed)
        indices = rnd.sample(indices, max_samples)

    train_idx, val_idx, test_idx = _split_indices(len(indices), seed)

    # Map split indices into the (possibly subsampled) index list
    train_idx = [indices[i] for i in train_idx]
    val_idx = [indices[i] for i in val_idx]
    test_idx = [indices[i] for i in test_idx]

    train_ds = datasets.ImageFolder(DATA_DIR, transform=build_transforms(train=True))
    val_ds = datasets.ImageFolder(DATA_DIR, transform=build_transforms(train=False))
    test_ds = datasets.ImageFolder(DATA_DIR, transform=build_transforms(train=False))

    train_set = Subset(train_ds, train_idx)
    val_set = Subset(val_ds, val_idx)
    test_set = Subset(test_ds, test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader