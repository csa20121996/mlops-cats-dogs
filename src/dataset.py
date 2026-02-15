from torch.utils.data import Subset
import random
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE


def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess a numpy image to tensor format expected by model.
    """

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    tensor = transform(image)
    return tensor


def get_dataloaders():
    dataset = datasets.ImageFolder(DATA_DIR, transform=get_transforms())

    # Use small subset for faster training
    max_samples = 2000  # change to 500 for even faster
    indices = random.sample(range(len(dataset)), min(max_samples, len(dataset)))
    dataset = Subset(dataset, indices)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader
