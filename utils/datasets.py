from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    data_root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    use_augmentation: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return Train/Val/Test dataloaders. Val is split from MNIST train set."""
    assert 0.0 < val_split < 1.0, "val_split must be in (0, 1)"

    # MNIST normalization (used in training and evaluation)
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    # Train transform (optional augmentation)
    train_transform_aug = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        normalize,
    ])

    # Val/Test transform (no augmentation)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = train_transform_aug if use_augmentation else base_transform
    val_transform = base_transform
    test_transform = base_transform

    # Use separate dataset objects to apply different transforms
    full_train_for_train = datasets.MNIST(
        root=data_root, train=True, download=True, transform=train_transform
    )
    full_train_for_val = datasets.MNIST(
        root=data_root, train=True, download=True, transform=val_transform
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=True, transform=test_transform
    )

    n_total = len(full_train_for_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    # Deterministic split for reproducibility
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(full_train_for_train, train_indices.tolist())
    val_dataset = Subset(full_train_for_val, val_indices.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def preprocess_numpy_to_tensor(
    img_np,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Convert (H,W) float image in [0,1] to normalized tensor (1,1,H,W)."""
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    mean = 0.1307
    std = 0.3081
    tensor = (tensor - mean) / std

    if device is not None:
        tensor = tensor.to(device)

    return tensor
