from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_dataloaders(
    data_root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for MNIST dataset.

    Args:
        data_root: Folder to store the MNIST data.
        batch_size: Batch size for training and testing.
        num_workers: Number of workers for data loading.
        use_augmentation: Whether to apply data augmentation for training.

    Returns:
        train_loader, test_loader
    """
    # Base transform: convert to tensor and normalize to mean=0.1307, std=0.3081
    # (standard MNIST normalization)
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    return train_loader, test_loader


def preprocess_numpy_to_tensor(
    img_np,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert a numpy grayscale image (H, W) in [0, 1] to a normalized tensor
    compatible with the MNIST-trained models.

    Args:
        img_np: Numpy array with shape (H, W), values in [0, 1].
        device: Optional torch.device.

    Returns:
        tensor: shape (1, 1, H, W) with normalization applied.
    """
    # Convert to tensor
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Apply the same normalization as MNIST
    mean = 0.1307
    std = 0.3081
    tensor = (tensor - mean) / std

    if device is not None:
        tensor = tensor.to(device)

    return tensor
