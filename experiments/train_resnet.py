import os
import sys
from typing import Tuple, Optional, Dict, List

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

# Add project root to Python path for local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def expand_to_3_channels(x: torch.Tensor) -> torch.Tensor:
    # ResNet expects 3-channel input; MNIST is 1-channel
    return x.expand(3, -1, -1)


def get_mnist_resnet_dataloaders(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 0,  # Use 0 to avoid Windows multiprocessing issues
    max_train_samples: int = 10000,
    max_val_samples: int = 2000,
    max_test_samples: int = 2000,
    val_split: float = 0.1,
    seed: int = 42,
):
    """
    Build Train/Val/Test loaders.
    Val is split from the original MNIST training set.
    """
    assert 0.0 < val_split < 1.0, "val_split must be in (0, 1)"

    common_transforms = transforms.Compose([
        transforms.Resize((224, 224)),            # ResNet input size
        transforms.ToTensor(),
        transforms.Lambda(expand_to_3_channels),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]),
    ])

    full_train = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=common_transforms
    )
    full_test = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=common_transforms
    )

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    # Deterministic split for reproducibility
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Use fixed-size subsets for faster training/evaluation
    train_size = min(len(train_indices), max_train_samples)
    val_size = min(len(val_indices), max_val_samples)
    test_size = min(len(full_test), max_test_samples)

    train_ds = Subset(full_train, train_indices[:train_size].tolist())
    val_ds = Subset(full_train, val_indices[:val_size].tolist())
    test_ds = Subset(full_test, list(range(test_size)))

    print(f"ResNet train subset size: {len(train_ds)}")
    print(f"ResNet val   subset size: {len(val_ds)}")
    print(f"ResNet test  subset size: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # Training mode
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return {"loss": running_loss / total, "acc": correct / total}


def evaluate(model, dataloader, criterion, device):
    model.eval()  # Evaluation mode
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():  # No gradients during evaluation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return {"loss": running_loss / total, "acc": correct / total}


def plot_history(history, save_path):
    # Save loss/accuracy curves for the report
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("ResNet18 Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("ResNet18 Accuracy")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ResNet18 training curves saved to: {save_path}")


def main():
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_root = os.path.join(PROJECT_ROOT, "data")
    model_save_path = os.path.join(PROJECT_ROOT, "models", "resnet18_mnist.pth")
    curves_save_path = os.path.join(PROJECT_ROOT, "experiments", "resnet_training_curves.png")

    # Load Train/Val/Test loaders (Val is used for model selection)
    train_loader, val_loader, test_loader = get_mnist_resnet_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=0,
        max_train_samples=10000,
        max_val_samples=2000,
        max_test_samples=2000,
        val_split=0.1,
        seed=42,
    )

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)  # Replace classifier head for 10 classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)  # Validation evaluation

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        print(f"Train Loss={train_metrics['loss']:.4f}, Acc={train_metrics['acc']:.4f}")
        print(f" Val  Loss={val_metrics['loss']:.4f}, Acc={val_metrics['acc']:.4f}")

        # Save best model based on validation accuracy only
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(model.state_dict(), model_save_path)
            print(f"Best ResNet model updated! Val Acc={best_val_acc:.4f}")

    plot_history(history, curves_save_path)

    # Final test evaluation (not used for model selection)
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test - Loss={test_metrics['loss']:.4f}, Acc={test_metrics['acc']:.4f}")

    print("\nResNet18 training finished.")
    print("Best Val Acc:", best_val_acc)
    print("Best model saved to:", model_save_path)


if __name__ == "__main__":
    main()
