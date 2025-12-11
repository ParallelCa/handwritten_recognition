import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ---------- helper: expand MNIST 1-channel to 3-channel ----------
def expand_to_3_channels(x: torch.Tensor) -> torch.Tensor:
    """
    x: (1, H, W) -> (3, H, W) by repeating the single channel.
    """
    return x.expand(3, -1, -1)
# -----------------------------------------------------------------


def get_mnist_resnet_dataloaders(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 0,  # Windows: use 0 to avoid multiprocessing issues
    max_train_samples: int = 10000,
    max_test_samples: int = 2000,
):
    """
    MNIST -> transformed for ResNet18 (3 x 224 x 224),
    and we only use a subset of the dataset to speed up training.
    """
    common_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),              # (1, 224, 224)
        transforms.Lambda(expand_to_3_channels),  # (3, 224, 224)
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    full_train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=common_transforms
    )
    full_test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=common_transforms
    )

    # ---- use only a subset to speed up training ----
    train_size = min(len(full_train_dataset), max_train_samples)
    test_size = min(len(full_test_dataset), max_test_samples)

    train_dataset = Subset(full_train_dataset, range(train_size))
    test_dataset = Subset(full_test_dataset, range(test_size))

    print(f"ResNet train subset size: {train_size}, test subset size: {test_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
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

    return {
        "loss": running_loss / total,
        "acc": correct / total
    }


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": running_loss / total,
        "acc": correct / total
    }


def plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("ResNet18 Loss")

    # Accuracy
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
    # ======= config (fast version) =======
    batch_size = 64
    num_epochs = 2          # fewer epochs for speed
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_root = os.path.join(PROJECT_ROOT, "data")
    model_save_path = os.path.join(PROJECT_ROOT, "models", "resnet18_mnist.pth")
    curves_save_path = os.path.join(PROJECT_ROOT, "experiments", "resnet_training_curves.png")

    # ======= data =======
    train_loader, test_loader = get_mnist_resnet_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=0,  # keep 0 on Windows
        max_train_samples=10000,
        max_test_samples=2000,
    )

    # ======= model =======
    model = models.resnet18(weights=None)  # random init is enough for this project
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0

    # ======= training loop =======
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        print(f"Train Loss={train_metrics['loss']:.4f}, Acc={train_metrics['acc']:.4f}")
        print(f" Val  Loss={val_metrics['loss']:.4f}, Acc={val_metrics['acc']:.4f}")

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(model.state_dict(), model_save_path)
            print(f"Best ResNet model updated! Val Acc={best_val_acc:.4f}")

    # ======= plot curves =======
    plot_history(history, curves_save_path)

    print("\nResNet18 (FAST) training finished.")
    print("Best Val Acc:", best_val_acc)
    print("Best model saved to:", model_save_path)


if __name__ == "__main__":
    main()
