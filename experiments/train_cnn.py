import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# Make sure we can import from project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.datasets import get_mnist_dataloaders
from models.cnn import SimpleCNN


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return {"loss": epoch_loss, "acc": epoch_acc}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return {"loss": epoch_loss, "acc": epoch_acc}


def plot_history(history: Dict[str, List[float]], save_path: str) -> None:
    """
    Plot training and validation loss/accuracy curves and save to file.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def main():
    # =========================
    # 1. Basic configuration
    # =========================
    batch_size = 64
    num_epochs = 2
    learning_rate = 1e-3

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output paths
    model_save_path = os.path.join(PROJECT_ROOT, "models", "cnn_best.pth")
    curves_save_path = os.path.join(PROJECT_ROOT, "experiments", "cnn_training_curves.png")

    # =========================
    # 2. Data loaders
    # =========================
    train_loader, test_loader = get_mnist_dataloaders(
        data_root=os.path.join(PROJECT_ROOT, "data"),
        batch_size=batch_size,
        num_workers=2,
        use_augmentation=True
    )

    # =========================
    # 3. Model, loss, optimizer
    # =========================
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    # =========================
    # 4. Training loop
    # =========================
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.4f}")
        print(f" Val  Loss: {val_metrics['loss']:.4f},  Val  Acc: {val_metrics['acc']:.4f}")

        # Save best model
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with val_acc = {best_val_acc:.4f}")

    # =========================
    # 5. Plot training curves
    # =========================
    plot_history(history, curves_save_path)

    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {model_save_path}")


if __name__ == "__main__":
    main()

