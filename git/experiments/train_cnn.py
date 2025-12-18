import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# Add project root to Python path for local imports
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
    model.train()  # Enable training mode
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

    return {"loss": running_loss / total, "acc": correct / total}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()  # Evaluation mode, no parameter updates
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradients for evaluation
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return {"loss": running_loss / total, "acc": correct / total}


def plot_history(history: Dict[str, List[float]], save_path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

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


def main():
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_save_path = os.path.join(PROJECT_ROOT, "models", "cnn_best.pth")
    curves_save_path = os.path.join(PROJECT_ROOT, "experiments", "cnn_training_curves.png")

    # Load MNIST with explicit train / validation / test split
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_root=os.path.join(PROJECT_ROOT, "data"),
        batch_size=batch_size,
        num_workers=2,
        use_augmentation=True,
        val_split=0.1,
        seed=42
    )

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)  # Use validation set

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        # Save model based on validation accuracy only
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(model.state_dict(), model_save_path)

    plot_history(history, curves_save_path)

    # Final test evaluation (not used for model selection)
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Acc: {test_metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
