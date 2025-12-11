import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from joblib import load


# --------- project root import hack ----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ---------------------------------------------

from models.cnn import SimpleCNN
from utils.traditional import hog_from_tensor


# --------------- helpers ----------------

def compute_metrics(y_true, y_pred):
    """Return a dict of accuracy, macro recall, macro f1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }


def save_confusion_matrix(cm: np.ndarray, title: str, save_path: str):
    """Plot and save confusion matrix as an image."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to: {save_path}")


# ==================================================
# 1) Evaluate CNN (SimpleCNN)
# ==================================================

def evaluate_cnn(device: torch.device):
    from utils.datasets import get_mnist_dataloaders

    print("\n=== Evaluating SimpleCNN on MNIST test set ===")

    # get data (no augmentation for test)
    _, test_loader = get_mnist_dataloaders(
        data_root=os.path.join(PROJECT_ROOT, "data"),
        batch_size=128,
        num_workers=0,
        use_augmentation=False
    )

    # load model
    model_path = os.path.join(PROJECT_ROOT, "models", "cnn_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find CNN model: {model_path}")

    model = SimpleCNN(num_classes=10)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\nSimpleCNN Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Recall:   {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1']:.4f}")

    save_path = os.path.join(PROJECT_ROOT, "experiments", "cnn_confusion.png")
    save_confusion_matrix(cm, "SimpleCNN Confusion Matrix", save_path)

    return metrics, cm


# ==================================================
# 2) Evaluate traditional model (HOG + SVM)
# ==================================================

def evaluate_traditional(max_test_samples: int = 5000):
    print("\n=== Evaluating HOG + SVM on MNIST test subset ===")

    data_root = os.path.join(PROJECT_ROOT, "data")
    transform = transforms.ToTensor()

    test_dataset_full = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )

    test_size = min(len(test_dataset_full), max_test_samples)
    test_dataset = Subset(test_dataset_full, range(test_size))

    model_path = os.path.join(PROJECT_ROOT, "models", "traditional_hog_svm.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find traditional model: {model_path}")

    clf = load(model_path)

    all_features, all_labels = [], []

    for img_tensor, label in test_dataset:
        feat = hog_from_tensor(img_tensor)
        all_features.append(feat)
        all_labels.append(label)

    X_test = np.stack(all_features, axis=0)
    y_true = np.array(all_labels, dtype=np.int64)
    y_pred = clf.predict(X_test)

    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\nHOG + SVM Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Recall:   {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1']:.4f}")

    save_path = os.path.join(PROJECT_ROOT, "experiments", "hog_svm_confusion.png")
    save_confusion_matrix(cm, "HOG + SVM Confusion Matrix", save_path)

    return metrics, cm


# ==================================================
# 3) Evaluate ResNet18
# ==================================================

def expand_to_3_channels(x: torch.Tensor) -> torch.Tensor:
    return x.expand(3, -1, -1)


def evaluate_resnet(device: torch.device, max_test_samples: int = 2000):
    print("\n=== Evaluating ResNet18 on MNIST test subset ===")

    data_root = os.path.join(PROJECT_ROOT, "data")

    common_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(expand_to_3_channels),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    test_dataset_full = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=common_transforms
    )

    test_size = min(len(test_dataset_full), max_test_samples)
    test_dataset = Subset(test_dataset_full, range(test_size))

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model_path = os.path.join(PROJECT_ROOT, "models", "resnet18_mnist.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find ResNet18 model: {model_path}")

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 10)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\nResNet18 Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Recall:   {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1']:.4f}")

    save_path = os.path.join(PROJECT_ROOT, "experiments", "resnet_confusion.png")
    save_confusion_matrix(cm, "ResNet18 Confusion Matrix", save_path)

    return metrics, cm


# ==================================================
# main
# ==================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    cnn_metrics, _ = evaluate_cnn(device)
    trad_metrics, _ = evaluate_traditional(max_test_samples=5000)
    resnet_metrics, _ = evaluate_resnet(device, max_test_samples=2000)

    print("\n=== Summary of Evaluation Metrics ===")
    print(f"SimpleCNN : Acc={cnn_metrics['accuracy']:.4f}, Rec={cnn_metrics['recall']:.4f}, F1={cnn_metrics['f1']:.4f}")
    print(f"HOG+SVM   : Acc={trad_metrics['accuracy']:.4f}, Rec={trad_metrics['recall']:.4f}, F1={trad_metrics['f1']:.4f}")
    print(f"ResNet18  : Acc={resnet_metrics['accuracy']:.4f}, Rec={resnet_metrics['recall']:.4f}, F1={resnet_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
