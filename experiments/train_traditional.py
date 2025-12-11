import os
import sys

import numpy as np
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import torch
from torchvision import datasets, transforms

# Make sure we can import from project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.traditional import hog_from_tensor
from joblib import dump


def build_mnist_hog_features(
    data_root: str,
    max_train_samples: int = 20000,
    max_test_samples: int = 5000
):
    """
    Load MNIST dataset and compute HOG features for a subset of samples.

    Args:
        data_root: folder to store MNIST data.
        max_train_samples: limit number of training samples (for speed).
        max_test_samples: limit number of test samples.

    Returns:
        X_train, y_train, X_test, y_test (all numpy arrays)
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )

    # Limit dataset size for faster training
    train_size = min(len(train_dataset), max_train_samples)
    test_size = min(len(test_dataset), max_test_samples)

    print(f"Using {train_size} training samples, {test_size} test samples for HOG+SVM.")

    # Initialize arrays
    X_train_list = []
    y_train_list = []

    print("Extracting HOG features for training set...")
    for i in tqdm(range(train_size)):
        img_tensor, label = train_dataset[i]  # img_tensor: (1, 28, 28)
        features = hog_from_tensor(img_tensor)
        X_train_list.append(features)
        y_train_list.append(label)

    X_test_list = []
    y_test_list = []

    print("Extracting HOG features for test set...")
    for i in tqdm(range(test_size)):
        img_tensor, label = test_dataset[i]
        features = hog_from_tensor(img_tensor)
        X_test_list.append(features)
        y_test_list.append(label)

    X_train = np.stack(X_train_list, axis=0)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_test = np.stack(X_test_list, axis=0)
    y_test = np.array(y_test_list, dtype=np.int64)

    print("Feature shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    return X_train, y_train, X_test, y_test


def main():
    data_root = os.path.join(PROJECT_ROOT, "data")
    model_save_path = os.path.join(PROJECT_ROOT, "models", "traditional_hog_svm.joblib")

    # 1. Prepare features
    X_train, y_train, X_test, y_test = build_mnist_hog_features(
        data_root=data_root,
        max_train_samples=20000,  # you can increase later if training is fast enough
        max_test_samples=5000
    )

    # 2. Build pipeline: StandardScaler + SVM
    print("\nTraining SVM classifier...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", gamma="scale"))
    ])

    clf.fit(X_train, y_train)

    # 3. Evaluate on test set
    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"HOG + SVM accuracy on test subset: {acc:.4f}")

    # 4. Save model
    dump(clf, model_save_path)
    print(f"Traditional model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
