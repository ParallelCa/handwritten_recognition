import os
import sys

import numpy as np
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, learning_curve

import torch
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# Add project root to Python path for local imports
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
    Load MNIST and extract HOG features for a limited number of samples.
    Returns numpy arrays for training and testing.
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

    # Use subsets for faster feature extraction and training
    train_size = min(len(train_dataset), max_train_samples)
    test_size = min(len(test_dataset), max_test_samples)

    print(f"Using {train_size} training samples, {test_size} test samples for HOG+SVM.")

    X_train_list = []
    y_train_list = []

    print("Extracting HOG features for training set...")
    for i in tqdm(range(train_size)):
        img_tensor, label = train_dataset[i]  # img_tensor shape: (1, 28, 28)
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


def build_svm_pipeline() -> Pipeline:
    """
    HOG + SVM pipeline: feature scaling + RBF SVM classifier.
    """
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", gamma="scale"))
    ])
    return clf


def train_and_evaluate_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_save_path: str,
):
    """
    Train one HOG+SVM model on the training set and evaluate on the test set.
    Saves the fitted pipeline for later use.
    """
    print("\n=== Baseline: Train HOG + SVM classifier on training set ===")
    clf = build_svm_pipeline()
    clf.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print("\nBaseline HOG + SVM metrics on test subset:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  Confusion matrix shape: {cm.shape}")

    # Save the trained pipeline as a single file
    dump(clf, model_save_path)
    print(f"Traditional model saved to: {model_save_path}")

    # Save confusion matrix figure for the report
    cm_fig_path = os.path.join(PROJECT_ROOT, "experiments", "hog_svm_confusion.png")
    plot_confusion_matrix(cm, cm_fig_path)

    return acc, rec, f1, cm


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    """
    Plot and save the confusion matrix.
    """
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title("HOG + SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def run_kfold_cross_validation_hog_svm(
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
):
    """
    Compute a learning curve using stratified k-fold cross validation.
    The curve shows accuracy vs. number of training samples.
    """
    print(f"\n=== Running {k_folds}-fold Cross Validation for HOG + SVM ===")

    clf = build_svm_pipeline()

    train_sizes, train_scores, val_scores = learning_curve(
        clf,
        X,
        y,
        cv=StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42),
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )

    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    val_scores_mean = val_scores.mean(axis=1)
    val_scores_std = val_scores.std(axis=1)

    print("\nTrain sizes:", train_sizes)
    print("CV mean accuracies (val):", [f"{s:.4f}" for s in val_scores_mean])

    curve_path = os.path.join(PROJECT_ROOT, "experiments", "hog_svm_learning_curve.png")
    plot_learning_curve(
        train_sizes,
        train_scores_mean,
        train_scores_std,
        val_scores_mean,
        val_scores_std,
        save_path=curve_path,
        title=f"HOG + SVM Learning Curve (k-fold={k_folds})"
    )


def plot_learning_curve(
    train_sizes,
    train_scores_mean,
    train_scores_std,
    val_scores_mean,
    val_scores_std,
    save_path: str,
    title: str = "Learning Curve",
):
    """
    Plot training and cross-validation accuracy with standard deviation bands.
    """
    plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        label="Train std",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", label="Train Accuracy")

    plt.fill_between(
        train_sizes,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.2,
        label="CV std",
    )
    plt.plot(train_sizes, val_scores_mean, "o-", label="CV Accuracy")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Learning curve saved to: {save_path}")


def run_bootstrap_hog_svm(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_bootstrap: int = 5,
):
    """
    Bootstrap evaluation:
    train on bootstrap samples and evaluate on a fixed test set.
    """
    print(f"\n=== Bootstrap evaluation for HOG + SVM (B={num_bootstrap}) ===")
    N = X.shape[0]
    bootstrap_accs = []

    for b in range(1, num_bootstrap + 1):
        print(f"\n--- Bootstrap sample {b}/{num_bootstrap} ---")

        sample_indices = np.random.choice(N, size=N, replace=True)
        X_boot = X[sample_indices]
        y_boot = y[sample_indices]

        clf = build_svm_pipeline()
        clf.fit(X_boot, y_boot)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bootstrap_accs.append(acc)

        print(f"  Test Accuracy for bootstrap model {b}: {acc:.4f}")

    mean_acc = np.mean(bootstrap_accs)
    std_acc = np.std(bootstrap_accs)
    print("\n=== Bootstrap summary (HOG + SVM) ===")
    print("Acc per bootstrap model:", [f"{a:.4f}" for a in bootstrap_accs])
    print(f"Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}")


def main():
    data_root = os.path.join(PROJECT_ROOT, "data")
    model_save_path = os.path.join(PROJECT_ROOT, "models", "traditional_hog_svm.joblib")

    # Extract HOG features from MNIST subsets
    X_train, y_train, X_test, y_test = build_mnist_hog_features(
        data_root=data_root,
        max_train_samples=20000,
        max_test_samples=5000
    )

    # Baseline training and evaluation on the test subset
    train_and_evaluate_baseline(
        X_train, y_train, X_test, y_test, model_save_path
    )

    # Learning curve using stratified k-fold cross validation
    run_kfold_cross_validation_hog_svm(
        X=X_train,
        y=y_train,
        k_folds=5,
    )

    # Bootstrap evaluation on a fixed test subset
    run_bootstrap_hog_svm(
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test,
        num_bootstrap=5,
    )


if __name__ == "__main__":
    main()
