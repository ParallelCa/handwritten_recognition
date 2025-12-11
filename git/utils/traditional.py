import numpy as np
import torch
from skimage.feature import hog

# Global HOG parameters (keep them consistent for training and inference)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    block_norm="L2-Hys"
)


def hog_from_numpy(img_np: np.ndarray) -> np.ndarray:
    """
    Compute HOG features from a grayscale numpy image.

    Args:
        img_np: 2D array (H, W), values can be in [0, 1] or [0, 255].

    Returns:
        1D feature array (float32).
    """
    if img_np.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {img_np.shape}")

    # Ensure values in [0, 1]
    if img_np.max() > 1.5:
        img_np = img_np / 255.0

    features = hog(img_np, **HOG_PARAMS)
    return features.astype(np.float32)


def hog_from_tensor(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Compute HOG features from a PyTorch tensor image.

    Args:
        img_tensor: shape (1, H, W) or (H, W), values in [0, 1].

    Returns:
        1D feature array (float32).
    """
    # If tensor has shape (C, H, W), squeeze channel dimension
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.squeeze(0)

    img_np = img_tensor.numpy()
    return hog_from_numpy(img_np)
