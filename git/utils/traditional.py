import numpy as np
import torch
from skimage.feature import hog

# Global HOG parameters (shared by training and inference)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    block_norm="L2-Hys"
)


def hog_from_numpy(img_np: np.ndarray) -> np.ndarray:
    """Compute HOG features from a 2D grayscale numpy image."""
    if img_np.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {img_np.shape}")

    # Normalize to [0,1] if input is in [0,255]
    if img_np.max() > 1.5:
        img_np = img_np / 255.0

    features = hog(img_np, **HOG_PARAMS)
    return features.astype(np.float32)


def hog_from_tensor(img_tensor: torch.Tensor) -> np.ndarray:
    """Compute HOG features from a PyTorch tensor image."""
    # Accept (1,H,W) or (H,W)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.squeeze(0)

    img_np = img_tensor.numpy()
    return hog_from_numpy(img_np)
