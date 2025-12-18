import cv2
import numpy as np
from typing import Tuple

def preprocess_cv_image(
    img_bgr: np.ndarray,
    output_size: Tuple[int, int] = (28, 28),
    invert: bool = True,
) -> np.ndarray:
    """
    Preprocess a BGR image to a MNIST-style 28x28 float array in [0,1].

    Steps:
    - Convert to grayscale and denoise
    - Otsu thresholding, optional inversion
    - Crop the foreground region with padding
    - Resize while keeping aspect ratio, then center on a 28x28 canvas
    - Optional dilation to thicken strokes
    - Normalize to [0,1]
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu thresholding
    _, bin_img = cv2.threshold(
        gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # If input is black-on-white, invert to white-on-black
    if invert:
        bin_img = 255 - bin_img

    coords = cv2.findNonZero(bin_img)
    if coords is None:
        return np.zeros(output_size, dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)

    # Small padding to avoid cutting strokes
    pad = int(0.05 * max(w, h))
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, bin_img.shape[1])
    y1 = min(y + h + pad, bin_img.shape[0])
    digit_roi = bin_img[y0:y1, x0:x1]

    target_h, target_w = output_size
    h, w = digit_roi.shape[:2]

    # Resize so the longest side becomes 20 pixels
    max_side = max(h, w)
    scale = 20.0 / max_side
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    digit_resized = cv2.resize(
        digit_roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )

    # Thicken strokes for better visibility
    kernel = np.ones((2, 2), np.uint8)
    digit_resized = cv2.dilate(digit_resized, kernel, iterations=1)

    # Center on a fixed 28x28 canvas
    canvas = np.zeros(output_size, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit_resized

    canvas = canvas.astype(np.float32) / 255.0
    return canvas
