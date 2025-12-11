import cv2
import numpy as np
from typing import Tuple

def preprocess_cv_image(
    img_bgr: np.ndarray,
    output_size: Tuple[int, int] = (28, 28),
    invert: bool = True,
) -> np.ndarray:
    """
    将任意 BGR 图像预处理成 MNIST 风格的 28x28 黑底白字浮点数组 [0,1].

    步骤：
    - BGR -> 灰度
    - 轻微高斯模糊去噪
    - Otsu 二值化（得到前景 / 背景）
    - 根据 invert 决定黑底白字 / 白底黑字
    - 找前景的外接矩形并裁剪，留少量 padding
    - 按比例缩放到约 20x20，并居中放在 28x28 画布上（和 MNIST 风格一致）
    - 适当膨胀一下笔画，让太细的线条变粗一点
    - 归一化到 [0,1]
    """
    # 1. BGR -> 灰度
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. 轻微高斯模糊，降低噪声但不过度模糊
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Otsu 二值化
    #    先得到黑字白底的二值图，然后根据 invert 决定是否反色
    _, bin_img = cv2.threshold(
        gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 上传的图片一般是白底黑字；若希望最终是“黑底白字”，则需要 invert=True
    if invert:
        # 黑底白字
        bin_img = 255 - bin_img

    # 4. 找前景（非零）区域
    coords = cv2.findNonZero(bin_img)
    if coords is None:
        # 没有检测到前景，直接返回全黑
        return np.zeros(output_size, dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)

    # 加一点 padding，避免裁得太紧
    pad = int(0.05 * max(w, h))  # 5% padding
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, bin_img.shape[1])
    y1 = min(y + h + pad, bin_img.shape[0])
    digit_roi = bin_img[y0:y1, x0:x1]

    # 5. 按比例缩到大约 20x20（MNIST 中数字占 28x28 的约 2/3~3/4）
    target_h, target_w = output_size
    h, w = digit_roi.shape[:2]
    max_side = max(h, w)
    # 这里 20 可以根据你训练时的风格微调
    scale = 20.0 / max_side
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    digit_resized = cv2.resize(
        digit_roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )

    # 6. 对细线做一次膨胀，让线条更粗、更接近 MNIST
    kernel = np.ones((2, 2), np.uint8)
    digit_resized = cv2.dilate(digit_resized, kernel, iterations=1)

    # 7. 把缩放后的数字居中贴到 28x28 的黑底画布
    canvas = np.zeros(output_size, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit_resized

    # 8. 归一化到 [0,1]
    canvas = canvas.astype(np.float32) / 255.0

    return canvas
