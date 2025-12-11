import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models, datasets, transforms

import streamlit as st
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

import cv2  # 用于图像预处理

from joblib import load

# 可选：画布输入
try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False

# -------------- project path --------------
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
# -----------------------------------------

from models.cnn import SimpleCNN
from utils.datasets import preprocess_numpy_to_tensor
from utils.traditional import hog_from_numpy


# ---------------- Metric helpers ----------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算 Accuracy、宏平均 Recall 和宏平均 F1-score。
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }


def plot_cm(cm: np.ndarray, title: str):
    """
    画混淆矩阵，并返回 matplotlib Figure，方便在 Streamlit 中显示。
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    return fig


# ---------------- Utility functions ----------------

def pil_to_cv2_bgr(img_pil: Image.Image) -> np.ndarray:
    """PIL -> OpenCV BGR"""
    img_rgb = np.array(img_pil.convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1]
    return img_bgr


def preprocess_for_resnet_from_processed28(img28: np.ndarray, device: torch.device):
    """
    已经是 28×28 单通道的 float32 [0,1] 图像，
    转成 3×224×224 的张量，并做与 ResNet 训练时一致的归一化。
    """
    tensor = torch.from_numpy(img28).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    tensor = tensor.to(device=device, dtype=torch.float32)

    tensor = F.interpolate(
        tensor, size=(224, 224), mode="bilinear", align_corners=False
    )  # (1,1,224,224)
    tensor = tensor.expand(-1, 3, -1, -1)  # (1,3,224,224)

    tensor = (tensor - 0.5) / 0.5
    return tensor


def topk_from_probs(probs: np.ndarray, k: int = 3):
    """
    给定概率向量，返回 top-k 的类别及其概率。
    """
    k = min(k, probs.shape[0])
    idx = np.argsort(probs)[-k:][::-1]
    return idx, probs[idx]


# ---------------- 预处理调试：返回每一步 ----------------

def debug_preprocess_steps(
    img_bgr: np.ndarray,
    output_size: Tuple[int, int] = (28, 28),
    invert: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    返回 processed28（float 28×28）和所有中间步骤（uint8 图像）
    用于 GUI 按「Grayscale → Binarization → Unified size → Normalization」展示。
    """

    # Step 1: Grayscale conversion
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 轻微高斯模糊去噪
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 2: Binarization（Otsu + 反二值化，得到黑底白字）
    # 对于“上传的白底黑字”图片：使用 THRESH_BINARY_INV + OTSU -> 黑底白字
    _, thresh = cv2.threshold(
        gray_blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 如果是画布（已经是黑底白字），可以选择不反，交给 invert 参数控制
    if not invert:
        thresh = 255 - thresh

    # 找非零区域（数字区域）
    coords = cv2.findNonZero(thresh)
    if coords is None:
        blank = np.zeros(output_size, dtype=np.float32)
        steps = {
            "gray": gray,
            "thresh": thresh,
            "canvas": np.zeros(output_size, dtype=np.uint8),
        }
        return blank, steps

    x, y, w, h = cv2.boundingRect(coords)

    # padded 裁剪，防止裁得过紧
    pad = int(0.05 * max(w, h))
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, thresh.shape[1])
    y1 = min(y + h + pad, thresh.shape[0])
    digit_roi = thresh[y0:y1, x0:x1]
    roi_h, roi_w = digit_roi.shape

    # Step 3: Unified image size（裁剪、缩放、居中到 28×28）
    target_h, target_w = output_size
    scale = min(target_w / roi_w, target_h / roi_h)
    new_w = int(roi_w * scale)
    new_h = int(roi_h * scale)

    digit_resized = cv2.resize(
        digit_roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )

    canvas = np.zeros(output_size, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit_resized

    # Step 4: Normalization（0–255 -> 0–1 float）
    processed28 = canvas.astype(np.float32) / 255.0

    steps = {
        "gray": gray,          # Step 1
        "thresh": thresh,      # Step 2
        "canvas": canvas,      # Step 3（统一尺寸）
    }
    return processed28, steps


# ---------------- Model loading (cached) ----------------

@st.cache_resource
def load_cnn_model(device: torch.device):
    model_path = BASE_DIR / "models" / "cnn_best.pth"
    model = SimpleCNN(num_classes=10)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_resnet_model(device: torch.device):
    model_path = BASE_DIR / "models" / "resnet18_mnist.pth"
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 10)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_traditional_model():
    model_path = BASE_DIR / "models" / "traditional_hog_svm.joblib"
    clf = load(model_path)
    return clf


# ---------------- Single image prediction ----------------

def predict_cnn(processed28: np.ndarray, device: torch.device):
    """
    processed28: float32 (28,28), [0,1]
    """
    model = load_cnn_model(device)
    tensor = preprocess_numpy_to_tensor(processed28, device=device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs


def predict_resnet(processed28: np.ndarray, device: torch.device):
    """
    processed28: float32 (28,28), [0,1]
    """
    model = load_resnet_model(device)
    tensor = preprocess_for_resnet_from_processed28(processed28, device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs


def predict_traditional(processed28: np.ndarray):
    """
    processed28: float32 (28,28), [0,1]
    """
    clf = load_traditional_model()
    feat = hog_from_numpy(processed28).reshape(1, -1)
    pred = clf.predict(feat)[0]
    return int(pred)

# ---------------- Real-time MNIST evaluation ----------------

@st.cache_data(show_spinner=True)
def evaluate_cnn(test_samples: int, device_str: str):
    """
    返回：metrics(dict: accuracy/recall/f1) + confusion matrix
    """
    device = torch.device(device_str)
    data_root = BASE_DIR / "data"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    ds = Subset(full_test, range(min(test_samples, len(full_test))))
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    model = load_cnn_model(device)
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, p = logits.max(1)
            preds.append(p.cpu().numpy())
            labels.append(y.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)

    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm


@st.cache_data(show_spinner=True)
def evaluate_traditional(test_samples: int):
    """
    HOG+SVM：返回 metrics + confusion matrix
    """
    data_root = BASE_DIR / "data"
    transform = transforms.ToTensor()
    full_test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    ds = Subset(full_test, range(min(test_samples, len(full_test))))
    clf = load_traditional_model()

    feats, labels = [], []
    for img, y in ds:
        feat = hog_from_numpy(img.squeeze(0).numpy())
        feats.append(feat)
        labels.append(y)

    X = np.stack(feats)
    y_true = np.array(labels)
    y_pred = clf.predict(X)

    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm


@st.cache_data(show_spinner=True)
def evaluate_resnet(test_samples: int, device_str: str):
    """
    ResNet18：返回 metrics + confusion matrix
    """
    device = torch.device(device_str)
    data_root = BASE_DIR / "data"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    full_test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    ds = Subset(full_test, range(min(test_samples, len(full_test))))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model = load_resnet_model(device)
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, p = logits.max(1)
            preds.append(p.cpu().numpy())
            labels.append(y.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)

    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm


# ---------------- Streamlit UI ----------------

def main():
    st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")

    st.title("Handwritten Digit Recognition Demo")
    st.write(
        "This interface demonstrates three models:\n"
        "- HOG + SVM (traditional baseline)\n"
        "- SimpleCNN\n"
        "- ResNet18\n"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    st.sidebar.write(f"**Device:** {device}")

    # ------------- Model selection -------------
    st.sidebar.subheader("Model selection")
    model_name = st.sidebar.selectbox("Choose model", ["SimpleCNN", "ResNet18", "HOG + SVM"])
    show_probs = st.sidebar.checkbox("Show probability bar chart (CNN / ResNet)", True)

    st.sidebar.subheader("Evaluation settings (real-time)")
    cnn_n = st.sidebar.slider("CNN test samples", 1000, 10000, 2000, 1000)
    trad_n = st.sidebar.slider("HOG+SVM test samples", 1000, 5000, 2000, 1000)
    resnet_n = st.sidebar.slider("ResNet test samples", 500, 2000, 1000, 500)

    # ------------- Input area -------------
    col1, col2 = st.columns(2)

    original_pil: Optional[Image.Image] = None
    mode = "Upload image"

    with col1:
        st.subheader("1. Input image")

        mode = st.radio("Input type", ["Upload image"] + (["Draw digit (canvas)"] if HAS_CANVAS else []))

        if mode == "Upload image":
            f = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
            if f:
                original_pil = Image.open(f)
                st.image(original_pil, caption="Original image", width=280)

        elif mode == "Draw digit (canvas)":
            canvas = st_canvas(
                fill_color="rgba(0, 0, 0, 1)",
                stroke_width=10,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            if canvas.image_data is not None:
                img = Image.fromarray(canvas.image_data.astype(np.uint8)).convert("RGB")
                original_pil = img
                st.image(img, caption="Canvas image", width=280)

    processed28: Optional[np.ndarray] = None

    with col2:
        st.subheader("2. Preprocessing steps")

        if original_pil is None:
            st.info("Upload or draw an image.")
        else:
            img_bgr = pil_to_cv2_bgr(original_pil)

            # 上传图片通常是白底黑字，需要 invert=True；
            # 画布已经是黑底白字，invert=False。
            invert_flag = True if mode == "Upload image" else False

            processed28, steps = debug_preprocess_steps(
                img_bgr,
                output_size=(28, 28),
                invert=invert_flag
            )

            # ---- Step 1: Grayscale conversion ----
            st.markdown("### Step 1: Grayscale conversion")
            st.write("Convert the input image from color (BGR) to a single-channel grayscale image.")
            st.image(steps["gray"], caption="Grayscale image", width=200)

            # ---- Step 2: Binarization ----
            st.markdown("### Step 2: Binarization")
            st.write("Apply Otsu's thresholding to obtain a clear black-and-white digit (black background, white digit).")
            st.image(steps["thresh"], caption="Binary image", width=200)

            # ---- Step 3: Unified image size ----
            st.markdown("### Step 3: Unified image size (28×28)")
            st.write("Resize the digit while keeping aspect ratio, then center it on a fixed 28×28 canvas.")
            st.image(steps["canvas"], caption="Centered 28×28 canvas", width=200)

            # ---- Step 4: Normalization ----
            st.markdown("### Step 4: Normalization")
            st.write("Normalize pixel values from [0, 255] to [0, 1] as the final model input.")
            st.image(processed28, caption="Normalized 28×28 (float32, [0,1])", width=200)

            # ------------- Prediction -------------
            st.markdown("---")
            st.subheader("3. Prediction")

            if st.button("Predict"):
                if model_name == "SimpleCNN":
                    pred, probs = predict_cnn(processed28, device)
                    st.success(f"SimpleCNN Prediction: {pred}")
                    top_idx, top_probs = topk_from_probs(probs)
                    st.table({"Class": top_idx, "Prob": top_probs})
                    if show_probs:
                        st.bar_chart(probs)

                elif model_name == "ResNet18":
                    pred, probs = predict_resnet(processed28, device)
                    st.success(f"ResNet18 Prediction: {pred}")
                    top_idx, top_probs = topk_from_probs(probs)
                    st.table({"Class": top_idx, "Prob": top_probs})
                    if show_probs:
                        st.bar_chart(probs)

                else:
                    pred = predict_traditional(processed28)
                    st.success(f"HOG + SVM Prediction: {pred}")
                    st.info("No probability output for SVM.")

    # ---------------- Model comparison ----------------
    st.markdown("---")
    st.header("Model comparison (same input)")

    if original_pil is not None:
        if st.button("Compare all models"):
            img_bgr = pil_to_cv2_bgr(original_pil)
            invert_flag = True if mode == "Upload image" else False
            processed28, _ = debug_preprocess_steps(
                img_bgr,
                output_size=(28, 28),
                invert=invert_flag
            )

            p1, _ = predict_cnn(processed28, device)
            p2, _ = predict_resnet(processed28, device)
            p3 = predict_traditional(processed28)

            st.table({
                "Model": ["SimpleCNN", "ResNet18", "HOG+SVM"],
                "Prediction": [p1, p2, p3]
            })
    else:
        st.info("Upload or draw an image to compare models.")

    # ---------------- Real-time evaluation ----------------
    st.markdown("---")
    st.header("Real-time MNIST evaluation")

    if st.button("Run Evaluation"):
        with st.spinner("Evaluating models..."):
            cnn_metrics, cm_cnn = evaluate_cnn(cnn_n, device_str)
            trad_metrics, cm_trad = evaluate_traditional(trad_n)
            resnet_metrics, cm_resnet = evaluate_resnet(resnet_n, device_str)

        st.subheader("Evaluation Metrics (MNIST test subsets)")

        st.table({
            "Model": ["SimpleCNN", "HOG+SVM", "ResNet18"],
            "Samples": [cnn_n, trad_n, resnet_n],
            "Accuracy": [
                cnn_metrics["accuracy"],
                trad_metrics["accuracy"],
                resnet_metrics["accuracy"],
            ],
            "Recall": [
                cnn_metrics["recall"],
                trad_metrics["recall"],
                resnet_metrics["recall"],
            ],
            "F1-score": [
                cnn_metrics["f1"],
                trad_metrics["f1"],
                resnet_metrics["f1"],
            ],
        })

        st.subheader("Confusion Matrices")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(plot_cm(cm_cnn, "SimpleCNN"))
        with c2:
            st.pyplot(plot_cm(cm_trad, "HOG + SVM"))
        with c3:
            st.pyplot(plot_cm(cm_resnet, "ResNet18"))
        # ---------------- Performance Curves ----------------
        st.markdown("---")
        st.header("Performance Curves")

        st.write(
            "Here we show pre-computed training and validation curves as well as "
            "learning curves for the three models. These figures are generated "
            "offline in the training scripts and loaded here for visualization."
        )

        curves_dir = BASE_DIR / "experiments"

        # 一行三列：CNN / HOG+SVM / ResNet
        col_cnn, col_trad, col_res = st.columns(3)

        # ----- SimpleCNN 曲线 -----
        with col_cnn:
            st.subheader("SimpleCNN")
            cnn_curve_path = curves_dir / "cnn_training_curves.png"
            if cnn_curve_path.exists():
                st.image(str(cnn_curve_path), caption="Training & Validation Curves", use_container_width=True)
            else:
                st.info("cnn_training_curves.png not found in experiments/")

        # ----- HOG+SVM 曲线（learning curve） -----
        with col_trad:
            st.subheader("HOG + SVM")
            hog_curve_path = curves_dir / "hog_svm_learning_curve.png"
            if hog_curve_path.exists():
                st.image(str(hog_curve_path), caption="Learning Curve (train size vs CV accuracy)", use_container_width=True)
            else:
                st.info("hog_svm_learning_curve.png not found in experiments/")

        # ----- ResNet18 曲线 -----
        with col_res:
            st.subheader("ResNet18")
            resnet_curve_path = curves_dir / "resnet_training_curves.png"
            if resnet_curve_path.exists():
                st.image(str(resnet_curve_path), caption="Training & Validation Curves", use_container_width=True)
            else:
                st.info("resnet_training_curves.png not found in experiments/")

if __name__ == "__main__":
    main()
