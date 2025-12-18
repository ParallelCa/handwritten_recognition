
# Handwritten Digit Recognition System

**A Comparative Study of Traditional Machine Learning and Deep Neural Networks on MNIST**

---

## 1. Overview

This project presents a complete handwritten digit recognition system based on the MNIST dataset, implementing and comparing **three representative model families**:

1. **HOG + SVM** — a traditional computer vision and machine learning baseline
2. **SimpleCNN** — a lightweight custom convolutional neural network
3. **ResNet18** — a deep residual network adapted for handwritten digit classification

In addition to model implementation and training, the project provides:

* An interactive **Streamlit-based graphical user interface (GUI)** for real-time inference
* Side-by-side comparison of model predictions
* Deterministic evaluation on subsets of the MNIST test set
* Visualization tools including probability distributions, confusion matrices, and training curves

The system is designed as a **clean, modular, and reproducible experimental framework** for studying handwritten digit recognition using both traditional and deep learning approaches.

---

## 2. Motivation

Handwritten digit recognition is a classical benchmark problem in computer vision and machine learning. Although MNIST is considered a relatively simple dataset, it remains valuable for analyzing and contrasting different modeling paradigms.

By comparing traditional machine learning methods with modern deep neural networks, this project highlights:

* Differences in **feature extraction strategies** (hand-crafted vs. learned features)
* Variations in **generalization performance**
* Trade-offs between **model complexity and computational cost**
* Sensitivity to noise, stroke thickness, rotation, and scale variations

The unified system allows these differences to be observed under consistent preprocessing, evaluation, and visualization settings.

---

## 3. System Features

### 3.1 Multi-Model Inference

* Digit prediction using **HOG + SVM**, **SimpleCNN**, or **ResNet18**
* Probability estimation for CNN-based models (SimpleCNN and ResNet18)

### 3.2 Interactive Graphical User Interface

* Image upload and freehand digit drawing via canvas
* Step-by-step visualization of preprocessing (MNIST-style 28×28 normalization)
* Real-time prediction display
* Model comparison table for identical inputs
* Probability bar charts and confusion matrices

### 3.3 Experimental and Evaluation Framework

* Reproducible training scripts for all models
* Deterministic evaluation on configurable MNIST test subsets
* Automatic saving of trained weights and performance curves
* Offline training visualizations integrated into the GUI

---

## 4. Project Structure

```
handwritten_recognition/
│
├── gui/
│   └── app_streamlit.py
│
├── models/
│   ├── cnn.py
│   ├── cnn_best.pth
│   ├── resnet18_mnist.pth
│   └── traditional_hog_svm.joblib
│
├── utils/
│   ├── preprocess.py
│   ├── datasets.py
│   └── traditional.py
│
├── experiments/
│   ├── train_cnn.py
│   ├── train_resnet.py
│   ├── train_traditional.py
│   └── evaluate_all_models.py
│
└── README.md
```

---

## 5. Model Descriptions

### 5.1 HOG + SVM

This approach represents a classical computer vision pipeline:

* Histogram of Oriented Gradients (HOG) is used for feature extraction
* A Support Vector Machine (SVM) classifier performs digit classification

**Advantages**

* Strong baseline performance on clean, centered digits
* Low training cost and interpretable features

**Limitations**

* Sensitivity to rotation, scale changes, and stroke variation
* Limited robustness compared to deep neural networks

**Typical Accuracy:** approximately 96%

---

### 5.2 SimpleCNN

A lightweight convolutional neural network specifically designed for MNIST:

* Two convolutional blocks (Conv → ReLU → MaxPooling)
* Fully connected classifier with dropout regularization
* End-to-end feature learning from raw pixel inputs

**Characteristics**

* Fast training and inference
* High accuracy with relatively few parameters

**Typical Accuracy:** approximately 99%

---

### 5.3 ResNet18

A deep residual network adapted from standard ImageNet architectures:

* Input expanded from 1 channel to 3 channels
* Images resized to 224×224 to match ResNet input requirements
* Final fully connected layer replaced for 10-class classification

**Characteristics**

* Higher model capacity and deeper representation learning
* Increased computational cost relative to MNIST task complexity

**Typical Accuracy:** approximately 96–98%

---

## 6. Running the System

### 6.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 6.2 Train Models

```bash
python experiments/train_cnn.py
python experiments/train_resnet.py
python experiments/train_traditional.py
```

### 6.3 Launch the Graphical Interface

```bash
streamlit run gui/app_streamlit.py
```
