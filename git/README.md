# Handwritten Digit Recognition System  
*A Comparative Study of Traditional Machine Learning and Deep Neural Networks on MNIST*

---

## 🧠 Overview

This project implements a complete handwritten digit recognition system using **three different model families**:

1. **HOG + SVM** — Traditional computer vision baseline  
2. **SimpleCNN** — Lightweight custom convolutional neural network  
3. **ResNet18** — Deep residual network adapted for MNIST  

In addition, the project includes:

- A full **Streamlit GUI** for real-time inference  
- Side-by-side comparison between all models  
- Real-time evaluation on MNIST test set  
- Visualizations: probability bars, confusion matrices, training curves  

The goal is to provide a clean, extensible, and research-oriented framework for exploring handwritten digit recognition.

---

## 🎯 Motivation

Handwritten digit recognition has long been a benchmark problem in computer vision and machine learning. While MNIST is simple, comparing **traditional ML** and **modern deep learning** techniques on this dataset reveals:

- Differences in feature extraction  
- Variations in generalization capability  
- Trade-offs in computational cost  
- Sensitivity to noise, stroke variation, and rotation  

This project demonstrates those contrasts through a unified, interactive system.

---

## 🚀 Features

### 🔍 Multi-Model Inference
- Predict digits using **HOG+SVM**, **SimpleCNN**, or **ResNet18**
- Real-time computation of prediction probabilities (CNN/ResNet18)

### 🖼 Interactive GUI (Streamlit)
- Upload digit images or draw using a canvas  
- Real-time preprocessing visualization (28×28 MNIST style)  
- Model comparison table  
- Probability bar charts  
- Dynamic MNIST evaluation (accuracy + confusion matrix)

### 📊 Experimental Framework
- Reproducible training scripts  
- Accuracy benchmarks  
- Automatic saving of weights and training curves  
- Real-time evaluation of each model on MNIST subsets (configurable via GUI)

---

## 📁 Project Structure
handwritten_recognition/
│
├── gui/
│ └── app_streamlit.py
│
├── models/
│ ├── cnn.py
│ ├── cnn_best.pth
│ ├── resnet18_mnist.pth
│ └── traditional_hog_svm.joblib
│
├── utils/
│ ├── preprocess.py
│ ├── datasets.py
│ └── traditional.py
│
├── experiments/
│ ├── train_cnn.py
│ ├── train_resnet.py
│ ├── train_traditional.py
│ └── evaluate_all_models.py
│
│
└── README.md



---

## 🧬 Model Descriptions

### **1. HOG + SVM**
- Extracts Histogram of Oriented Gradients (HOG) features  
- Trains a Support Vector Machine (SVM) classifier  
- Strong baseline for clean digits  
- Limitations: rotation sensitivity, stroke variation  

**Typical Accuracy:** ~96%

---

### **2. SimpleCNN**
A lightweight convolutional model tailored to MNIST:

- 2× Conv → ReLU → MaxPool  
- Fully connected classifier  
- Fast to train and highly accurate  

**Typical Accuracy:** ~99%

---

### **3. ResNet18**
A deeper model adapted for small grayscale inputs:

- Input expanded to 3 channels  
- Final FC layer replaced (10 classes)  
- Trained on upsampled 224×224 images  

**Typical Accuracy:** 96–98%

---

## 🖥 Running the Application

### Install dependencies
```bash
pip install -r requirements.txt


python experiments/train_cnn.py
python experiments/train_resnet.py
python experiments/train_traditional.py


After installation, launch the Streamlit GUI:
streamlit run gui/app_streamlit.py
