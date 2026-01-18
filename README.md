# 🌍 EarthGC-SegNet: Lightweight Remote Sensing Segmentation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)  
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

This repository implements **EarthGC-SegNet**, a **lightweight semantic segmentation framework** for  
**high-resolution remote sensing imagery**.

The model is designed to balance:

- 🎯 High segmentation accuracy  
- ⚡ Fast inference on CPU/GPU  
- 🧠 Interpretability using Grad-CAM  
- 📦 Low parameter count for deployment  

It is evaluated on the **EarthVQA dataset** and achieves:

- **65.17% mIoU**
- **78.26% mDice**
- Near real-time GPU inference  
- CPU-compatible deployment

---

## 🧠 Core Features

- ConvNeXt-Tiny encoder  
- Global Context (GC) modeling  
- Multi-Head Self-Attention (MHSA)  
- Multi-scale decoder  
- Boundary-aware learning  
- Composite loss: CE + Dice + Lovász + Boundary  
- Grad-CAM interpretability  
- Monte Carlo Dropout for uncertainty  

---

## 🏗️ Architecture Summary

Pipeline:

1. Input Image  
2. ConvNeXt-Tiny Encoder (4 stages)  
3. Global Context Module  
4. Multi-Head Self-Attention  
5. Multi-Scale Decoder  
6. Detail Branch for boundaries  
7. Segmentation Head  
8. Boundary Head (optional)  
9. Grad-CAM Visualization  

---

## 📊 Dataset

- Dataset: **[EarthVQA (Segmentation Subset)](https://www.kaggle.com/datasets/jawadulkarim117/loveda)**
- Classes:
  - Background  
  - Building  
  - Road  
  - Water  
  - Barren  
  - Forest  
  - Agriculture  

- Augmentation:
  - Rotation (90°, 180°, 270°)  
  - Color jitter  
  - Gaussian noise  
  - Cut-join content mixing  

---

## 🔧 Training Strategy

- Loss:
  - Cross-Entropy  
  - Dice  
  - Lovász-Softmax  
  - Boundary Loss  
  - Auxiliary Supervision  

- Optimizer: AdamW  
- Scheduler: Cosine Annealing  
- Mixed Precision Training  
- Early Stopping  

---

## 📈 Evaluation Metrics

- mIoU  
- Dice  
- Pixel Accuracy  
- Boundary F1  
- Boundary IoU  
- Cohen’s Kappa  
- Expected Calibration Error (ECE)  
- Parameters & GFLOPs  
- Inference Time  

---

## 🧪 Ablation Components

- Global Context Block  
- MHSA  
- Boundary Supervision  
- Lovász Loss  

Each contributes to accuracy, boundary quality, and calibration.

---

## 🔍 Interpretability

- Class-wise Grad-CAM visualizations  
- Shows which regions drive predictions  
- Supports explainable remote sensing decisions  

---

## ⚡ Inference Speed

| Platform | Time / Image |
|--------|--------------|
| Ryzen 5 CPU | ~880 ms |
| Xeon CPU | ~378 ms |
| Tesla P100 GPU | ~7 ms |
| T4 GPU | ~7.5 ms |

---

## 🌍 Applications

- Urban mapping  
- Land-use analysis  
- Environmental monitoring  
- Agriculture assessment  
- Disaster analysis  
- Edge & cloud deployment  

---
