# Multimodal Lung Cancer Detection (3D CT + Structured Fusion)

A modular deep learning pipeline for lung cancer classification using volumetric CT scans and radiologist-annotated structured features from the LIDC-IDRI dataset.

This project implements a multimodal architecture combining:

- 3D CNN for volumetric CT modeling
- MLP for structured radiologist semantic features
- Late fusion for combined prediction
- Rigorous evaluation with leakage analysis

Designed with clean engineering practices, reproducibility, and experimental transparency in mind.

---

## 🔍 Problem Overview

Lung cancer detection from CT scans is challenging due to:

- Weak supervision
- Limited dataset size
- High class imbalance
- Variability in radiologist interpretation

This project investigates whether combining structured expert annotations with volumetric CT modeling improves predictive performance.

---

## 🏗 Architecture

The system consists of two branches:

### 1️⃣ CT Branch (3D CNN)
- Input: 64×64×64 nodule-centered CT volumes
- Model: 3D ResNet-style convolutional network
- Output: Learned volumetric representation

### 2️⃣ Structured Branch (MLP)
- Input: Radiologist semantic ratings parsed from XML:
  - subtlety
  - internalStructure
  - calcification
  - sphericity
  - margin
  - lobulation
  - spiculation
  - texture
- Model: Multi-layer perceptron
- Output: Structured feature embedding

### 3️⃣ Late Fusion
- Concatenation of both embeddings
- Final classification head

---

## 📊 Evaluation Metrics

Models are evaluated using:

- ROC-AUC
- F1-score
- Sensitivity (malignant recall)
- Specificity
- Confusion Matrix

### Key Findings

| Model Type        | AUC  |
|-------------------|------|
| CT-only (3D CNN)  | ~0.65–0.70 |
| Structured (MLP)  | 0.98 |
| Late Fusion       | Similar to structured |

---

## ⚠ Important Insight: Feature Leakage

Structured radiologist semantic ratings encode malignancy-related information.

This resulted in:

- Extremely high structured AUC (0.98)
- Fusion not significantly outperforming structured model

After analysis, we identified near-label leakage due to expert-defined attributes strongly correlating with malignancy.

This highlights:
- The importance of feature auditing
- Risks of inflated performance in medical ML
- The need for careful dataset design

---

## 🛠 Data Pipeline

Preprocessing steps include:

- DICOM loading using SimpleITK
- HU clipping and normalization
- Isotropic resampling
- Nodule-centered 3D cropping
- XML parsing using pylidc
- Structured feature extraction
- Patient-wise train/val/test split

---

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/multimodal-lung-cancer-detection.git
cd multimodal-lung-cancer-detection
pip install -r requirements.txt

---

##📚 Dataset

LIDC-IDRI dataset (The Cancer Imaging Archive)

Dataset is not included in this repository.
Users must download it separately from TCIA.
