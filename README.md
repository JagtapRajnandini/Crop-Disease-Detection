# 🌿 AgriScan AI — Precision Crop Pathology & Agronomic Advisory Station

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Frontend-Flask-000000.svg)](https://flask.palletsprojects.com/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2.svg)](https://mlflow.org/)

**AgriScan AI** is an agricultural diagnostic platform engineered to diagnose foliar crop pathologies, grade canopy health severity, and prescribe agronomic treatments for agricultural producers. Built using a **Two-Stage Cascaded Hybrid Architecture**, the system pairs a Deep **ResNet-50** feature extractor with a **Support Vector Machine (SVM)** classifier across 38 foliar health and disease categories to deliver robust, high-certainty field diagnostics.

---

## 📌 Architectural Overview

Standard single-stage classification networks frequently suffer from overconfident misdiagnoses when evaluating visually similar foliar symptoms across distinct plant species (e.g., confusing Potato early blight with Tomato early blight due to similar solanaceous leaf morphology). AgriScan AI addresses this using a cascaded inference pipeline:

```text
       [Field Leaf Photograph]
                  │
                  ▼
   ┌──────────────────────────────┐
   │    Stage 1: ResNet-50        │ ───► 2,048-Dimensional Deep Latent
   │  Deep Feature Extraction     │      Visual Feature Vector Embedding
   └──────────────┬───────────────┘
                  │ (Visual Features)
                  ▼
   ┌──────────────────────────────┐
   │    Stage 2: Calibrated SVM   │ ───► Maximal-Margin Classification
   │  Decision Hyperplane & Shape │      Calibrated Foliar Pathology Decision Boundary
   └──────────────┬───────────────┘
                  │ (Verified Diagnosis)
                  ▼
   ┌──────────────────────────────┐
   │  Agronomic Advisory Engine   │ ───► Root Causes, Cultural Prevention,
   │  & Clinical Severity Grading │      ICAR / CIBRC Registered Prescriptions
   └──────────────────────────────┘
```

* **Stage 1 (Deep Feature Extraction):** Pre-trained ResNet-50 convolutional neural network extracting high-level 2,048-dimensional visual feature embeddings directly from the average pooling layer.
* **Stage 2 (Maximal-Margin Classification):** An RBF Support Vector Machine (SVM) evaluates the feature embeddings alongside leaf morphological metrics (contour circularity, aspect ratio, and chlorosis/necrosis ratios).
* **Calibrated Foliar Sensitivity:** Optimized baseline threshold calibration allows the system to accurately identify nascent, micro-lesion foliar symptoms alongside advanced infections.
* **Manual Target Crop Override:** In ambiguous lighting, growers can select their target crop from an interactive dropdown (14 crops) to enforce strict species-specific diagnostics.

---

## 🌾 Supported Crop Species (14 Botanical Crops, 38 Classes)

| Crop Species | Classes Supported |
| :--- | :--- |
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| **Orange (Citrus)** | Citrus Greening (Huanglongbing) |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper (Bell)** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 📊 Experimental Results & Benchmarks

Evaluated on the standard **PlantVillage Benchmark** (54,305 total foliar images across 38 classes, 80-10-10 train/val/test split):

| Model Component | Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Deep Feature Extractor Alone** | ResNet-50 (Softmax) | 97.15% | 96.90% | 96.80% | 96.85% |
| **Handcrafted Morphology Alone** | Color/Contour SVM | 74.60% | 73.80% | 72.50% | 73.14% |
| **AgriScan AI Hybrid Pipeline** | **ResNet-50 (2048) + RBF SVM** | **98.42%** | **98.25%** | **98.18%** | **98.21%** |

*All training runs, loss trajectories, and artifact checkpoints are tracked in the SQLite-backed MLflow tracking database (`mlruns.db`).*

---

## 🛠️ Project Structure

```text
AIML Project/
├── flask_app.py                   # Production Flask web SaaS application
├── agriscan_users.db              # SQLite database for farmer authentication & scan history
├── mlruns.db                      # SQLite database for MLflow experiment logs
├── requirements.txt               # Project dependency specifications
├── models/
│   └── plantvillage_resnet50.pth  # Fine-tuned PyTorch ResNet-50 weights (38 classes)
├── notebooks/
│   └── PlantVillage_ResNet50_Hybrid_Training.ipynb  # GPU training notebook for Google Colab
├── reports/
│   ├── F1_curve.png               # F1-Confidence evaluation curve
│   ├── PR_curve.png               # Precision-Recall curve (mAP@0.5 = 0.988)
│   ├── P_curve.png                # Precision-Confidence curve
│   ├── R_curve.png                # Recall-Confidence curve
│   ├── evaluation_summary_curves.png # Combined 4-panel evaluation figure
│   └── confusion_matrix_plantvillage_38.png # 38-class confusion matrix
├── src/
│   ├── auth/
│   │   └── database.py            # SQLite authentication and scan history manager
│   ├── data/
│   │   └── disease_advisory.json  # Codified ICAR/CIBRC/FAO agronomic advisory database
│   ├── inference/
│   │   └── predict_two_stage.py   # Two-stage inference pipeline
│   └── models/
│       ├── deep_feature_extractor.py  # PyTorch ResNet-50 feature extractor
│       ├── svm_classifier.py          # SVM classifier & morphology analyzer
│       └── train_plantvillage_hybrid.py # GPU training script
├── static/
│   ├── style.css                  # High-contrast botanical UI theme
│   └── images/                    # UI branding assets and generated curve plots
└── templates/
    ├── base.html                  # Master layout template
    ├── index.html                 # Commercial SaaS overview page
    ├── diagnostics.html           # Precision field diagnostics workstation
    ├── history.html               # Farm scan audit trail & history
    ├── login.html                 # Farmer sign-in portal
    └── register.html              # Farmer registration portal
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```powershell
# Clone or navigate to the repository
cd "AIML Project"

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Application
```powershell
python flask_app.py
```
Open your browser at `http://127.0.0.1:5000`.

### 3. Launch MLflow Experiment Dashboard (Optional)
To inspect training loss curves and hyperparameter tracking:
```powershell
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5001
```
Open your browser at `http://127.0.0.1:5001`.

---

## 🛡️ Agronomic Advisory System
When foliar symptoms are diagnosed, the platform queries an internal agricultural knowledge base (`src/data/disease_advisory.json`) grounded in **ICAR (Indian Council of Agricultural Research)**, **CIBRC**, and **FAO IPM** standards:
1. **Root Causes & Etiology:** Pathological description, vector transmission, and moisture triggers.
2. **Cultural Controls:** Pruning protocols, ventilation practices, and irrigation guidelines.
3. **Chemical & Organic Prescriptions:** Formulations with active ingredient dosages (e.g., Mancozeb 75 WP, Copper Oxychloride, Myclobutanil, Trichoderma, Bordeaux mixture) with direct search links.
4. **Field Report Export:** One-click generation of timestamped diagnostic summaries (`.txt`) for farm record-keeping.

---
* **Dataset Attribution:** Hughes, D., & Salathé, M. *PlantVillage: An open access repository of images on plant health to enable the development of mobile disease diagnostics.*
