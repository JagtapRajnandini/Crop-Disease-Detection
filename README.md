# 🌿 AgriScan AI — Precision Crop Pathology & Agronomic Advisory Station

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2.svg)](https://mlflow.org/)

**AgriScan AI** is an in-the-wild agricultural diagnostic platform designed to detect, localize, and prescribe agronomic treatments for foliar crop pathologies. Built using a **Two-Stage Deep Vision Architecture**, the system first identifies the host plant species using a deep classification backbone before localizing disease hotspots with an object detection head—preventing cross-species diagnostic errors.

---

## 📌 Architectural Overview

Standard single-stage object detectors often suffer from **cross-species hallucination** when evaluating visually similar foliage (e.g., misclassifying Potato late blight as Tomato early blight due to similar solanaceous leaf morphology). AgriScan AI addresses this using a cascaded inference pipeline:

```text
       [Field Leaf Photograph]
                  │
                  ▼
   ┌──────────────────────────────┐
   │    Stage 1: ResNet18         │ ───► Predicted Crop Species (13 Classes)
   │  Species Identification      │      Confidence Gate (Threshold: ≥ 75%)
   └──────────────┬───────────────┘
                  │ (Verified Species)
                  ▼
   ┌──────────────────────────────┐
   │    Stage 2: YOLOv8n          │ ───► Lesion Localization (29 Classes)
   │  Pathology Localization     │      Cross-Species Bounding Box Filter
   └──────────────┬───────────────┘
                  │ (Filtered Detections)
                  ▼
   ┌──────────────────────────────┐
   │  Agronomic Advisory Engine   │ ───► Root Causes, Cultural Prevention,
   │  & Pathology Severity Meter  │      Chemical/Organic Spray Prescriptions
   └──────────────────────────────┘
```

* **Stage 1 (Host Identification):** ResNet18 fine-tuned over 13 botanical crop classes (Validation Accuracy: **72.42%**).
* **Stage 2 (Pathology Detection):** YOLOv8n trained on the PlantDoc in-the-wild benchmark across 29 pathological classes (**mAP@0.50: 57.52%**).
* **Confidence Gating & Species Filtering:** If Stage 1 detects the crop with ≥ 75% confidence, Stage 2 suppresses bounding boxes outside the identified botanical family.
* **Universal Crop Override:** In ambiguous field lighting, agronomists can select their target crop from an interactive dropdown to bypass Stage 1 and enforce strict species-specific diagnostics.

---

## 🌾 Supported Crop Species (13 Classes)

| Crop Family | Species Supported |
| :--- | :--- |
| **Solanaceae** | Bell Pepper, Potato, Tomato |
| **Rosaceae** | Apple, Cherry, Peach, Raspberry, Strawberry |
| **Vitaceae** | Grape |
| **Poaceae** | Corn |
| **Fabaceae** | Soyabean |
| **Ericaceae** | Blueberry |
| **Cucurbitaceae**| Squash |

---

## 📊 Experimental Results & Benchmarks

All models were evaluated on the independent **PlantDoc in-the-wild test set** (231 annotated field images):

| Model Component | Architecture | Epochs | Primary Metric | Result |
| :--- | :--- | :---: | :--- | :---: |
| **Lesion Detector** | YOLOv8n (Nano) | 20 | mAP@0.50 | **57.52%** |
| **Species Classifier (v1)** | ResNet18 (Baseline) | 5 | Val Accuracy | 62.11% |
| **Species Classifier (v2)** | ResNet18 (StepLR) | 20 | Val Accuracy | **72.42%** |
| **Cascaded System** | Two-Stage Gated | — | Correct Plant ID Rate | **63.6%** |

*All training runs, loss trajectories, and artifact weights are tracked inside the SQLite-backed MLflow tracking database (`mlruns.db`).*

---

## 🛠️ Project Structure

```text
AIML Project/
├── app.py                         # Production Streamlit field station UI
├── dataset.yaml                   # YOLO dataset configuration
├── mlruns.db                      # SQLite database for MLflow experiment logs
├── test_model.py                  # Evaluation benchmark script on test images
├── test_results.csv               # Complete per-image benchmark output log
├── requirements.txt               # Project dependency specifications
├── runs/
│   ├── classify/
│   │   ├── resnet18_species.pth   # Stage 1 fine-tuned PyTorch model
│   │   └── species_classes.json   # 13 class label mappings
│   └── detect/experiments/
│       └── yolov8n_plantdoc-2/
│           └── weights/best.pt    # Stage 2 trained YOLOv8n weights
├── src/
│   ├── data/
│   │   └── disease_advisory.json  # Agronomic treatment and pathology database
│   └── inference/
│       └── predict_two_stage.py   # Unified Two-Stage inference engine
└── data/
    └── plantdoc/                  # Processed PlantDoc dataset splits
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
Clone the repository and set up a virtual environment:
```powershell
# Clone repository
git clone https://github.com/JagtapRajnandini/Plant-Disease-Detection.git
cd "Plant-Disease-Detection"

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Application
Start the Streamlit Field Diagnostics Station:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Launch MLflow Tracking Dashboard (Optional)
To inspect training curves, hyperparameter configurations, and loss charts:
```powershell
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```
Open your browser at `http://127.0.0.1:5000`.

### 4. Run Model Benchmark Evaluation
To reproduce evaluation metrics across all test images:
```powershell
python test_model.py
```

---

## 🛡️ Agronomic Advisory System
When foliar disease regions are localized, the platform queries an internal agricultural knowledge base (`src/data/disease_advisory.json`) and delivers:
1. **Pathological Etiology:** Root causes, fungal/bacterial vectors, and environmental moisture triggers.
2. **Cultural Controls:** Pruning protocols, canopy aeration guidelines, and drip-irrigation management.
3. **Chemical & Biological Treatments:** Recommended fungicides/bactericides with targeted active ingredients.
4. **Field Report Export:** One-click generation of timestamped diagnostic summaries (`.txt`) for farm record-keeping.

---

## 🎓 Academic Attribution
* **Institution:** MKSSS Cummins College of Engineering for Women, Pune
* **Department:** Department of Computer Engineering
* **Dataset Attribution:** Singh, D., et al. *PlantDoc: A Dataset for Visual Plant Disease Detection in the Wild* (CoDS-COMAD 2020).
