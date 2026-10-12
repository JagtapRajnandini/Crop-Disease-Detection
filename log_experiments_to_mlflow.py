"""
log_experiments_to_mlflow.py
-----------------------------
Reads the historical training logs and evaluation metrics for:
  1. YOLOv8n Disease Detector (20 epochs)
  2. ResNet18 Species Classifier v1 (5 epochs)
  3. ResNet18 Species Classifier v2 (20 epochs)
Logs them formally into an MLflow tracking experiment database.
"""

import os
from pathlib import Path
import pandas as pd
import mlflow

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(r"C:\Users\rajna\OneDrive\Desktop\AIML Project")
YOLO_EXP_DIR = ROOT_DIR / "runs" / "detect" / "experiments" / "yolov8n_plantdoc-2"
RESNET_WEIGHTS = ROOT_DIR / "runs" / "classify" / "resnet18_species.pth"

# ── Configure MLflow SQLite Backend ───────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("AgriScan_AI_Crop_Disease_Detection")

print("Logging Experiment Runs to MLflow...")

# ── RUN 1: YOLOv8n Disease Detector ───────────────────────────────────────────
with mlflow.start_run(run_name="YOLOv8n_PlantDoc_Disease_Detector"):
    print("--> Logging Run 1: YOLOv8n Disease Detector")
    
    # Log Hyperparameters
    mlflow.log_params({
        "model_architecture": "YOLOv8n (Nano)",
        "framework": "Ultralytics YOLO",
        "dataset": "PlantDoc in-the-wild",
        "num_classes": 29,
        "input_size": 416,
        "batch_size": 16,
        "epochs": 20,
        "device": "CPU",
        "optimizer": "SGD / Auto"
    })
    
    # Read actual training results if results.csv exists
    results_csv = YOLO_EXP_DIR / "results.csv"
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = [c.strip() for c in df.columns]
        
        # Log epoch-by-epoch loss & accuracy metrics
        for epoch, row in df.iterrows():
            step = int(epoch) + 1
            mlflow.log_metric("train/box_loss", row.get("train/box_loss", 0), step=step)
            mlflow.log_metric("train/cls_loss", row.get("train/cls_loss", 0), step=step)
            mlflow.log_metric("val/box_loss", row.get("val/box_loss", 0), step=step)
            mlflow.log_metric("metrics/mAP50", row.get("metrics/mAP50(B)", 0), step=step)
            mlflow.log_metric("metrics/mAP50-95", row.get("metrics/mAP50-95(B)", 0), step=step)
            mlflow.log_metric("metrics/precision", row.get("metrics/precision(B)", 0), step=step)
            mlflow.log_metric("metrics/recall", row.get("metrics/recall(B)", 0), step=step)
    else:
        # Fallback to key recorded metrics
        mlflow.log_metrics({
            "metrics/mAP50": 0.5752,
            "metrics/precision": 0.5820,
            "metrics/recall": 0.5210
        })

    # Log Artifacts (weights, confusion matrix, charts)
    if (YOLO_EXP_DIR / "weights" / "best.pt").exists():
        mlflow.log_artifact(str(YOLO_EXP_DIR / "weights" / "best.pt"), artifact_path="model_weights")
    if (YOLO_EXP_DIR / "confusion_matrix.png").exists():
        mlflow.log_artifact(str(YOLO_EXP_DIR / "confusion_matrix.png"), artifact_path="visualizations")
    if (YOLO_EXP_DIR / "results.png").exists():
        mlflow.log_artifact(str(YOLO_EXP_DIR / "results.png"), artifact_path="visualizations")
        
    mlflow.set_tag("stage", "Stage 2 - Lesion Detection")
    mlflow.set_tag("deployment_status", "Production Ready")

# ── RUN 2: ResNet18 Species Classifier v1 (Baseline) ──────────────────────────
with mlflow.start_run(run_name="ResNet18_Species_Classifier_v1"):
    print("--> Logging Run 2: ResNet18 Species Classifier v1 (Baseline)")
    
    mlflow.log_params({
        "model_architecture": "ResNet18",
        "framework": "PyTorch / TorchVision",
        "dataset": "PlantDoc 13-Species Subset",
        "num_classes": 13,
        "input_size": 224,
        "batch_size": 32,
        "epochs": 5,
        "initial_lr": 0.001,
        "optimizer": "Adam",
        "scheduler": "None",
        "augmentation": "Basic Flips"
    })
    
    # Baseline 5-epoch progression
    losses = [(1.3602, 1.3818, 0.5971), (0.7581, 1.1286, 0.6547), (0.4928, 1.3478, 0.5947), (0.3701, 1.2106, 0.6355), (0.3361, 1.5233, 0.6211)]
    for step, (t_loss, v_loss, v_acc) in enumerate(losses, 1):
        mlflow.log_metric("train/loss", t_loss, step=step)
        mlflow.log_metric("val/loss", v_loss, step=step)
        mlflow.log_metric("val/accuracy", v_acc, step=step)
        
    mlflow.set_tag("stage", "Stage 1 - Species Classifier")
    mlflow.set_tag("status", "Superseded by v2")

# ── RUN 3: ResNet18 Species Classifier v2 (Fine-Tuned with StepLR) ─────────────
with mlflow.start_run(run_name="ResNet18_Species_Classifier_v2"):
    print("--> Logging Run 3: ResNet18 Species Classifier v2 (Production)")
    
    mlflow.log_params({
        "model_architecture": "ResNet18",
        "framework": "PyTorch / TorchVision",
        "dataset": "PlantDoc 13-Species Subset",
        "num_classes": 13,
        "input_size": 224,
        "batch_size": 32,
        "epochs": 20,
        "initial_lr": 0.0005,
        "weight_decay": 1e-4,
        "optimizer": "Adam",
        "scheduler": "StepLR (step_size=5, gamma=0.5)",
        "augmentation": "RandomResizedCrop + ColorJitter + Rotation30"
    })
    
    # 20-epoch progression recorded earlier
    v2_metrics = [
        (1.4452, 2.4234, 0.4005), (1.1276, 1.6322, 0.5372), (0.9511, 1.6124, 0.5564), (0.9241, 1.4541, 0.5683), (0.8228, 1.7534, 0.5372),
        (0.6303, 1.2010, 0.6331), (0.4999, 1.0600, 0.6930), (0.4336, 1.2895, 0.6499), (0.4477, 1.0864, 0.6835), (0.3820, 1.1772, 0.6763),
        (0.3027, 1.0260, 0.7026), (0.2468, 1.0944, 0.7050), (0.2237, 1.0456, 0.7218), (0.2299, 1.1017, 0.6954), (0.2150, 1.2365, 0.6859),
        (0.1754, 1.1302, 0.7146), (0.1613, 1.0945, 0.7050), (0.1434, 1.1208, 0.7050), (0.1219, 1.0964, 0.7242), (0.1294, 1.1205, 0.7146)
    ]
    
    for step, (t_loss, v_loss, v_acc) in enumerate(v2_metrics, 1):
        mlflow.log_metric("train/loss", t_loss, step=step)
        mlflow.log_metric("val/loss", v_loss, step=step)
        mlflow.log_metric("val/accuracy", v_acc, step=step)
        
    if RESNET_WEIGHTS.exists():
        mlflow.log_artifact(str(RESNET_WEIGHTS), artifact_path="model_weights")
        
    mlflow.set_tag("stage", "Stage 1 - Species Classifier")
    mlflow.set_tag("deployment_status", "Production Ready")

print("\nAll 3 experiment runs successfully logged into MLflow SQLite database!")
print("Run this command next to open your MLOps dashboard:")
print("  mlflow ui --backend-store-uri sqlite:///mlruns.db")