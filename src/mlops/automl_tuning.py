"""
automl_tuning.py
----------------
AutoML Hyperparameter Optimization for AgriScan AI.
Uses Ultralytics Genetic Algorithm Tuning to optimize:
  - lr0: Initial learning rate
  - lrf: Final learning rate factor
  - momentum: SGD/Adam momentum
  - weight_decay: L2 regularization
"""

import os
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(r"C:\Users\rajna\OneDrive\Desktop\AIML Project")
DATA_YAML = ROOT_DIR / "dataset.yaml"

def run_automl_tuning():
    print("=" * 65)
    print("Starting AutoML Hyperparameter Search (Genetic Algorithm)...")
    print("=" * 65)

    # 1. Load baseline Nano model
    model = YOLO("yolov8n.pt")

    # 2. Run AutoML tuning
    # NOTE: Set iterations=2 and epochs=1 for demonstration on CPU
    # In production with GPU clusters, iterations=30, epochs=10
    results = model.tune(
        data=str(DATA_YAML),
        epochs=1,          # Fast evaluation per trial
        iterations=2,      # 2 hyperparameter search trials
        optimizer="AdamW",
        plots=False,
        save=False,
        val=True,
        device="cpu"
    )

    print("\nAutoML Search Complete!")
    print("Optimal Hyperparameters Discovered:")
    print(results)

if __name__ == "__main__":
    run_automl_tuning()
    run_automl_tuning()
    