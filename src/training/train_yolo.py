"""
train_yolo.py
-------------
Fast and optimized YOLOv8 training on CPU.
"""

from pathlib import Path
from ultralytics import YOLO

def train():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    YAML_PATH = PROJECT_ROOT / "dataset.yaml"

    # Load YOLOv8n (Nano - 3x lighter and much faster on CPU)
    print("⏳ Loading pretrained YOLOv8n weights...")
    model = YOLO("yolov8n.pt")

    # Start Fine-Tuning
    print("🚀 Starting fast fine-tuning on PlantDoc dataset...")
    results = model.train(
        data=str(YAML_PATH),
        epochs=20,             # 20 epochs is fast and achieves strong convergence
        imgsz=416,             # 416x416 resolution trains 3x faster than 640
        batch=16,              # Efficient batch size
        patience=8,
        save=True,
        device="cpu",
        workers=4,             # Multi-core CPU processing
        project="experiments",
        name="yolov8n_plantdoc"
    )

    print("\n✅ Training complete!")
    print("📁 Best model saved at: experiments/yolov8n_plantdoc/weights/best.pt")

if __name__ == "__main__":
    train()