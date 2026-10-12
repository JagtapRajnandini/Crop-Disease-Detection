"""
evaluate_test.py
----------------
Evaluates the trained YOLOv8 model on the unseen PlantDoc test split
and generates visual predictions on sample test images.
"""

from pathlib import Path
from ultralytics import YOLO

def main():
    # ── Paths ──────────────────────────────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    YAML_PATH = PROJECT_ROOT / "dataset.yaml"
    MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "experiments" / "yolov8n_plantdoc-2" / "weights" / "best.pt"

    if not MODEL_PATH.exists():
        print(f"❌ Model weights not found at: {MODEL_PATH}")
        return

    # 1. Load the trained model
    print(f"⏳ Loading trained model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # 2. Run official evaluation on the TEST split
    print("\n📊 Running evaluation on UNSEEN TEST set (231 images)...")
    metrics = model.val(
        data=str(YAML_PATH),
        split="test",          # Explicitly evaluate on TEST split
        imgsz=416,
        batch=16,
        device="cpu",
        workers=0,
        project="runs/detect",
        name="test_evaluation"
    )

    print("\n" + "=" * 60)
    print("  🏆 OFFICIAL TEST SET EVALUATION METRICS")
    print("=" * 60)
    print(f"  Precision (P)    : {metrics.box.mp:.4f} ({metrics.box.mp * 100:.2f}%)")
    print(f"  Recall (R)       : {metrics.box.mr:.4f} ({metrics.box.mr * 100:.2f}%)")
    print(f"  mAP@0.50         : {metrics.box.map50:.4f} ({metrics.box.map50 * 100:.2f}%)")
    print(f"  mAP@0.50:0.95    : {metrics.box.map:.4f} ({metrics.box.map * 100:.2f}%)")
    print("=" * 60)

    # 3. Run visual prediction on 3 sample test images
    test_images_dir = PROJECT_ROOT / "data" / "plantdoc" / "images" / "test"
    sample_images = list(test_images_dir.glob("*.jpg"))[:3]

    if sample_images:
        print("\n🖼️ Generating visual predictions on sample test images...")
        model.predict(
            source=[str(img) for img in sample_images],
            save=True,
            conf=0.25,
            project="runs/detect",
            name="test_sample_predictions"
        )
        print("📁 Sample predictions saved to: runs/detect/test_sample_predictions/")

if __name__ == "__main__":
    main()