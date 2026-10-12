"""
predict.py
----------
Unified Inference Engine for Disease Detection & Localization.
Accepts an image path or PIL Image/NumPy array, runs YOLOv8 detection,
and returns a standardized prediction dictionary with bounding boxes.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "experiments" / "yolov8n_plantdoc-2" / "weights" / "best.pt"

# Global model instance (loaded once for fast repeated inference)
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
        _MODEL = YOLO(str(MODEL_PATH))
    return _MODEL

def predict_image(image_input, conf_threshold=0.25):
    """
    Runs disease detection on an input image.

    Args:
        image_input: File path (str/Path), PIL Image, or NumPy BGR array.
        conf_threshold: Minimum confidence score (0.0 to 1.0) to keep boxes.

    Returns:
        dict: Standardized output with detected plant, condition, confidence,
              bounding boxes, and an annotated image (NumPy array RGB).
    """
    model = get_model()

    # Convert PIL Image to RGB NumPy if necessary
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input)
    elif isinstance(image_input, (str, Path)):
        img_np = cv2.imread(str(image_input))
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    else:
        img_np = image_input

    # Run inference
    results = model.predict(source=img_np, conf=conf_threshold, verbose=False)[0]

    detections = []
    names = results.names

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_name = names.get(cls_id, f"Class_{cls_id}")
        
        # Extract plant name from class_name (e.g. 'tomato_early_blight_leaf' -> 'Tomato')
        plant_name = class_name.split("_")[0].capitalize()

        detections.append({
            "class": class_name,
            "plant": plant_name,
            "confidence": round(conf, 4),
            "box": [x1, y1, x2, y2]
        })

    # Render annotated image (with boxes and labels drawn)
    annotated_bgr = results.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Primary detection summary (highest confidence detection)
    if detections:
        top_detection = max(detections, key=lambda d: d["confidence"])
        primary_plant = top_detection["plant"]
        primary_condition = top_detection["class"]
        top_conf = top_detection["confidence"]
    else:
        primary_plant = "Unknown / No detection"
        primary_condition = "Healthy / No visible disease pattern detected"
        top_conf = 0.0

    return {
        "primary_plant": primary_plant,
        "primary_condition": primary_condition,
        "highest_confidence": top_conf,
        "total_regions_detected": len(detections),
        "detections": detections,
        "annotated_image": annotated_rgb
    }

# Quick test if run directly
if __name__ == "__main__":
    test_img = list((PROJECT_ROOT / "data" / "plantdoc" / "images" / "test").glob("*.jpg"))[0]
    result = predict_image(test_img)
    print("✅ Inference Test Successful!")
    print(f"🌿 Primary Plant       : {result['primary_plant']}")
    print(f"🦠 Primary Condition   : {result['primary_condition']}")
    print(f"🎯 Confidence Score    : {result['highest_confidence'] * 100:.2f}%")
    print(f"📦 Regions Detected    : {result['total_regions_detected']}")