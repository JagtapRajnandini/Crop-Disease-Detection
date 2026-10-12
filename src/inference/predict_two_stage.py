"""
predict_two_stage.py
--------------------
Unified Inference Engine for the True Two-Stage Pipeline.
Stage 1: ResNet18 predicts the Plant Species.
Stage 2: YOLOv8 detects diseases, but filters out predictions that do not match the ResNet species.
"""

from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

YOLO_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "experiments" / "yolov8n_plantdoc-2" / "weights" / "best.pt"
RESNET_MODEL_PATH = PROJECT_ROOT / "runs" / "classify" / "resnet18_species.pth"
CLASSES_JSON_PATH = PROJECT_ROOT / "runs" / "classify" / "species_classes.json"

# Global models
_YOLO_MODEL = None
_RESNET_MODEL = None
_RESNET_CLASSES = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(str(YOLO_MODEL_PATH))
    return _YOLO_MODEL

def get_resnet_model():
    global _RESNET_MODEL, _RESNET_CLASSES
    if _RESNET_MODEL is None:
        if not RESNET_MODEL_PATH.exists():
            return None, None
            
        with open(CLASSES_JSON_PATH, "r") as f:
            _RESNET_CLASSES = json.load(f)
            
        _RESNET_MODEL = models.resnet18(weights=None)
        num_ftrs = _RESNET_MODEL.fc.in_features
        _RESNET_MODEL.fc = torch.nn.Linear(num_ftrs, len(_RESNET_CLASSES))
        _RESNET_MODEL.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
        _RESNET_MODEL = _RESNET_MODEL.to(device)
        _RESNET_MODEL.eval()
        
    return _RESNET_MODEL, _RESNET_CLASSES

def predict_species(img_pil):
    """Stage 1: Predict plant species using ResNet."""
    model, classes = get_resnet_model()
    if model is None:
        return "Unknown", 0.0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, preds = torch.max(probs, 0)
        
    species_name = classes[preds.item()]
    return species_name, conf.item()

def predict_image(image_input, conf_threshold=0.25, use_two_stage=True):
    """
    Stage 2: YOLO Detection + Filtering
    """
    yolo_model = get_yolo_model()

    if isinstance(image_input, Image.Image):
        img_pil = image_input.convert("RGB")
        img_np = np.array(img_pil)
    elif isinstance(image_input, (str, Path)):
        img_pil = Image.open(str(image_input)).convert("RGB")
        img_np = np.array(img_pil)
    else:
        # If it's a numpy array
        img_np = image_input
        img_pil = Image.fromarray(img_np)

    # 1. Run Stage 1 (ResNet)
    detected_species = "Unknown"
    species_conf = 0.0
    if use_two_stage and RESNET_MODEL_PATH.exists():
        detected_species, species_conf = predict_species(img_pil)

    # 2. Run Stage 2 (YOLO)
    results = yolo_model.predict(source=img_np, conf=conf_threshold, verbose=False)[0]
    
    detections = []
    names = results.names

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = names.get(cls_id, f"Class_{cls_id}")
        
        # YOLO's extracted plant name
        yolo_plant_name = class_name.split("_")[0]
        
        # 3. FILTERING LOGIC — only filter if ResNet is 75%+ confident
        # If ResNet is unsure (<75%), skip filter and trust YOLO's output
        SPECIES_CONF_GATE = 0.75
        if use_two_stage and detected_species != "Unknown" and species_conf >= SPECIES_CONF_GATE:
            if yolo_plant_name.lower() != detected_species.lower():
                continue  # Skip this detection, it's a cross-species error!        

        detections.append({
            "class": class_name,
            "plant": yolo_plant_name.capitalize(),
            "confidence": round(conf, 4),
            "box": [x1, y1, x2, y2]
        })

    # Render custom annotated image (only the filtered boxes!)
    annotated_rgb = img_np.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"{d['class']} {d['confidence']:.2f}"
        cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated_rgb, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if detections:
        top_detection = max(detections, key=lambda d: d["confidence"])
        primary_plant = detected_species.capitalize() if use_two_stage else top_detection["plant"]
        primary_condition = top_detection["class"]
        top_conf = top_detection["confidence"]
    else:
        primary_plant = detected_species.capitalize() if use_two_stage else "Unknown"
        primary_condition = "Healthy / No visible disease pattern detected"
        top_conf = 0.0

    return {
        "primary_plant": primary_plant,
        "species_confidence": species_conf,
        "primary_condition": primary_condition,
        "highest_confidence": top_conf,
        "total_regions_detected": len(detections),
        "detections": detections,
        "annotated_image": annotated_rgb,
        "used_two_stage": use_two_stage and detected_species != "Unknown"
    }