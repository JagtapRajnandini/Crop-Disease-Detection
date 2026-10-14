"""
predict_two_stage.py
--------------------
Unified Inference Engine for the True Two-Stage Cascaded Pipeline.
Stage 1: ResNet-18 predicts the botanical plant species.
Stage 2: YOLOv8 detects localized foliar pathologies and lesions.

Guarantees 100% species-disease consistency:
- The predicted host plant and disease condition will NEVER contradict each other.
- Proper multi-word plant name extraction (e.g. 'bell_pepper' -> 'Bell Pepper').
- Intelligent cross-species validation between whole-image classification and localized detection.
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

# Global models (cached for fast repeated inference)
_YOLO_MODEL = None
_RESNET_MODEL = None
_RESNET_CLASSES = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

KNOWN_SPECIES = [
    "bell_pepper", "apple", "blueberry", "cherry", "corn", "grape",
    "peach", "potato", "raspberry", "soyabean", "squash", "strawberry", "tomato"
]

def extract_plant_from_class(class_name: str) -> str:
    """
    Accurately extracts the botanical plant species key from any class name.
    Correctly handles compound names such as 'bell_pepper' without truncation.
    """
    c = class_name.lower().strip()
    for sp in KNOWN_SPECIES:
        if c.startswith(sp):
            return sp
    return c.split("_")[0]

def format_plant_name(plant_key: str) -> str:
    """Formats internal plant key (e.g. 'bell_pepper') to display name ('Bell Pepper')."""
    if not plant_key or plant_key == "Unknown":
        return "Unknown"
    return plant_key.replace("_", " ").title()

def get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model weights not found at {YOLO_MODEL_PATH}")
        _YOLO_MODEL = YOLO(str(YOLO_MODEL_PATH))
    return _YOLO_MODEL

def get_resnet_model():
    global _RESNET_MODEL, _RESNET_CLASSES
    if _RESNET_MODEL is None:
        if not RESNET_MODEL_PATH.exists() or not CLASSES_JSON_PATH.exists():
            return None, None

        try:
            with open(CLASSES_JSON_PATH, "r") as f:
                _RESNET_CLASSES = json.load(f)

            _RESNET_MODEL = models.resnet18(weights=None)
            num_ftrs = _RESNET_MODEL.fc.in_features
            _RESNET_MODEL.fc = torch.nn.Linear(num_ftrs, len(_RESNET_CLASSES))
            _RESNET_MODEL.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device, weights_only=False))
            _RESNET_MODEL = _RESNET_MODEL.to(device)
            _RESNET_MODEL.eval()
        except Exception as e:
            print(f"[Warning] Failed to load ResNet model: {e}")
            return None, None

    return _RESNET_MODEL, _RESNET_CLASSES

def predict_species(img_pil):
    """Stage 1: Predict plant species using ResNet18."""
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

def predict_image(image_input, conf_threshold=0.25, use_two_stage=True, manual_crop=None):
    """
    Unified Two-Stage Inference Engine.

    Parameters
    ----------
    image_input    : PIL.Image, numpy array, or file path
    conf_threshold : float  — YOLO minimum detection confidence (0.0–1.0)
    use_two_stage  : bool   — enable ResNet species validation
    manual_crop    : str or None — user manual crop selection override

    Returns
    -------
    dict with keys:
        primary_plant, species_confidence, primary_condition,
        highest_confidence, total_regions_detected, detections,
        annotated_image, used_two_stage, manual_override
    """
    yolo_model = get_yolo_model()

    # ── 1. Normalise image input ───────────────────────────────────────────────
    if isinstance(image_input, Image.Image):
        img_pil = image_input.convert("RGB")
        img_np = np.array(img_pil)
    elif isinstance(image_input, (str, Path)):
        img_pil = Image.open(str(image_input)).convert("RGB")
        img_np = np.array(img_pil)
    else:
        img_np = image_input
        img_pil = Image.fromarray(img_np)

    # ── 2. Handle manual crop selection ────────────────────────────────────────
    manual_override = False
    target_crop_key = None

    if manual_crop is not None:
        crop_clean = manual_crop.strip().lower()
        if crop_clean not in ("", "none", "auto-detect (ai)", "auto-detect via field vision", "auto-detect"):
            target_crop_key = crop_clean.replace(" ", "_")
            manual_override = True

    # ── 3. Stage 1: Species identification via ResNet ──────────────────────────
    resnet_species = "Unknown"
    resnet_conf = 0.0

    if manual_override:
        resnet_species = target_crop_key
        resnet_conf = 1.0
    elif use_two_stage:
        resnet_species, resnet_conf = predict_species(img_pil)

    # ── 4. Stage 2: YOLO localized disease detection ───────────────────────────
    results = yolo_model.predict(source=img_np, conf=conf_threshold, verbose=False)[0]
    names = results.names

    raw_detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = names.get(cls_id, f"Class_{cls_id}")
        plant_key = extract_plant_from_class(class_name)

        raw_detections.append({
            "class": class_name,
            "plant": format_plant_name(plant_key),
            "plant_key": plant_key,
            "confidence": round(conf, 4),
            "box": [x1, y1, x2, y2]
        })

    # ── 5. Intelligent Cross-Stage Consistency & Filtering ─────────────────────
    filtered_detections = []

    if manual_override:
        # User explicitly stated the crop: only keep detections matching that crop
        for d in raw_detections:
            if d["plant_key"] == target_crop_key:
                filtered_detections.append(d)
        dominant_plant_key = target_crop_key
        dominant_plant_name = format_plant_name(target_crop_key)

    elif raw_detections:
        # Auto-detect mode with detections:
        # Determine the dominant plant family supported by YOLO detections
        plant_votes = {}
        for d in raw_detections:
            pk = d["plant_key"]
            plant_votes[pk] = plant_votes.get(pk, 0.0) + d["confidence"]

        # If ResNet has high confidence (>= 0.80) and matches a YOLO detection family, bolster it
        if use_two_stage and resnet_species in plant_votes and resnet_conf >= 0.80:
            plant_votes[resnet_species] += (resnet_conf * 1.5)

        dominant_plant_key = max(plant_votes, key=plant_votes.get)
        dominant_plant_name = format_plant_name(dominant_plant_key)

        # Keep all detections consistent with the verified dominant plant
        for d in raw_detections:
            if d["plant_key"] == dominant_plant_key:
                filtered_detections.append(d)

    else:
        # No detections (healthy foliage)
        if use_two_stage and resnet_species != "Unknown":
            dominant_plant_key = resnet_species
            dominant_plant_name = format_plant_name(resnet_species)
        else:
            dominant_plant_key = "Unknown"
            dominant_plant_name = "Unknown"

    # ── 6. Draw Bounding Boxes on Verified Detections ──────────────────────────
    annotated_rgb = img_np.copy()
    for d in filtered_detections:
        x1, y1, x2, y2 = d["box"]
        label = f"{d['class']} {d['confidence']:.2f}"
        cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated_rgb, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # ── 7. Primary Condition & Result Assembly (Guaranteed Consistency) ───────
    if filtered_detections:
        top_detection = max(filtered_detections, key=lambda d: d["confidence"])
        # Primary plant is GUARANTEED to match the detected disease family
        primary_plant = dominant_plant_name
        primary_condition = top_detection["class"]
        top_conf = top_detection["confidence"]
    else:
        primary_plant = dominant_plant_name
        primary_condition = "Healthy / No visible disease pattern detected"
        top_conf = 0.0

    # Clean detections dictionary (exclude internal helper keys)
    final_detections = []
    for d in filtered_detections:
        final_detections.append({
            "class": d["class"],
            "plant": d["plant"],
            "confidence": d["confidence"],
            "box": d["box"]
        })

    return {
        "primary_plant":          primary_plant,
        "species_confidence":     resnet_conf if not manual_override else 1.0,
        "primary_condition":      primary_condition,
        "highest_confidence":     top_conf,
        "total_regions_detected": len(final_detections),
        "detections":             final_detections,
        "annotated_image":        annotated_rgb,
        "used_two_stage":         use_two_stage,
        "manual_override":        manual_override,
    }