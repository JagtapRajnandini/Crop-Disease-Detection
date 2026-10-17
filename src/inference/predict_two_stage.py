"""
predict_two_stage.py
--------------------
AgriScan AI — Two-Stage Hybrid Diagnostic Pipeline.
Stage 1: Calls DeepFeatureExtractor (ResNet-50)
Stage 2: Calls PlantVillageSVMClassifier (SVM + Botanical Vision Engine)
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image

# Import from separated modules
from src.models.deep_feature_extractor import extract_features
from src.models.svm_classifier import classify_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ADVISORY_PATH = PROJECT_ROOT / "src" / "data" / "disease_advisory.json"

# Load Advisory Knowledgebase
ADVISORY_DATA = {}
if ADVISORY_PATH.exists():
    try:
        with open(ADVISORY_PATH, "r", encoding="utf-8") as f:
            ADVISORY_DATA = json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to load advisory data: {e}")

def predict_image(
    image_input,
    conf_threshold: float = 0.25,
    use_two_stage: bool = True,
    manual_crop: str = None,
    **kwargs
) -> dict:
    """
    Two-Stage Hybrid Diagnostic Function invoked by Flask:
    1. Deep Learning (ResNet-50) extracts 2,048 spatial features.
    2. Machine Learning (SVM) predicts disease and calculates confidence.
    3. Maps diagnosis to Agronomic Advisory and medicine links.
    """
    # Obtain original PIL image
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        pil_img = Image.fromarray(image_input).convert("RGB")

    # Stage 1: Deep Learning Feature Extraction
    deep_features = extract_features(pil_img)

    # Stage 2: Classical Machine Learning Classification (SVM + Morphology + Sensitivity Threshold)
    species, disease, raw_class, confidence = classify_features(
        deep_features, pil_image=pil_img, sensitivity=conf_threshold, crop_hint=manual_crop
    )

    # Honor user crop override if specified in UI dropdown
    if manual_crop and manual_crop != "Auto-Detect via Field Vision":
        species = manual_crop

    is_healthy = "healthy" in raw_class.lower()

    # Retrieve Advisory Dossier
    adv = ADVISORY_DATA.get(raw_class, {})
    if not adv:
        adv = {
            "symptoms": "Foliar disease symptoms detected on leaf surface.",
            "chemical": ["Broad-spectrum agricultural copper fungicide", "Mancozeb 75 WP"],
            "organic": ["Neem oil 1%", "Bio-fungicide Trichoderma"],
            "prevention": "Inspect plants regularly, ensure adequate spacing, and avoid overhead watering."
        }

    condition_label = "Healthy Foliage" if is_healthy else disease

    return {
        "status": "success",
        "primary_plant": species,
        "primary_condition": condition_label,
        "crop_species": species,
        "disease_name": disease,
        "raw_class": raw_class,
        "highest_confidence": round(confidence, 4),
        "confidence": round(confidence, 4),
        "is_healthy": is_healthy,
        "total_detections": 0 if is_healthy else 1,
        "detections": [] if is_healthy else [{
            "class": raw_class,
            "condition": disease,
            "plant": species,
            "crop_species": species,
            "severity": "Critical",
            "confidence": round(confidence, 4)
        }],
        "annotated_image": np.array(pil_img),
        "advisory": adv
    }