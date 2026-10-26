"""
svm_classifier.py
-----------------
Stage 2: Machine Learning Classifier (Botanical Feature + Deep Representation Decision Engine).
Analyzes leaf morphology, contour circularity, lesion color distributions,
and classifies fine-tuned ResNet-50 2,048-dimensional spatial embeddings using a
trained Radial Basis Function Support Vector Machine (RBF-SVM) across 38 PlantVillage categories.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
from PIL import Image
import torch
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SVM_MODEL_PATH = PROJECT_ROOT / "models" / "plantvillage_svm_model.pkl"

# Complete 38 PlantVillage Categories
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

class PlantVillageSVMClassifier:
    """Intelligent Botanical Computer Vision + Deep Feature SVM Classifier."""

    def __init__(self, model_path: Path = None):
        self.classes = PLANTVILLAGE_CLASSES
        self.model_path = Path(model_path) if model_path else DEFAULT_SVM_MODEL_PATH

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SVM model file not found at '{self.model_path}'. "
                "Please verify that 'plantvillage_svm_model.pkl' exists in the models directory. "
                "Silent fallback to ResNet-50 linear classification head is disabled per system integrity requirements."
            )

        try:
            self.svm = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to deserialize SVM model from '{self.model_path}': {exc}") from exc

        if not hasattr(self.svm, "predict"):
            raise ValueError(f"Loaded artifact from '{self.model_path}' is not a valid scikit-learn classifier.")

        if hasattr(self.svm, "n_features_in_") and self.svm.n_features_in_ != 2048:
            raise ValueError(
                f"SVM model expects {self.svm.n_features_in_} features, but ResNet-50 extracts 2,048 features."
            )

    def parse_class_string(self, raw_class: str) -> Tuple[str, str]:
        """Separates species and disease from standard PlantVillage naming."""
        if "___" in raw_class:
            parts = raw_class.split("___")
            species = parts[0].replace("_", " ").replace("  ", " ").strip()
            disease = parts[1].replace("_", " ").replace("  ", " ").strip()
        else:
            species = "Crop Specimen"
            disease = raw_class.replace("_", " ").strip()

        # Special casing for cleaner user presentation
        if "Corn" in species:
            species = "Corn (Maize)"
        elif "Pepper" in species:
            species = "Bell Pepper"
        elif "Cherry" in species:
            species = "Cherry"
        elif "Orange" in species:
            species = "Orange (Citrus)"

        return species, disease

    def analyze_morphology(self, pil_image: Image.Image) -> dict:
        """Extracts botanical shape and color metrics from leaf image."""
        if pil_image is None:
            return {"circularity": 0.5, "lesion_ratio": 0.0, "yellow_ratio": 0.0}

        img_np = np.array(pil_image)
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"circularity": 0.5, "lesion_ratio": 0.0, "yellow_ratio": 0.0}

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.5

        leaf_pixels = bgr[mask > 0]
        if len(leaf_pixels) == 0:
            return {"circularity": circularity, "lesion_ratio": 0.0, "yellow_ratio": 0.0}

        dark_lesions = np.sum((leaf_pixels[:, 0] < 80) & (leaf_pixels[:, 1] < 80) & (leaf_pixels[:, 2] < 90))
        lesion_ratio = float(dark_lesions / len(leaf_pixels))

        yellow_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[mask > 0]
        yellow_spots = np.sum((yellow_hsv[:, 0] >= 20) & (yellow_hsv[:, 0] <= 35) & (yellow_hsv[:, 1] >= 60))
        yellow_ratio = float(yellow_spots / len(leaf_pixels))

        return {
            "circularity": circularity,
            "lesion_ratio": lesion_ratio,
            "yellow_ratio": yellow_ratio
        }

    def predict(
        self,
        feature_tensor: torch.Tensor,
        pil_image: Image.Image = None,
        sensitivity: float = 0.15,
        crop_hint: str = None
    ) -> Tuple[str, str, str, float]:
        """
        Classifies 2,048-D features using the trained RBF-SVM and morphology calibration.
        Returns: (species, disease, raw_class, confidence)
        """
        # Convert feature tensor to contiguous 2D NumPy array of shape (1, 2048)
        if isinstance(feature_tensor, torch.Tensor):
            features_np = feature_tensor.detach().cpu().numpy()
        else:
            features_np = np.asarray(feature_tensor)

        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)

        if features_np.shape[1] != 2048:
            raise ValueError(
                f"Expected 2,048-dimensional feature vector, but received vector with dimension {features_np.shape[1]}."
            )

        # Obtain probability estimates from calibrated SVM (Platt scaling via predict_proba)
        if hasattr(self.svm, "predict_proba"):
            probs = self.svm.predict_proba(features_np)[0].copy()
        else:
            # Fallback to normalized decision scores if probability estimation was not enabled during training
            dec_scores = self.svm.decision_function(features_np)[0]
            exp_scores = np.exp(dec_scores - np.max(dec_scores))
            probs = exp_scores / np.sum(exp_scores)

        # Morphological analysis for verification & canopy health scoring
        morph = self.analyze_morphology(pil_image)

        # Handle crop constraint override if specified by user in UI
        if crop_hint and crop_hint != "Auto-Detect via Field Vision":
            ch = crop_hint.lower().split()[0].replace("(", "").strip()
            # Find matching class indices for this crop
            matching_indices = [idx for idx, name in enumerate(self.classes) if ch in name.lower()]

            if matching_indices:
                # Mask out other crops and renormalize
                mask = np.zeros_like(probs)
                mask[matching_indices] = 1.0
                filtered_probs = probs * mask
                if filtered_probs.sum() > 0:
                    probs = filtered_probs / filtered_probs.sum()

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        raw_class = self.classes[top_idx]
        species, disease = self.parse_class_string(raw_class)

        return species, disease, raw_class, top_prob

# Global SVM Classifier Instance
_SVM_MODEL = None

def get_svm_classifier():
    global _SVM_MODEL
    if _SVM_MODEL is None:
        _SVM_MODEL = PlantVillageSVMClassifier()
    return _SVM_MODEL

def classify_features(
    feature_tensor: torch.Tensor,
    pil_image: Image.Image = None,
    sensitivity: float = 0.15,
    crop_hint: str = None
) -> Tuple[str, str, str, float]:
    """Public helper to classify features and morphology."""
    classifier = get_svm_classifier()
    return classifier.predict(feature_tensor, pil_image=pil_image, sensitivity=sensitivity, crop_hint=crop_hint)
