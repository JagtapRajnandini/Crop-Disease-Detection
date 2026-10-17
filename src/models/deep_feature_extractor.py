"""
deep_feature_extractor.py
---------------------------
Stage 1: Deep Learning Module (PyTorch ResNet-50).
Extracts 2,048 high-dimensional spatial/visual features from raw leaf images.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Standard ImageNet normalization pipeline for ResNet-50
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DeepFeatureExtractor(nn.Module):
    """ResNet-50 Convolutional Neural Network used as a Deep Visual Feature Extractor."""
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet-50 backbone
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the final classification layer, keeping the 2,048-dim feature pool
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_extractor.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            # Flatten from (Batch, 2048, 1, 1) to (Batch, 2048)
            return torch.flatten(features, 1)

# Global feature extractor instance
_EXTRACTOR = None

def get_feature_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = DeepFeatureExtractor()
    return _EXTRACTOR

def extract_features(image_input) -> torch.Tensor:
    """
    Accepts either an image path (str/Path) or a PIL Image object.
    Passes leaf image through ResNet-50 and returns 2,048-dim visual features.
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_input}")
        image = Image.open(path).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        # Array or other format
        image = Image.fromarray(image_input).convert("RGB")

    image_tensor = TRANSFORM(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
    extractor = get_feature_extractor()
    features = extractor(image_tensor)           # Shape: (1, 2048)
    return features