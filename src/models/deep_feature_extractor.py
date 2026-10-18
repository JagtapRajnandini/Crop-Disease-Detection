"""
deep_feature_extractor.py
---------------------------
Stage 1: Deep Learning Feature Extraction Module (Fine-Tuned PyTorch ResNet-50).
Extracts 2,048-dimensional high-level latent embeddings directly from the
global average pooling layer of the fine-tuned PlantVillage ResNet-50 network.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import ResNetConfig, ResNetForImageClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_WEIGHTS_PATH = PROJECT_ROOT / "models" / "plantvillage_resnet50.pth"
CONFIG_PATH = PROJECT_ROOT / "models" / "config.json"

# Standard ImageNet normalization pipeline for 224x224 ResNet-50 input tensors
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DeepFeatureExtractor(nn.Module):
    """Fine-tuned ResNet-50 backbone extracting 2,048-D deep semantic feature embeddings."""

    def __init__(self):
        super().__init__()
        if CONFIG_PATH.exists() and MODEL_WEIGHTS_PATH.exists():
            self.config = ResNetConfig.from_pretrained(str(CONFIG_PATH))
            self.full_model = ResNetForImageClassification(self.config)
            state_dict = torch.load(str(MODEL_WEIGHTS_PATH), map_location="cpu")
            self.full_model.load_state_dict(state_dict)
            self.id2label = self.config.id2label
        else:
            raise FileNotFoundError(f"Model artifacts missing: {MODEL_WEIGHTS_PATH} or {CONFIG_PATH}")

        self.full_model.eval()

    def forward(self, x):
        """Passes input tensor through backbone and extracts (Batch, 2048) pooling vector."""
        with torch.no_grad():
            out = self.full_model.resnet(x)
            # Global Average Pooling output: (Batch, 2048)
            features = out.pooler_output.squeeze(-1).squeeze(-1)
            return features

    def classify_features(self, features: torch.Tensor):
        """Evaluates 2,048-D feature embeddings through the classification head."""
        with torch.no_grad():
            logits = self.full_model.classifier(features)
            return logits

# Singleton instance
_EXTRACTOR = None

def get_feature_extractor() -> DeepFeatureExtractor:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = DeepFeatureExtractor()
    return _EXTRACTOR

def extract_features(image_input) -> torch.Tensor:
    """
    Accepts either an image path (str/Path) or a PIL Image object.
    Returns 2,048-dim latent visual feature embedding tensor of shape (1, 2048).
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_input}")
        image = Image.open(path).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.fromarray(image_input).convert("RGB")

    image_tensor = TRANSFORM(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
    extractor = get_feature_extractor()
    features = extractor(image_tensor)           # Shape: (1, 2048)
    return features