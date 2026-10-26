"""
PlantVillage Hybrid ResNet-50 + SVM Training Pipeline
=====================================================
Executed on GPU (Google Colab / Kaggle T4 / A100):
1. Downloads and loads the PlantVillage foliar dataset (54,305 images across 38 classes).
2. Performs data augmentation (flips, affine rotations, color jitter, ImageNet normalization).
3. Fine-tunes ResNet-50 feature extractor with Cross-Entropy Loss & AdamW.
4. Extracts 2,048-dimensional dense visual feature vectors.
5. Fits the Support Vector Machine (SVM) decision hyperplanes.
6. Exports the calibrated model weights to models/plantvillage_resnet50.pth.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ── 1. Configuration & Hyperparameters ─────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
IMG_SIZE = 224
NUM_CLASSES = 38
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_MODEL_PATH = "models/plantvillage_resnet50.pth"
OUTPUT_SVM_PATH = "models/plantvillage_svm_model.pkl"

print(f"[*] Training Device: {DEVICE}")

# ── 2. Dataset Loading & Augmentation Pipeline ──────────────────────────────────
# In Colab: Download PlantVillage dataset via Kaggle API or direct URL:
# !pip install -q kagglehub
# import kagglehub
# dataset_path = kagglehub.dataset_download("emmarex/plantdisease")

DATASET_DIR = "dataset/plantvillage"  # Directory with 38 class folders

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_data(dataset_dir):
    """Loads dataset and splits into 80% train, 10% val, 10% test."""
    full_dataset = datasets.ImageFolder(root=dataset_dir)
    total_size = len(full_dataset)
    train_size = int(0.80 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms
    train_data.dataset.transform = train_transforms
    val_data.dataset.transform = val_transforms
    test_data.dataset.transform = val_transforms

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, full_dataset.classes

# ── 3. Stage 1: Fine-Tune ResNet-50 Backbone ────────────────────────────────────
def build_model(num_classes):
    """Initializes ResNet-50 with pre-trained ImageNet weights and replaces classifier."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Freeze lower base layers
    for param in list(model.parameters())[:-15]:
        param.requires_grad = False
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)

def train_feature_extractor(model, train_loader, val_loader, epochs=EPOCHS):
    """Trains ResNet-50 with AdamW and Label Smoothing Cross-Entropy."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {running_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            print(f"[*] New best weights saved to {OUTPUT_MODEL_PATH} ({val_acc*100:.2f}%)")

# ── 4. Stage 2: Feature Extraction + SVM Classifier ───────────────────────────
def extract_embeddings(model, dataloader):
    """Extracts 2,048-dim feature vectors from the penultimate pooling layer."""
    model.eval()
    # Strip terminal classification head to get pure feature embeddings
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(DEVICE)
    
    embeddings = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            feats = feature_extractor(images)
            feats = torch.flatten(feats, 1)  # Shape: (batch_size, 2048)
            embeddings.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    import numpy as np
    return np.vstack(embeddings), np.concatenate(labels_list)

def train_svm_classifier(train_features, train_labels, test_features, test_labels, save_path=OUTPUT_SVM_PATH):
    """Fits an RBF Support Vector Machine on 2,048 deep feature vectors and serializes to disk."""
    print("[*] Training SVM Classifier on 2,048-dimensional visual feature vectors...")
    svm = SVC(C=1.0, kernel='rbf', probability=True, random_state=42, cache_size=1000)
    svm.fit(train_features, train_labels)

    preds = svm.predict(test_features)
    acc = accuracy_score(test_labels, preds)
    print(f"[*] SVM Final Test Accuracy: {acc * 100:.2f}%")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(svm, save_path)
        print(f"[*] Saved trained RBF-SVM model to {save_path} ({os.path.getsize(save_path):,} bytes)")

    return svm

# ── 5. Main Execution Entrypoint ───────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        print(f"[Notice] Dataset path '{DATASET_DIR}' not found locally.")
        print("[Notice] In Colab: Download PlantVillage via Kaggle: `kagglehub.dataset_download('emmarex/plantdisease')`")
        sys.exit(0)

    train_loader, val_loader, test_loader, classes = load_data(DATASET_DIR)
    print(f"[*] Loaded 38 classes: {classes[:5]}...")

    # Train Deep Feature Extractor
    model = build_model(num_classes=len(classes))
    train_feature_extractor(model, train_loader, val_loader, epochs=EPOCHS)

    # Extract 2,048-dim vectors & Train SVM
    model.load_state_dict(torch.load(OUTPUT_MODEL_PATH))
    train_feats, train_y = extract_embeddings(model, train_loader)
    test_feats, test_y = extract_embeddings(model, test_loader)
    svm_clf = train_svm_classifier(train_feats, train_y, test_feats, test_y)
    print("[*] Training completed successfully!")
