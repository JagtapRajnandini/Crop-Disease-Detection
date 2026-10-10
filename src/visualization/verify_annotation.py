"""
verify_annotation.py
--------------------
Loads 1 image and its YOLO .txt label to visually verify bounding boxes.
"""

from pathlib import Path
import cv2
import json

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "plantdoc" / "images" / "train"
LABELS_DIR = PROJECT_ROOT / "data" / "plantdoc" / "labels" / "train"
LABELMAP_PATH = PROJECT_ROOT / "plant_doc_detection_dataset" / "plant_diseases" / "labelmap.json"

# Load class names (remapping ID: 1-29 in labelmap -> 0-28 in YOLO)
with open(LABELMAP_PATH, "r", encoding="utf-8") as f:
    raw_labelmap = json.load(f)

# class_id 0 was background, so YOLO class 0 = raw label 1, etc.
class_names = [item["object_name"] for item in raw_labelmap if item["label_id"] != 0]

# Pick the first available label file
label_files = list(LABELS_DIR.glob("*.txt"))
if not label_files:
    print("❌ No label files found in data/plantdoc/labels/train!")
    exit()

sample_label = label_files[0]
sample_image = IMAGES_DIR / f"{sample_label.stem}.jpg"

# Read image
img = cv2.imread(str(sample_image))
h, w, _ = img.shape

# Read and draw bounding boxes
with open(sample_label, "r", encoding="utf-8") as f:
    for line in f:
        cls_id, xc, yc, nw, nh = map(float, line.strip().split())
        cls_id = int(cls_id)

        # Denormalize coordinates to pixels
        xmin = int((xc - nw / 2) * w)
        ymin = int((yc - nh / 2) * h)
        xmax = int((xc + nw / 2) * w)
        ymax = int((yc + nh / 2) * h)

        # Draw green bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Draw label text
        label_text = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        cv2.putText(img, label_text, (xmin, max(20, ymin - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Save visual verification output
output_path = PROJECT_ROOT / "milestone1_verification.jpg"
cv2.imwrite(str(output_path), img)
print(f"✅ Verification image saved at: {output_path}")
print(f"🖼️ Image: {sample_image.name} | Detected Classes drawn.")