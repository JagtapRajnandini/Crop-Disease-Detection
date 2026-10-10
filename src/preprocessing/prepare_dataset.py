"""
prepare_dataset.py
------------------
Converts the raw plant_doc_detection_dataset into YOLO-ready format.

What this script does:
  1. Reads train.txt / val.txt / test.txt to get the official image splits.
  2. For each image stem in each split:
       a. Locates the image  →  plant_diseases/images/{stem}.jpg
       b. Locates the CSV    →  plant_diseases/csv/{stem}.csv
       c. Skips the image if no matching CSV annotation exists.
       d. Reads image dimensions (needed for coordinate normalization).
       e. Converts CSV pixel coordinates → YOLO normalized format.
       f. Copies the image   →  data/plantdoc/images/{split}/{stem}.jpg
       g. Writes label file  →  data/plantdoc/labels/{split}/{stem}.txt
  3. Prints a per-split and grand-total summary.

CSV format (input):
    #item, x, y, width, height, label
    x, y   = top-left corner in pixels
    label  = integer class ID  (0 = background, 1–29 = plant conditions)

YOLO .txt format (output, one line per bounding box):
    class_id  x_center  y_center  width  height
    All values normalized to [0.0, 1.0] relative to image dimensions.
    class_id remapping:  CSV label 1–29  →  YOLO 0–28  (background dropped)
"""

import csv
import shutil
from pathlib import Path

from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
# Script is at: AIML Project/src/preprocessing/prepare_dataset.py
# Project root: AIML Project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_DIR    = PROJECT_ROOT / "plant_doc_detection_dataset" / "plant_diseases"
IMAGES_SRC = RAW_DIR / "images"
CSV_SRC    = RAW_DIR / "csv"
SETS_DIR   = RAW_DIR / "sets"

OUT_DIR    = PROJECT_ROOT / "data" / "plantdoc"

SPLITS = {
    "train": SETS_DIR / "train.txt",
    "val":   SETS_DIR / "val.txt",
    "test":  SETS_DIR / "test.txt",
}

BACKGROUND_LABEL = 0   # label 0 in CSV = background, always skipped


# ── Helpers ────────────────────────────────────────────────────────────────────

def read_split(txt_path: Path) -> list[str]:
    """Return list of image stems from a split .txt file."""
    return [
        line.strip()
        for line in txt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def csv_to_yolo_lines(csv_path: Path, img_w: int, img_h: int) -> list[str]:
    """
    Read one CSV annotation file and return YOLO-format strings.

    Conversion formulas:
        x_center = (x + w/2) / img_w
        y_center = (y + h/2) / img_h
        norm_w   = w / img_w
        norm_h   = h / img_h
        yolo_class = csv_label - 1       (drops background, remaps 1-29 → 0-28)
    """
    lines = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(
            f,
            fieldnames=["item", "x", "y", "width", "height", "label"],
        )
        for row in reader:
            # Skip the header comment row
            if str(row["item"]).startswith("#"):
                continue

            label = int(float(row["label"]))

            # Skip background annotations
            if label == BACKGROUND_LABEL:
                continue

            x = float(row["x"])
            y = float(row["y"])
            w = float(row["width"])
            h = float(row["height"])

            # Convert pixel coords → YOLO normalized format
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w   = w / img_w
            norm_h   = h / img_h

            # Clamp to [0.0, 1.0] — handles any annotations slightly outside borders
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w   = max(0.0, min(1.0, norm_w))
            norm_h   = max(0.0, min(1.0, norm_h))

            yolo_class = label - 1   # remap: CSV 1–29 → YOLO 0–28

            lines.append(
                f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

    return lines


def process_split(split_name: str, stems: list[str]) -> dict:
    """
    Process all image stems for one split (train / val / test).
    Returns a stats dictionary.
    """
    img_out = OUT_DIR / "images" / split_name
    lbl_out = OUT_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = {
        "processed":   0,
        "no_image":    0,   # image file missing from images/ folder
        "no_csv":      0,   # no matching CSV annotation found
        "empty_label": 0,   # CSV existed but had only background rows
    }

    for stem in stems:
        img_src = IMAGES_SRC / f"{stem}.jpg"
        csv_src = CSV_SRC    / f"{stem}.csv"

        # 1. Check image file exists
        if not img_src.exists():
            stats["no_image"] += 1
            continue

        # 2. Check annotation CSV exists
        if not csv_src.exists():
            stats["no_csv"] += 1
            continue

        # 3. Read image dimensions (required for normalization)
        with Image.open(img_src) as img:
            img_w, img_h = img.size

        # 4. Convert CSV annotations → YOLO lines
        yolo_lines = csv_to_yolo_lines(csv_src, img_w, img_h)

        if not yolo_lines:
            # CSV had entries but all were background rows — skip
            stats["empty_label"] += 1
            continue

        # 5. Copy image to output split folder
        shutil.copy2(img_src, img_out / f"{stem}.jpg")

        # 6. Write YOLO label .txt file
        (lbl_out / f"{stem}.txt").write_text(
            "\n".join(yolo_lines), encoding="utf-8"
        )

        stats["processed"] += 1

    return stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PlantDoc  →  YOLO Format Preprocessing")
    print("=" * 60)

    grand_total = 0

    for split_name, txt_path in SPLITS.items():
        stems = read_split(txt_path)
        print(f"\n[{split_name.upper()}]  {len(stems)} entries in split list")

        stats = process_split(split_name, stems)

        print(f"  ✅  Written        : {stats['processed']} images + label files")
        print(f"  ⚠️   No image file  : {stats['no_image']}  (skipped)")
        print(f"  ⚠️   No CSV found   : {stats['no_csv']}  (skipped — no annotation)")
        print(f"  ⚠️   Empty label    : {stats['empty_label']}  (skipped — background only)")

        grand_total += stats["processed"]

    print("\n" + "=" * 60)
    print(f"  Total images written : {grand_total}")
    print(f"  Output location      : {OUT_DIR}")
    print("  Folder layout:")
    print("    data/plantdoc/images/{train, val, test}/")
    print("    data/plantdoc/labels/{train, val, test}/")
    print("=" * 60)


if __name__ == "__main__":
    main()