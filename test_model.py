"""
test_model.py
-------------
Evaluates the Two-Stage Pipeline (ResNet18 + YOLOv8) on all 231 test images
from the PlantDoc dataset. Compares predicted disease class against ground truth
annotation labels and prints a full report with accuracy percentages.

Usage:
    .\venv\Scripts\python.exe test_model.py
"""

import json
import urllib.parse
from pathlib import Path
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent
DATASET_ROOT   = PROJECT_ROOT / "plant_doc_detection_dataset" / "plant_diseases"
IMAGES_DIR     = DATASET_ROOT / "images"
JSON_DIR       = DATASET_ROOT / "json"
TEST_LIST_FILE = DATASET_ROOT / "sets" / "test.txt"

# ── Import inference engine ────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.inference.predict_two_stage import predict_image

# ── Helper: load ground truth from per-image JSON ─────────────────────────────
def load_ground_truth(image_stem: str):
    """
    Returns a set of ground truth disease class names (e.g. {'tomato_septoria_leaf_spot'})
    from the PlantDoc annotation JSON for the given image stem.
    Returns empty set if annotation not found or has no categories.
    """
    json_path = JSON_DIR / f"{image_stem}.json"
    if not json_path.exists():
        return set(), None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set(), None

    # categories can be a single dict or a list depending on the file
    cats = data.get("categories", [])
    if isinstance(cats, dict):
        cats = [cats]

    gt_labels = set()
    img_filename = None

    images = data.get("images", [])
    if images:
        img_filename = images[0].get("file_name", None)

    for cat in cats:
        name = cat.get("name", "").strip()
        if name:
            gt_labels.add(name.lower())

    return gt_labels, img_filename


# ── Matching logic ─────────────────────────────────────────────────────────────
def is_plant_match(pred_plant: str, gt_labels: set) -> bool:
    """True if the predicted plant species prefix matches any GT label prefix."""
    pred_prefix = pred_plant.lower().replace(" ", "_")
    for gt in gt_labels:
        if gt.startswith(pred_prefix) or pred_prefix in gt:
            return True
    return False


def is_full_match(pred_detections: list, gt_labels: set) -> bool:
    """
    True if at least one predicted YOLO class exactly matches
    (or is contained in) a ground truth label.
    """
    for det in pred_detections:
        pred_cls = det["class"].lower().replace(" ", "_")
        for gt in gt_labels:
            if pred_cls == gt or pred_cls in gt or gt in pred_cls:
                return True
    return False


# ── Main Evaluation Loop ───────────────────────────────────────────────────────
def run_evaluation():
    print("=" * 70)
    print("  AgriScan AI — Two-Stage Model Evaluation on PlantDoc Test Set")
    print("=" * 70)

    if not TEST_LIST_FILE.exists():
        print(f"[ERROR] Test list not found: {TEST_LIST_FILE}")
        return

    test_stems = [line.strip() for line in TEST_LIST_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(test_stems)
    print(f"\n  Total test images listed : {total}")

    results = []
    skipped = 0

    for i, stem in enumerate(test_stems, 1):
        # Load ground truth
        gt_labels, img_filename = load_ground_truth(stem)

        if not gt_labels or not img_filename:
            skipped += 1
            continue

        # Decode URL-encoded filename (e.g. %2320 → #20)
        img_filename_decoded = urllib.parse.unquote(img_filename)
        img_path = IMAGES_DIR / img_filename_decoded

        if not img_path.exists():
            skipped += 1
            continue

        # Run inference
        try:
            image = Image.open(img_path).convert("RGB")
            result = predict_image(image, conf_threshold=0.25, use_two_stage=True)
        except Exception as e:
            skipped += 1
            continue

        # Evaluate
        pred_plant      = result.get("primary_plant", "Unknown")
        pred_detections = result.get("detections", [])
        pred_condition  = result.get("primary_condition", "")

        plant_ok = is_plant_match(pred_plant, gt_labels)
        full_ok  = is_full_match(pred_detections, gt_labels)

        status = "✅ FULL MATCH" if full_ok else ("🟡 PLANT OK" if plant_ok else "❌ WRONG")

        results.append({
            "index":        i,
            "image":        img_filename_decoded,
            "gt":           ", ".join(sorted(gt_labels)),
            "pred_plant":   pred_plant,
            "pred_disease": pred_condition,
            "plant_ok":     plant_ok,
            "full_ok":      full_ok,
            "status":       status,
        })

        # Live progress every 20 images
        if i % 20 == 0 or i == total:
            done = len(results)
            print(f"  [{i}/{total}] Processed {done} valid | Skipped {skipped}")

    # ── Summary Stats ──────────────────────────────────────────────────────────
    evaluated  = len(results)
    full_match = sum(1 for r in results if r["full_ok"])
    plant_only = sum(1 for r in results if r["plant_ok"] and not r["full_ok"])
    wrong      = sum(1 for r in results if not r["plant_ok"])

    full_pct   = (full_match / evaluated * 100) if evaluated else 0
    plant_pct  = (( full_match + plant_only) / evaluated * 100) if evaluated else 0
    wrong_pct  = (wrong / evaluated * 100) if evaluated else 0

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Total test images listed   : {total}")
    print(f"  Skipped (no annotation)    : {skipped}")
    print(f"  Evaluated                  : {evaluated}")
    print("-" * 70)
    print(f"  ✅ Full Match  (disease OK) : {full_match:3d} / {evaluated}  →  {full_pct:.1f}%")
    print(f"  🟡 Plant Only  (species OK) : {full_match + plant_only:3d} / {evaluated}  →  {plant_pct:.1f}%")
    print(f"  ❌ Wrong       (species bad): {wrong:3d} / {evaluated}  →  {wrong_pct:.1f}%")
    print("=" * 70)

    # ── Detailed per-image log ─────────────────────────────────────────────────
    print("\n  DETAILED LOG (Wrong predictions only)\n" + "-" * 70)
    for r in results:
        if not r["full_ok"]:
            print(f"  {r['status']}")
            print(f"    Image      : {r['image']}")
            print(f"    Ground Truth: {r['gt']}")
            print(f"    Predicted  : plant={r['pred_plant']}  |  disease={r['pred_disease']}")
            print()

    # ── Save full CSV report ───────────────────────────────────────────────────
    csv_path = PROJECT_ROOT / "test_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,image,ground_truth,pred_plant,pred_disease,plant_correct,full_match,status\n")
        for r in results:
            f.write(
                f"{r['index']},\"{r['image']}\",\"{r['gt']}\","
                f"\"{r['pred_plant']}\",\"{r['pred_disease']}\","
                f"{r['plant_ok']},{r['full_ok']},\"{r['status']}\"\n"
            )
    print(f"  Full CSV report saved → {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()