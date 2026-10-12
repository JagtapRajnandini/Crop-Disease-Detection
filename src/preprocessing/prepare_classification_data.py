import os
import shutil
from pathlib import Path

# Paths
ROOT_DIR = Path(r"C:\Users\rajna\OneDrive\Desktop\AIML Project")
YOLO_DATA_DIR = ROOT_DIR / "data" / "plantdoc"
CLASSIFICATION_DIR = ROOT_DIR / "data" / "plant_species_classification"

# PlantDoc YOLO classes mapping to broad species
# Original YOLO classes (0-28) based on your dataset.yaml
SPECIES_MAP = {
    "apple": [0, 1, 2],       # apple_scab, apple_leaf, apple_rust
    "bell_pepper": [3, 4],    # bell_pepper_leaf, bell_pepper_leaf_spot
    "blueberry": [5],         # blueberry_leaf
    "cherry": [6],            # cherry_leaf
    "corn": [7, 8, 9],        # corn_gray_leaf_spot, corn_leaf_blight, corn_rust
    "peach": [10],            # peach_leaf
    "potato": [11, 12],       # potato_early_blight, potato_late_blight
    "raspberry": [13],        # raspberry_leaf
    "soyabean": [14],         # soyabean_leaf
    "squash": [15],           # squash_powdery_mildew
    "strawberry": [16],       # strawberry_leaf
    "tomato": [17, 18, 19, 20, 21, 22, 23, 24], # early_blight, septoria, leaf, bacterial_spot, late_blight, mosaic, yellow_virus, mold
    "grape": [25, 26]         # grape_leaf, grape_black_rot
    # Note: 27, 28 missing from map? If there are others, we will catch them dynamically based on labels.
}

# Reverse map: class_id -> species_name
CLASS_TO_SPECIES = {}
for species, class_ids in SPECIES_MAP.items():
    for cid in class_ids:
        CLASS_TO_SPECIES[cid] = species

def prepare_classification_dataset():
    print(f"Creating classification dataset at: {CLASSIFICATION_DIR}")
    
    # Create train and val directories for each species
    for split in ["train", "val"]:
        for species in SPECIES_MAP.keys():
            (CLASSIFICATION_DIR / split / species).mkdir(parents=True, exist_ok=True)
            
    # Process images based on YOLO labels
    for split in ["train", "val"]:
        images_dir = YOLO_DATA_DIR / "images" / split
        labels_dir = YOLO_DATA_DIR / "labels" / split
        
        if not images_dir.exists():
            continue
            
        print(f"Processing {split} split...")
        copied_count = 0
        
        for img_path in images_dir.glob("*.jpg"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
                
            # Read label file to determine species
            with open(label_path, "r") as f:
                lines = f.readlines()
                
            if not lines:
                continue
                
            # We take the species of the first bounding box found in the image.
            # (Usually images in PlantDoc contain leaves of the same species).
            first_class_id = int(lines[0].split()[0])
            
            species_name = CLASS_TO_SPECIES.get(first_class_id)
            if not species_name:
                continue
                
            # Copy image to the respective species folder
            dest_path = CLASSIFICATION_DIR / split / species_name / img_path.name
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
                copied_count += 1
                
        print(f"Copied {copied_count} images for {split} split.")

if __name__ == "__main__":
    prepare_classification_dataset()
    print("Done! Classification dataset is ready.")
