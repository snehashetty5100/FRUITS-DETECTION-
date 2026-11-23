"""Import and Validate Hand-Corrected Ground-Truth Masks

Reads corrected masks from dataset/annotation_done/:
1. Validates dimensions (must be exactly 224x224).
2. Strictly binarises (threshold > 128 -> 255, else 0).
3. Ensures letterbox padding areas are strictly 0 (background).
4. Saves verified mask to dataset/ground_truth_masks/.
5. Updates dataset/ground_truth_masks/manifest.csv (source=hand, verified=yes).
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import numpy as np
from PIL import Image

from src.preprocessing import load_image
from src.segmentation import get_valid_content_bbox

MANIFEST_PATH = Path("dataset/ground_truth_masks/manifest.csv")
GT_DIR = Path("dataset/ground_truth_masks")
PROCESSED_DIR = Path("dataset/processed")
DONE_DIR = Path("dataset/annotation_done")


def import_corrected_masks():
    if not DONE_DIR.exists():
        print(f"Directory {DONE_DIR} does not exist.")
        return

    corrected_files = sorted(DONE_DIR.glob("*_mask.png"))
    if not corrected_files:
        print(f"No corrected masks found in {DONE_DIR}. Drop corrected *_mask.png files there.")
        return

    # Load current manifest
    manifest_rows = []
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))

    manifest_map = {row["filename"]: row for row in manifest_rows}
    imported_count = 0

    print(f"=== Found {len(corrected_files)} corrected mask(s) to process ===")

    for mask_path in corrected_files:
        mask_filename = mask_path.name
        stem = mask_filename.replace("_mask.png", "")
        parts = stem.split("_")
        fruit, stage = parts[0], parts[1]

        # 1. Check corresponding image
        image_path = PROCESSED_DIR / fruit / stage / f"{stem}.png"
        if not image_path.exists():
            print(f"Error: Matching processed image not found for {mask_filename} at {image_path}. Skipping.")
            continue

        img_rgb = load_image(str(image_path))
        _, _, _, _, valid_mask = get_valid_content_bbox(img_rgb)

        # 2. Check dimensions
        with Image.open(mask_path) as m:
            w, h = m.size
            if (w, h) != (224, 224):
                print(f"Error: {mask_filename} has size {w}x{h}, expected 224x224. Skipping.")
                continue
            raw_mask = np.array(m.convert("L"))

        # 3. Binarise strictly
        binary_mask = (raw_mask > 128).astype(np.uint8) * 255

        # 4. Enforce that letterbox padding is strictly zero
        padding_violations = np.sum((binary_mask > 0) & (~valid_mask))
        if padding_violations > 0:
            print(f"  [!] Notice: Cleared {padding_violations} foreground pixels in letterbox padding for {mask_filename}.")
            binary_mask[~valid_mask] = 0

        # 5. Save to ground_truth_masks/
        dest_path = GT_DIR / mask_filename
        Image.fromarray(binary_mask).save(dest_path, "PNG")

        # 6. Update Manifest entry
        if mask_filename in manifest_map:
            manifest_map[mask_filename]["source"] = "hand"
            manifest_map[mask_filename]["verified"] = "yes"
        else:
            manifest_map[mask_filename] = {
                "filename": mask_filename,
                "source": "hand",
                "split": "heldout",  # Default new manual masks to heldout unless specified
                "verified": "yes"
            }

        imported_count += 1
        print(f"  [✓] Verified & imported: {mask_filename} (Foreground pixels: {np.sum(binary_mask > 0)})")

    # Write updated manifest
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "source", "split", "verified"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for filename in sorted(manifest_map.keys()):
            writer.writerow(manifest_map[filename])

    print(f"\nManifest successfully updated at {MANIFEST_PATH}. Total imported: {imported_count}.")


if __name__ == "__main__":
    import_corrected_masks()
