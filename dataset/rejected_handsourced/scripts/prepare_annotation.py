"""Annotation Preparation Utility

For each sample in the ground-truth manifest:
1. Generates a visual review overlay (auto-mask outline on top of the original image).
2. Copies the raw auto-mask template for easy editing with external tools (GIMP, Paint, CVAT, LabelMe).
3. Saves output files into dataset/annotation_queue/.
4. Ensures dataset/annotation_done/ is ready for corrected masks.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import numpy as np
from PIL import Image
import scipy.ndimage as ndi

from src.preprocessing import load_image

MANIFEST_PATH = Path("dataset/ground_truth_masks/manifest.csv")
GT_DIR = Path("dataset/ground_truth_masks")
PROCESSED_DIR = Path("dataset/processed")
QUEUE_DIR = Path("dataset/annotation_queue")
DONE_DIR = Path("dataset/annotation_done")

QUEUE_DIR.mkdir(parents=True, exist_ok=True)
DONE_DIR.mkdir(parents=True, exist_ok=True)


def create_outline_overlay(image_rgb: np.ndarray, mask_bin: np.ndarray, outline_color=(0, 255, 255)) -> np.ndarray:
    """Draw a 2-pixel contour boundary of the mask directly on top of the RGB image."""
    bool_mask = mask_bin > 0
    # Morphological gradient gives the perimeter/boundary
    structure = ndi.generate_binary_structure(2, 1)
    dilated = ndi.binary_dilation(bool_mask, structure=structure, iterations=2)
    eroded = ndi.binary_erosion(bool_mask, structure=structure, iterations=1)
    outline = dilated & ~eroded

    overlay = image_rgb.copy()
    overlay[outline] = outline_color
    return overlay


def prepare_queue():
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    print(f"=== Preparing Annotation Queue for {len(reader)} Samples ===")

    for row in reader:
        mask_filename = row["filename"]
        stem = mask_filename.replace("_mask.png", "")
        parts = stem.split("_")
        fruit, stage = parts[0], parts[1]

        image_path = PROCESSED_DIR / fruit / stage / f"{stem}.png"
        mask_path = GT_DIR / mask_filename

        if not image_path.exists() or not mask_path.exists():
            print(f"Warning: Missing file for {stem} (Image: {image_path.exists()}, Mask: {mask_path.exists()})")
            continue

        img_rgb = load_image(str(image_path))
        with Image.open(mask_path) as m:
            mask_arr = np.array(m.convert("L"))

        # Generate boundary review overlay
        overlay = create_outline_overlay(img_rgb, mask_arr, outline_color=(0, 255, 255))

        # Side-by-side review sheet: [Image + Outline, Mask Template]
        sheet = Image.new("RGB", (224 * 2, 224))
        sheet.paste(Image.fromarray(overlay), (0, 0))
        sheet.paste(Image.fromarray(mask_arr).convert("RGB"), (224, 0))

        review_path = QUEUE_DIR / f"{stem}_review.png"
        sheet.save(review_path, "PNG")

        # Also place the editable mask template into queue
        editable_mask_path = QUEUE_DIR / f"{stem}_mask.png"
        Image.fromarray(mask_arr).save(editable_mask_path, "PNG")

        print(f"  [+] Queued: {stem} -> {review_path.name} & {editable_mask_path.name}")

    print("\nAnnotation Queue prepared successfully in 'dataset/annotation_queue/'.")
    print("Instructions:")
    print("1. Review each sample in dataset/annotation_queue/.")
    print("2. Edit {sample}_mask.png using GIMP, Paint, or any editor (white = fruit, black = background).")
    print("3. Save corrected masks into dataset/annotation_done/ as {sample}_mask.png.")
    print("4. Run 'python src/import_corrected_masks.py' to ingest and update the manifest.")


if __name__ == "__main__":
    prepare_queue()
