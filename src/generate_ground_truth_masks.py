"""Ground-Truth Mask Annotations Generator for Fruits-360

Generates binary segmentation masks for Fruits-360 dataset images
to enable quantitative evaluation using Intersection over Union (IoU).

In Fruits-360, images are photographed on pure white background [255, 255, 255].
Foreground fruit pixels are extracted by thresholding against the white background.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import scipy.ndimage as ndi

from src.preprocessing import load_image

FRUITS360_DIR = Path("dataset/fruits360")
MASKS_DIR = Path("dataset/fruits360_masks")
MASKS_DIR.mkdir(parents=True, exist_ok=True)


def create_ground_truth_mask_for_fruits360(image_path: Path, out_path: Path):
    """Generate high-precision ground-truth mask for a Fruits-360 image."""
    if not image_path.exists():
        print(f"Warning: Image {image_path} not found.")
        return None

    img_rgb = load_image(str(image_path))
    # Fruits-360 uses white background [255, 255, 255]
    # Foreground pixels are non-white (any channel < 250)
    fg_mask = np.any(img_rgb < 250, axis=2)

    # Clean with binary morphology
    structure = ndi.generate_binary_structure(2, 2)
    fg_mask = ndi.binary_fill_holes(fg_mask)
    fg_mask = ndi.binary_opening(fg_mask, structure=structure, iterations=1)
    fg_mask = ndi.binary_closing(fg_mask, structure=structure, iterations=1)
    fg_mask = ndi.binary_fill_holes(fg_mask)

    # Save 8-bit PNG mask
    mask_uint8 = (fg_mask.astype(np.uint8)) * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_uint8).save(out_path, "PNG")
    return out_path


def generate_sample_masks(num_per_class: int = 5):
    """Generate reference masks for target classes in Fruits-360 Training split."""
    training_dir = FRUITS360_DIR / "Training"
    if not training_dir.exists():
        print(f"Error: {training_dir} does not exist.")
        return

    target_keywords = ["apple", "banana", "mango", "orange", "papaya", "tomato"]
    count = 0

    for cls_dir in sorted(training_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_lower = cls_dir.name.lower()
        if any(kw in cls_lower for kw in target_keywords):
            images = sorted(cls_dir.glob("*.jpg"))[:num_per_class]
            for img_p in images:
                out_p = MASKS_DIR / cls_dir.name / f"{img_p.stem}_mask.png"
                create_ground_truth_mask_for_fruits360(img_p, out_p)
                count += 1

    print(f"Generated {count} ground-truth masks in {MASKS_DIR}")


if __name__ == "__main__":
    generate_sample_masks()
