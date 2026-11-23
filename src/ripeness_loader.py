"""Dataset loader for the Kaggle Ripeness Dataset.

Reads images through dataset/ripeness_manifest.csv or dataset/ripeness_splits.csv,
and ensures all images are converted to 3-channel RGB (compositing RGBA onto white).
Does not use cv2 or pandas.
"""

import os
import csv
from typing import List, Dict, Optional, Tuple, Generator
import numpy as np
from PIL import Image


def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Load an image and guarantee 3-channel RGB mode.
    
    If the image has an alpha channel (RGBA), it is composited onto
    a solid white background to avoid dark/black transparency artifacts.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA") or ("A" in img.mode):
            # Alpha composite onto solid white background
            alpha = img.split()[-1]
            bg = Image.new("RGB", img.size, (255, 255, 255))
            # Paste RGB representation masked by alpha
            bg.paste(img.convert("RGB"), mask=alpha)
            result = bg
        elif img.mode != "RGB":
            result = img.convert("RGB")
        else:
            result = img.copy()

    if target_size is not None:
        result = result.resize(target_size, Image.Resampling.LANCZOS)

    return result


def load_image_array(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False
) -> np.ndarray:
    """Load an image as a NumPy RGB array of shape (H, W, 3).
    
    If normalize=True, returns float32 in [0.0, 1.0]; otherwise uint8 in [0, 255].
    """
    img = load_image(image_path, target_size=target_size)
    arr = np.array(img)
    if normalize:
        arr = arr.astype(np.float32) / 255.0
    return arr


def load_manifest(
    manifest_path: str = "dataset/ripeness_manifest.csv",
    fruit: Optional[str] = None,
    stage: Optional[str] = None,
    original_split: Optional[str] = None,
    deduplicate: bool = False,
    official_test_clean_only: bool = False,
) -> List[Dict[str, str]]:
    """Read the full ripeness manifest with optional filters.
    
    Parameters:
        manifest_path: Path to dataset/ripeness_manifest.csv
        fruit: Filter by fruit name (apple, banana, mango, orange, tomato)
        stage: Filter by stage (unripe, ripe, overripe)
        original_split: Filter by original split ('Train' or 'Test')
        deduplicate: If True, keep only non-duplicate images (is_duplicate == 'false')
        official_test_clean_only: If True, keep only clean official Test images
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if fruit and row["fruit"].lower() != fruit.lower():
                continue
            if stage and row["stage"].lower() != stage.lower():
                continue
            if original_split and row["original_split"].lower() != original_split.lower():
                continue
            if deduplicate and row.get("is_duplicate", "false").lower() == "true":
                continue
            if official_test_clean_only and row.get("official_test_clean", "false").lower() != "true":
                continue
            records.append(row)

    return records


def load_splits(
    splits_path: str = "dataset/ripeness_splits.csv",
    split: Optional[str] = None,
    fruit: Optional[str] = None,
    stage: Optional[str] = None,
    official_test_clean_only: bool = False,
) -> List[Dict[str, str]]:
    """Read the deduplicated stratified splits with optional filters.
    
    Parameters:
        splits_path: Path to dataset/ripeness_splits.csv
        split: Filter by assigned split ('train', 'val', or 'test')
        fruit: Filter by fruit name
        stage: Filter by stage (unripe, ripe, overripe)
        official_test_clean_only: If True, keep only clean official Test images
    """
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Splits file not found: {splits_path}")

    records = []
    with open(splits_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split and row.get("split", "").lower() != split.lower():
                continue
            if fruit and row["fruit"].lower() != fruit.lower():
                continue
            if stage and row["stage"].lower() != stage.lower():
                continue
            if official_test_clean_only and row.get("official_test_clean", "false").lower() != "true":
                continue
            records.append(row)

    return records


class RipenessDataset:
    """Convenience dataset wrapper for loading images and metadata."""

    def __init__(
        self,
        splits_path: str = "dataset/ripeness_splits.csv",
        split: Optional[str] = None,
        fruit: Optional[str] = None,
        stage: Optional[str] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        self.records = load_splits(
            splits_path=splits_path,
            split=split,
            fruit=fruit,
            stage=stage,
        )
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, str]]:
        rec = self.records[idx]
        img = load_image(rec["path"], target_size=self.target_size)
        return img, rec

    def get_array(self, idx: int, normalize: bool = False) -> Tuple[np.ndarray, Dict[str, str]]:
        rec = self.records[idx]
        arr = load_image_array(rec["path"], target_size=self.target_size, normalize=normalize)
        return arr, rec


if __name__ == "__main__":
    print("Testing RipenessLoader...")

    # 1. Test reading manifest and splits
    manifest = load_manifest()
    splits = load_splits()
    print(f"Loaded manifest rows: {len(manifest)}")
    print(f"Loaded splits rows: {len(splits)}")

    # 2. Test RGBA image conversion specifically
    rgba_candidates = [r for r in manifest if r["mode"] == "RGBA"]
    print(f"Found {len(rgba_candidates)} RGBA candidates in manifest.")
    if rgba_candidates:
        test_rgba_path = rgba_candidates[0]["path"]
        print(f"Testing RGBA conversion on: {test_rgba_path}")
        with Image.open(test_rgba_path) as orig:
            print(f"  Original mode: {orig.mode}, size: {orig.size}")
        converted = load_image(test_rgba_path)
        print(f"  Converted mode: {converted.mode}, size: {converted.size}")
        arr = load_image_array(test_rgba_path)
        print(f"  NumPy array shape: {arr.shape}, dtype: {arr.dtype}")
        assert converted.mode == "RGB", "Converted image must be RGB"
        assert arr.shape == (300, 300, 3), "Array must be (300, 300, 3)"
        print("  RGBA -> RGB compositing test PASSED!")

    # 3. Test splits loader filtering
    for sp in ["train", "val", "test"]:
        sub = load_splits(split=sp)
        print(f"Split '{sp}' count: {len(sub)}")

    clean_test = load_splits(official_test_clean_only=True)
    print(f"Official test clean count in splits: {len(clean_test)}")

    # 4. Test RipenessDataset class
    ds = RipenessDataset(split="test")
    img, rec = ds[0]
    print(f"First test sample: fruit={rec['fruit']}, stage={rec['stage']}, img_size={img.size}")
    print("RipenessLoader tests fully PASSED!")
