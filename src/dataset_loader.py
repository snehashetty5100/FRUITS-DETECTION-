"""Dataset Preparation and Validation Utilities

Resilient module to:
1. Verify dataset structure across all 6 fruits and 3 ripeness stages.
2. Scan and check image file integrity (detecting corrupt or unreadable files).
3. Generate dataset summary statistics (class distribution and balance).
4. Standardize and process raw images into dataset/processed/ with resizing.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

# Supported fruits and ripeness stages
SUPPORTED_FRUITS = ["apple", "banana", "mango", "orange", "papaya", "tomato"]
RIPENESS_STAGES = ["unripe", "ripe", "overripe"]


def scan_dataset(base_dir: str = "dataset/raw") -> List[Dict[str, any]]:
    """Scan dataset directory and verify valid images per category using Pillow."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    records = []

    for fruit in SUPPORTED_FRUITS:
        for stage in RIPENESS_STAGES:
            stage_dir = Path(base_dir) / fruit / stage
            valid_count = 0
            corrupt_count = 0

            if stage_dir.exists():
                for file_path in stage_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                        try:
                            with Image.open(file_path) as img:
                                img.verify()
                            valid_count += 1
                        except Exception:
                            corrupt_count += 1

            records.append({
                "Fruit": fruit.capitalize(),
                "Ripeness Stage": stage.capitalize(),
                "Valid Images": valid_count,
                "Corrupt/Invalid": corrupt_count,
                "Path": str(stage_dir)
            })

    return records


def print_summary(base_dir: str = "dataset/raw"):
    """Display tabular breakdown of dataset."""
    records = scan_dataset(base_dir)
    print(f"\n{'Fruit':<10} | {'Ripeness Stage':<15} | {'Valid Images':<12} | {'Directory'}")
    print("-" * 75)
    total_valid = 0
    for r in records:
        print(f"{r['Fruit']:<10} | {r['Ripeness Stage']:<15} | {r['Valid Images']:<12} | {r['Path']}")
        total_valid += r["Valid Images"]
    print("-" * 75)
    print(f"Total Valid Images in '{base_dir}': {total_valid}\n")


def process_raw_dataset(
    raw_dir: str = "dataset/raw",
    processed_dir: str = "dataset/processed",
    target_size: Tuple[int, int] = (224, 224)
) -> Dict[str, int]:
    """Process raw images: open, resize to standard dimensions, and save to processed directory."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    processed_counts = {}

    for fruit in SUPPORTED_FRUITS:
        for stage in RIPENESS_STAGES:
            src_stage_dir = Path(raw_dir) / fruit / stage
            dest_stage_dir = Path(processed_dir) / fruit / stage
            dest_stage_dir.mkdir(parents=True, exist_ok=True)

            key = f"{fruit}_{stage}"
            processed_counts[key] = 0

            if not src_stage_dir.exists():
                continue

            for file_path in src_stage_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    try:
                        with Image.open(file_path) as img:
                            rgb_img = img.convert("RGB")
                            resized = rgb_img.resize(target_size, Image.Resampling.LANCZOS)
                            out_filename = f"{file_path.stem}.png"
                            dest_file = dest_stage_dir / out_filename
                            resized.save(dest_file, "PNG")
                            processed_counts[key] += 1
                    except Exception as e:
                        print(f"Skipping corrupt image {file_path}: {e}")

    return processed_counts


if __name__ == "__main__":
    print_summary("dataset/raw")
    print("Processing raw images into dataset/processed (224x224)...")
    counts = process_raw_dataset()
    print("\nProcessed images summary:")
    print_summary("dataset/processed")
