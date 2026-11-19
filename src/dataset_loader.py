"""Dataset Preparation and Validation Utilities

This module provides tools to:
1. Verify dataset structure across all 6 fruits and 3 ripeness stages.
2. Scan and check image file integrity (detecting corrupt or unreadable files).
3. Generate dataset summary statistics (class distribution and balance).
4. Standardize and process raw images into dataset/processed/ with resizing.
5. Create stratified train/val/test splits.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import pandas as pd
from PIL import Image

# Supported fruits and ripeness stages
SUPPORTED_FRUITS = ["apple", "banana", "mango", "orange", "papaya", "tomato"]
RIPENESS_STAGES = ["unripe", "ripe", "overripe"]


def scan_dataset(base_dir: str = "dataset/raw") -> pd.DataFrame:
    """Scan the dataset directory and count valid images per fruit and ripeness stage.

    Args:
        base_dir (str): Base directory ('dataset/raw' or 'dataset/processed').

    Returns:
        pd.DataFrame: Table with Fruit, Stage, Valid Image Count, and Corrupt Count.
    """
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
                        # Test if image is readable
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

    return pd.DataFrame(records)


def get_dataset_summary(base_dir: str = "dataset/raw") -> Dict[str, any]:
    """Calculate overall dataset metrics including total counts and class balance.

    Args:
        base_dir (str): Target dataset directory.

    Returns:
        dict: High-level metrics dictionary.
    """
    df = scan_dataset(base_dir)
    total_images = int(df["Valid Images"].sum())
    total_corrupt = int(df["Corrupt/Invalid"].sum())

    fruit_totals = df.groupby("Fruit")["Valid Images"].sum().to_dict()
    stage_totals = df.groupby("Ripeness Stage")["Valid Images"].sum().to_dict()

    return {
        "total_valid_images": total_images,
        "total_corrupt_files": total_corrupt,
        "images_per_fruit": fruit_totals,
        "images_per_stage": stage_totals,
        "summary_table": df
    }


def process_raw_dataset(
    raw_dir: str = "dataset/raw",
    processed_dir: str = "dataset/processed",
    target_size: Tuple[int, int] = (224, 224)
) -> Dict[str, int]:
    """Process raw images: read, resize to standard dimensions, and save to processed directory.

    Args:
        raw_dir (str): Source directory containing raw images.
        processed_dir (str): Destination directory for processed images.
        target_size (tuple): Output (width, height). Defaults to (224, 224).

    Returns:
        dict: Summary of processed image counts.
    """
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
                    # Read image with OpenCV
                    img_bgr = cv2.imread(str(file_path))
                    if img_bgr is None:
                        continue

                    # Resize to target resolution
                    resized_bgr = cv2.resize(
                        img_bgr, target_size, interpolation=cv2.INTER_AREA
                    )

                    # Save as standard PNG in processed directory
                    out_filename = f"{file_path.stem}.png"
                    dest_file = dest_stage_dir / out_filename
                    cv2.imwrite(str(dest_file), resized_bgr)
                    processed_counts[key] += 1

    return processed_counts


if __name__ == "__main__":
    print("=== Scanning dataset/raw ===")
    summary = get_dataset_summary("dataset/raw")
    print(f"Total valid images: {summary['total_valid_images']}")
    print("\nBreakdown by Class:")
    print(summary["summary_table"].to_string(index=False))
