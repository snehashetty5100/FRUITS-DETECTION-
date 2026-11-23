"""Dataset Preparation and Validation Utilities for Fruits-360

Module to:
1. Scan and verify Fruits-360 dataset structure across target fruit classes.
2. Check image file integrity in Training and Test partitions.
3. Provide summary breakdowns of available sample images.
"""

import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

FRUITS360_DIR = Path("dataset/fruits360")
TARGET_FRUITS = ["apple", "banana", "mango", "orange", "papaya", "tomato"]
EXCLUDE_WORDS = ["mangostan", "pepper", "pineapple"]


def scan_fruits360(split: str = "Training") -> List[Dict[str, any]]:
    """Scan Fruits-360 directory for target fruit classes and verify image integrity."""
    split_dir = FRUITS360_DIR / split
    valid_extensions = {".jpg", ".jpeg", ".png"}
    records = []

    if not split_dir.exists():
        return records

    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_lower = cls_dir.name.lower()
        if any(kw in cls_lower for kw in TARGET_FRUITS) and not any(ex in cls_lower for ex in EXCLUDE_WORDS):
            valid_count = 0
            corrupt_count = 0
            for file_path in cls_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    try:
                        with Image.open(file_path) as img:
                            img.verify()
                        valid_count += 1
                    except Exception:
                        corrupt_count += 1

            records.append({
                "Class": cls_dir.name,
                "Split": split,
                "Valid Images": valid_count,
                "Corrupt": corrupt_count,
                "Path": str(cls_dir)
            })

    return records


def print_fruits360_summary():
    """Display tabular breakdown of Fruits-360 target categories."""
    for split in ["Training", "Test"]:
        records = scan_fruits360(split)
        print(f"\n=== Fruits-360 {split} Split Summary ===")
        print(f"{'Class Folder':<28} | {'Valid Images':<12} | {'Directory'}")
        print("-" * 75)
        total_valid = 0
        for r in records:
            print(f"{r['Class']:<28} | {r['Valid Images']:<12} | {r['Path']}")
            total_valid += r["Valid Images"]
        print("-" * 75)
        print(f"Total Valid Images in '{split}': {total_valid}")


if __name__ == "__main__":
    print_fruits360_summary()
