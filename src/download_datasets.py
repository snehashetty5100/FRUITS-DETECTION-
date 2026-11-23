"""Dataset Management and Ingestion Module

The project standardizes on the benchmark Fruits-360 dataset located in dataset/fruits360/.
All prior ad-hoc downloading and search-based scraper logic has been removed.
"""

from pathlib import Path

FRUITS360_DIR = Path("dataset/fruits360")


def check_dataset_status():
    """Verify presence of official Fruits-360 dataset and report status."""
    print("=== Dataset Status Summary ===")
    print(f"Fruits-360 Directory: {FRUITS360_DIR} (Exists: {FRUITS360_DIR.exists()})")
    if FRUITS360_DIR.exists():
        tr_dir = FRUITS360_DIR / "Training"
        te_dir = FRUITS360_DIR / "Test"
        print(f"  Training Split Exists: {tr_dir.exists()}")
        print(f"  Test Split Exists: {te_dir.exists()}")


if __name__ == "__main__":
    check_dataset_status()
