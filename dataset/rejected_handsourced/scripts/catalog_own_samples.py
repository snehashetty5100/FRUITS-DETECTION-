"""Move and catalog existing hand-sourced images to dataset/own_samples/
Flagging known bad/ambiguous samples without deleting anything.
"""

from pathlib import Path
import shutil
import csv

OWN_DIR = Path("dataset/own_samples")
OWN_RAW = OWN_DIR / "raw"
OWN_PROC = OWN_DIR / "processed"

OWN_RAW.mkdir(parents=True, exist_ok=True)
OWN_PROC.mkdir(parents=True, exist_ok=True)

# Copy all current samples from dataset/raw and dataset/processed into dataset/own_samples/
SRC_RAW = Path("dataset/raw")
SRC_PROC = Path("dataset/processed")

raw_copied = 0
proc_copied = 0

for p in SRC_RAW.rglob("*.*"):
    if p.is_file() and p.name != ".gitkeep":
        rel = p.relative_to(SRC_RAW)
        dest = OWN_RAW / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        raw_copied += 1

for p in SRC_PROC.rglob("*.png"):
    if p.is_file():
        rel = p.relative_to(SRC_PROC)
        dest = OWN_PROC / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        proc_copied += 1

print(f"Copied {raw_copied} raw images and {proc_copied} processed images to dataset/own_samples/")

# Catalog and flag known bad ones
FLAGS = [
    {
        "filename": "tomato_overripe_003.jpg",
        "sample": "tomato_overripe_003",
        "fruit": "tomato",
        "stage": "overripe",
        "flag_reason": "Contains almonds / mixed nuts in the composition alongside the tomato."
    },
    {
        "filename": "apple_unripe_004.jpg",
        "sample": "apple_unripe_004",
        "fruit": "apple",
        "stage": "unripe",
        "flag_reason": "Cross-section of Granny Smith apple showing predominant white/pale flesh (mean H=30.0), masking the green skin hue."
    },
    {
        "filename": "banana_unripe_001.jpg",
        "sample": "banana_unripe_001",
        "fruit": "banana",
        "stage": "unripe",
        "flag_reason": "Transitionally ripening bananas with chartreuse/yellow peel (mean H=25.8) rather than physiological green."
    },
    {
        "filename": "banana_unripe_002.jpg",
        "sample": "banana_unripe_002",
        "fruit": "banana",
        "stage": "unripe",
        "flag_reason": "Baby bananas cluster / distinct cultivar rather than standard Cavendish variety."
    }
]

flags_csv = OWN_DIR / "flagged_samples.csv"
with open(flags_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "sample", "fruit", "stage", "flag_reason"])
    writer.writeheader()
    writer.writerows(FLAGS)

print(f"Cataloged flagged samples in {flags_csv}")
