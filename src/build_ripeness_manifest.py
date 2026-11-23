"""Build dataset/ripeness_manifest.csv and dataset/ripeness_splits.csv.

Follows strict rules:
- Uses standard library (csv, hashlib, os) + PIL + scikit-learn.
- No cv2 or pandas.
- Deduplicates images based on SHA-256 byte hashes.
- Assigns entire duplicate groups to a single split.
- Stratifies 70/15/15 train/val/test with random_state=42.
- Flags official_test_clean and leaks_train_test.
"""

import os
import csv
import hashlib
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split


def build_manifest_and_splits():
    root = "dataset/ripeness"
    all_files = []

    print("Scanning dataset/ripeness/...")
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in [".jpg", ".jpeg", ".png"]:
                p = os.path.join(dirpath, fn).replace("\\", "/")
                all_files.append(p)

    all_files.sort()
    print(f"Total images found: {len(all_files)}")

    # Extract metadata and calculate SHA256
    file_records = []
    hash_to_files = defaultdict(list)

    for p in all_files:
        # e.g., dataset/ripeness/Test/Overripe/apple_overripe_003.png
        parts = p.split("/")
        orig_split = parts[2]  # Train or Test
        stage_raw = parts[3]   # Overripe, Overipe, Ripe, Unripe
        fname = parts[4]

        fruit = fname.split("_")[0].lower()
        stage = stage_raw.lower().replace("overipe", "overripe")

        with open(p, "rb") as f:
            data = f.read()
        sha = hashlib.sha256(data).hexdigest()

        with Image.open(p) as img:
            mode = img.mode
            width, height = img.size

        rec = {
            "path": p,
            "fruit": fruit,
            "stage": stage,
            "original_split": orig_split,
            "sha256": sha,
            "mode": mode,
            "width": width,
            "height": height,
        }
        file_records.append(rec)
        hash_to_files[sha].append(rec)

    # Determine train/test leaks per hash group
    train_test_leaking_hashes = set()
    for sha, group in hash_to_files.items():
        has_train = any(r["original_split"] == "Train" for r in group)
        has_test = any(r["original_split"] == "Test" for r in group)
        if has_train and has_test:
            train_test_leaking_hashes.add(sha)

    # Assign duplicate group IDs and duplicate flags
    sorted_hashes = sorted(hash_to_files.keys())
    hash_to_group_id = {sha: idx + 1 for idx, sha in enumerate(sorted_hashes)}

    for rec in file_records:
        sha = rec["sha256"]
        rec["dup_group_id"] = hash_to_group_id[sha]
        rec["leaks_train_test"] = "true" if sha in train_test_leaking_hashes else "false"

    # Within each duplicate group, sort files deterministically by path
    # First file is is_duplicate=false, others are is_duplicate=true
    for sha in sorted_hashes:
        group = hash_to_files[sha]
        group.sort(key=lambda x: x["path"])
        group[0]["is_duplicate"] = "false"
        for member in group[1:]:
            member["is_duplicate"] = "true"

    # Flag official_test_clean
    for rec in file_records:
        if rec["original_split"] == "Test" and rec["leaks_train_test"] == "false":
            rec["official_test_clean"] = "true"
        else:
            rec["official_test_clean"] = "false"

    # Write ripeness_manifest.csv
    manifest_path = "dataset/ripeness_manifest.csv"
    manifest_headers = [
        "path",
        "fruit",
        "stage",
        "original_split",
        "sha256",
        "dup_group_id",
        "is_duplicate",
        "leaks_train_test",
        "official_test_clean",
        "mode",
        "width",
        "height",
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_headers)
        writer.writeheader()
        for rec in file_records:
            writer.writerow({k: rec[k] for k in manifest_headers})

    print(f"Saved {manifest_path} ({len(file_records)} rows)")

    # -------------------------------------------------------------
    # Step 3: Stratified 70/15/15 Split on Unique Groups
    # -------------------------------------------------------------
    # Keep one image per duplicate group (the canonical file with is_duplicate == 'false')
    unique_items = [hash_to_files[sha][0] for sha in sorted_hashes]
    for item in unique_items:
        item["stratum"] = f"{item['fruit']}_{item['stage']}"

    # Stratify: 70% train, 30% temp -> 15% val, 15% test
    strata = [x["stratum"] for x in unique_items]
    train_items, temp_items = train_test_split(
        unique_items, test_size=0.30, random_state=42, stratify=strata
    )
    temp_strata = [x["stratum"] for x in temp_items]
    val_items, test_items = train_test_split(
        temp_items, test_size=0.50, random_state=42, stratify=temp_strata
    )

    for item in train_items:
        item["split"] = "train"
    for item in val_items:
        item["split"] = "val"
    for item in test_items:
        item["split"] = "test"

    # Map group_id to assigned split (assigning whole duplicate group to this split)
    group_id_to_split = {item["dup_group_id"]: item["split"] for item in unique_items}

    # Write ripeness_splits.csv (keeping one image per duplicate group)
    splits_path = "dataset/ripeness_splits.csv"
    splits_headers = [
        "path",
        "fruit",
        "stage",
        "split",
        "official_test_clean",
        "original_split",
        "sha256",
        "dup_group_id",
        "mode",
        "width",
        "height",
    ]

    # Sort by fruit, stage, path for neat readability
    unique_items.sort(key=lambda x: (x["fruit"], x["stage"], x["path"]))

    with open(splits_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=splits_headers)
        writer.writeheader()
        for rec in unique_items:
            writer.writerow({k: rec[k] for k in splits_headers})

    print(f"Saved {splits_path} ({len(unique_items)} unique rows)")

    # -------------------------------------------------------------
    # Print Fruit x Stage x Split count table
    # -------------------------------------------------------------
    table_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for item in unique_items:
        table_counts[(item["fruit"], item["stage"])][item["split"]] += 1

    all_fruits = ["apple", "banana", "mango", "orange", "papaya", "tomato"]
    all_stages = ["unripe", "ripe", "overripe"]

    print("\n" + "=" * 65)
    print("FRUIT x STAGE x SPLIT COUNT TABLE (Deduplicated Unique Groups)")
    print("=" * 65)
    row_fmt = "{:<8} | {:<10} | {:<7} | {:<6} | {:<6} | {:<7}"
    print(row_fmt.format("Fruit", "Stage", "Train", "Val", "Test", "Total"))
    print("-" * 65)

    warnings = []
    for fruit in all_fruits:
        for stage in all_stages:
            counts = table_counts.get((fruit, stage), {"train": 0, "val": 0, "test": 0})
            tr = counts["train"]
            va = counts["val"]
            te = counts["test"]
            tot = tr + va + te
            print(row_fmt.format(fruit, stage, tr, va, te, tot))

            if tot < 50 or te < 10:
                warnings.append(
                    f"WARNING: Cell ({fruit}, {stage}) has {tot} total images (< 50) and {te} test images (< 10)."
                )

    print("-" * 65)
    total_tr = sum(c["train"] for c in table_counts.values())
    total_va = sum(c["val"] for c in table_counts.values())
    total_te = sum(c["test"] for c in table_counts.values())
    total_all = total_tr + total_va + total_te
    print(row_fmt.format("TOTAL", "ALL", total_tr, total_va, total_te, total_all))
    print("=" * 65)

    if warnings:
        print("\nDATASET WARNINGS:")
        for w in warnings:
            print(" ", w)
    else:
        print("\nAll cells meet the >= 50 total and >= 10 test threshold.")

    return manifest_path, splits_path


if __name__ == "__main__":
    build_manifest_and_splits()
