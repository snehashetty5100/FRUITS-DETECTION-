"""Rebuild ripeness splits with perceptual hash near-duplicate merging.

1. Calculates 16x16 average hash (aHash) via Pillow + NumPy.
2. Identifies all pairs within Hamming distance <= 5.
3. Merges them with exact SHA-256 duplicate groups via Union-Find.
4. Stratifies 70/15/15 train/val/test with seed 42.
5. Updates dataset/ripeness_manifest.csv and dataset/ripeness_splits.csv.
6. Analyzes official_test_clean distribution across the new splits.
"""

import os
import csv
import hashlib
from collections import defaultdict
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def compute_ahash_16x16(img: Image.Image) -> np.ndarray:
    """Compute 16x16 average hash as a 256-element boolean vector."""
    gray = img.convert("L").resize((16, 16), Image.Resampling.BILINEAR)
    arr = np.array(gray, dtype=np.float32)
    m = arr.mean()
    return (arr >= m).flatten()


def ahash_to_hex(bits: np.ndarray) -> str:
    """Convert boolean vector of 256 bits to a 64-character hex string."""
    packed = np.packbits(bits)
    return packed.tobytes().hex()


def run_rebuild():
    root = "dataset/ripeness"
    files = []
    for dirpath, _, fnames in os.walk(root):
        for fn in fnames:
            if fn.endswith(".jpg") or fn.endswith(".jpeg") or fn.endswith(".png"):
                files.append(os.path.join(dirpath, fn).replace("\\", "/"))
    files.sort()
    n_files = len(files)
    print(f"Total files: {n_files}")

    # 1. Compute SHA256 and 16x16 aHash for all images
    print("Computing SHA-256 and 16x16 perceptual hashes...")
    sha_list = []
    ahash_bits = []
    records = []

    for idx, p in enumerate(files):
        parts = p.split("/")
        orig_split = parts[2]
        stage_raw = parts[3]
        fname = parts[4]
        fruit = fname.split("_")[0].lower()
        stage = stage_raw.lower().replace("overipe", "overripe")

        with open(p, "rb") as f:
            data = f.read()
        sha = hashlib.sha256(data).hexdigest()
        sha_list.append(sha)

        with Image.open(p) as img:
            bits = compute_ahash_16x16(img)
            mode = img.mode
            w, h = img.size

        ahash_bits.append(bits.astype(np.float32))
        records.append({
            "idx": idx,
            "path": p,
            "fruit": fruit,
            "stage": stage,
            "original_split": orig_split,
            "sha256": sha,
            "ahash_hex": ahash_to_hex(bits),
            "mode": mode,
            "width": w,
            "height": h,
        })

    # 2. Pairwise Hamming distance calculation
    X = np.array(ahash_bits)
    row_sums = X.sum(axis=1, keepdims=True)
    D = row_sums + row_sums.T - 2.0 * np.dot(X, X.T)
    D = np.round(D).astype(int)

    # Pairs with Hamming distance <= 5 (upper triangle only)
    pairs_i, pairs_j = np.where((D <= 5) & (np.triu(np.ones_like(D, dtype=bool), k=1)))
    print(f"Total near-duplicate pairs with Hamming distance <= 5: {len(pairs_i)}")

    # 3. Union-Find to merge exact SHA-256 duplicates and near duplicates
    parent = list(range(n_files))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Add exact SHA256 duplicates
    sha_to_indices = defaultdict(list)
    for idx, sha in enumerate(sha_list):
        sha_to_indices[sha].append(idx)
    for idxs in sha_to_indices.values():
        for k in range(len(idxs) - 1):
            union(idxs[k], idxs[k + 1])

    # Add near duplicates
    for i, j in zip(pairs_i, pairs_j):
        union(i, j)

    # Group files into components
    components = defaultdict(list)
    for idx in range(n_files):
        components[find(idx)].append(idx)

    # Sort components deterministically by the first file path
    sorted_comp_roots = sorted(components.keys(), key=lambda r: sorted([records[i]["path"] for i in components[r]])[0])
    
    # Analyze components
    multi_components = [components[r] for r in sorted_comp_roots if len(components[r]) > 1]
    single_components = [components[r] for r in sorted_comp_roots if len(components[r]) == 1]
    
    cross_tt_components = []
    for comp in multi_components:
        splits = set(records[i]["original_split"] for i in comp)
        if "Train" in splits and "Test" in splits:
            cross_tt_components.append(comp)

    print("\n--- Duplicate & Near-Duplicate Summary ---")
    print(f"Total merged components (unique items): {len(sorted_comp_roots)}")
    print(f"Components with > 1 image: {len(multi_components)}")
    print(f"Total images involved in > 1 image components: {sum(len(c) for c in multi_components)}")
    print(f"Components crossing old Train/Test boundary: {len(cross_tt_components)}")
    print(f"Total images in Train/Test crossing components: {sum(len(c) for c in cross_tt_components)}")

    # 4. Canonical items and Stratification
    # Assign dup_group_id (1..N) and identify canonical item for each component
    group_id_map = {}
    canonical_items = []

    for group_num, root_idx in enumerate(sorted_comp_roots, 1):
        comp = components[root_idx]
        comp_records = [records[i] for i in comp]
        comp_records.sort(key=lambda x: x["path"])
        
        crosses_tt = "true" if any(r["original_split"] == "Train" for r in comp_records) and any(r["original_split"] == "Test" for r in comp_records) else "false"

        for idx, rec in enumerate(comp_records):
            rec["dup_group_id"] = group_num
            rec["group_size"] = len(comp)
            rec["is_duplicate"] = "false" if idx == 0 else "true"
            rec["leaks_train_test"] = crosses_tt

        canonical = comp_records[0]
        canonical["stratum"] = f"{canonical['fruit']}_{canonical['stage']}"
        canonical_items.append(canonical)

    # Determine official_test_clean per image:
    # An image is official_test_clean if its original_split was 'Test' AND its merged group does NOT contain any Train image
    for rec in records:
        if rec["original_split"] == "Test" and rec["leaks_train_test"] == "false":
            rec["official_test_clean"] = "true"
        else:
            rec["official_test_clean"] = "false"

    # 5. Stratify 70/15/15 with seed 42 on the canonical items
    strata = [x["stratum"] for x in canonical_items]
    train_items, temp_items = train_test_split(
        canonical_items, test_size=0.30, random_state=42, stratify=strata
    )
    temp_strata = [x["stratum"] for x in temp_items]
    val_items, test_items = train_test_split(
        temp_items, test_size=0.50, random_state=42, stratify=temp_strata
    )

    for x in train_items:
        x["split"] = "train"
    for x in val_items:
        x["split"] = "val"
    for x in test_items:
        x["split"] = "test"

    group_split_map = {x["dup_group_id"]: x["split"] for x in canonical_items}

    for rec in records:
        rec["split"] = group_split_map[rec["dup_group_id"]]
        if rec["official_test_clean"] == "true":
            rec["official_clean_assigned_split"] = rec["split"]
        else:
            rec["official_clean_assigned_split"] = "none"

    # 6. Save dataset/ripeness_manifest.csv
    manifest_headers = [
        "path",
        "fruit",
        "stage",
        "original_split",
        "sha256",
        "ahash",
        "dup_group_id",
        "group_size",
        "is_duplicate",
        "leaks_train_test",
        "official_test_clean",
        "split",
        "official_clean_assigned_split",
        "mode",
        "width",
        "height",
    ]

    manifest_path = "dataset/ripeness_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_headers)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "path": rec["path"],
                "fruit": rec["fruit"],
                "stage": rec["stage"],
                "original_split": rec["original_split"],
                "sha256": rec["sha256"],
                "ahash": rec["ahash_hex"],
                "dup_group_id": rec["dup_group_id"],
                "group_size": rec["group_size"],
                "is_duplicate": rec["is_duplicate"],
                "leaks_train_test": rec["leaks_train_test"],
                "official_test_clean": rec["official_test_clean"],
                "split": rec["split"],
                "official_clean_assigned_split": rec["official_clean_assigned_split"],
                "mode": rec["mode"],
                "width": rec["width"],
                "height": rec["height"],
            })
    print(f"Updated {manifest_path} ({len(records)} rows)")

    # 7. Save dataset/ripeness_splits.csv (canonical items only)
    splits_headers = [
        "path",
        "fruit",
        "stage",
        "split",
        "official_test_clean",
        "official_clean_assigned_split",
        "original_split",
        "sha256",
        "ahash",
        "dup_group_id",
        "group_size",
        "mode",
        "width",
        "height",
    ]

    canonical_items.sort(key=lambda x: (x["fruit"], x["stage"], x["path"]))

    splits_path = "dataset/ripeness_splits.csv"
    with open(splits_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=splits_headers)
        writer.writeheader()
        for rec in canonical_items:
            writer.writerow({
                "path": rec["path"],
                "fruit": rec["fruit"],
                "stage": rec["stage"],
                "split": rec["split"],
                "official_test_clean": rec["official_test_clean"],
                "official_clean_assigned_split": rec["official_clean_assigned_split"],
                "original_split": rec["original_split"],
                "sha256": rec["sha256"],
                "ahash": rec["ahash_hex"],
                "dup_group_id": rec["dup_group_id"],
                "group_size": rec["group_size"],
                "mode": rec["mode"],
                "width": rec["width"],
                "height": rec["height"],
            })
    print(f"Updated {splits_path} ({len(canonical_items)} rows)")

    # 8. Print Official Test Clean Distribution
    clean_counts = defaultdict(int)
    for rec in canonical_items:
        if rec["official_test_clean"] == "true":
            clean_counts[rec["official_clean_assigned_split"]] += 1

    print("\n--- Official Test Clean Images in New Splits ---")
    tot_clean = sum(clean_counts.values())
    for sp in ["train", "val", "test"]:
        print(f"  Official Clean in new {sp:<5}: {clean_counts[sp]} ({clean_counts[sp]/tot_clean*100:.1f}%)")
    print(f"  Total Official Clean images: {tot_clean}")

    # 9. Print Fruit x Stage x Split Table
    table_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for item in canonical_items:
        table_counts[(item["fruit"], item["stage"])][item["split"]] += 1

    all_fruits = ["apple", "banana", "mango", "orange", "tomato"]
    all_stages = ["unripe", "ripe", "overripe"]

    print("\n" + "=" * 65)
    print("FRUIT x STAGE x SPLIT COUNT TABLE (Near-Duplicate Merged Groups)")
    print("=" * 65)
    row_fmt = "{:<8} | {:<10} | {:<7} | {:<6} | {:<6} | {:<7}"
    print(row_fmt.format("Fruit", "Stage", "Train", "Val", "Test", "Total"))
    print("-" * 65)

    warnings = []
    for fruit in all_fruits:
        for stage in all_stages:
            counts = table_counts.get((fruit, stage), {"train": 0, "val": 0, "test": 0})
            tr, va, te = counts["train"], counts["val"], counts["test"]
            tot = tr + va + te
            print(row_fmt.format(fruit, stage, tr, va, te, tot))
            if tot < 50 or te < 10:
                warnings.append(f"Cell ({fruit}, {stage}): {tot} total images (< 50) and {te} test images (< 10).")

    print("-" * 65)
    tot_tr = sum(c["train"] for c in table_counts.values())
    tot_va = sum(c["val"] for c in table_counts.values())
    tot_te = sum(c["test"] for c in table_counts.values())
    tot_all = tot_tr + tot_va + tot_te
    print(row_fmt.format("TOTAL", "ALL", tot_tr, tot_va, tot_te, tot_all))
    print("=" * 65)

    if warnings:
        print("\nDATASET WARNINGS:")
        for w in warnings:
            print("  [WARNING]", w)

    return canonical_items


if __name__ == "__main__":
    run_rebuild()
