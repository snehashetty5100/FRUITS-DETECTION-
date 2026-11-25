"""Phase 3: Segmentation Pipeline Evaluation, IoU & Dice Benchmarking

Evaluates:
1. K-Means (K=3) with K-Means++, valid content bounding box, and multi-factor foreground scoring.
2. Configured HSV Color Thresholding (with unripe, ripe, and overripe ranges).
3. Full Morphological Post-Processing (Opening, Closing, Hole Filling, Largest Component).

Evaluation Protocol:
- Reads dataset/manifest.csv (code uses MANIFEST_PATH = Path("dataset/manifest.csv"), not dataset/ground_truth_masks/manifest.csv).
- ONLY evaluates masks marked as verified=yes.
- Reports per-image IoU/Dice, per-fruit mean (n), and overall mean (n).
- Evaluates Tune vs Held-out partitions separately with an explicit warning:
  Held-out set must NOT be used for tuning.
- Writes evaluation results to results/phase3/eval.csv using the csv module.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
from collections import defaultdict
import numpy as np
from PIL import Image

from src.preprocessing import load_image
from src.segmentation import (
    segment_by_kmeans,
    isolate_foreground_cluster,
    segment_by_configured_hsv,
    apply_morphological_cleanup,
    calculate_mask_iou,
    calculate_mask_dice,
    get_valid_content_bbox
)

MANIFEST_PATH = Path("dataset/manifest.csv")
GT_DIR = Path("dataset/fruits360_masks")
FRUITS360_DIR = Path("dataset/fruits360")
PHASE3_DIR = Path("results/phase3")
EVAL_CSV_PATH = PHASE3_DIR / "eval.csv"

PHASE3_DIR.mkdir(parents=True, exist_ok=True)


def create_segmentation_overlay(img_rgb: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Image.Image:
    """Create a 4-panel comparison: [Original, Ground Truth, Prediction, 3-Way Color Overlay]."""
    pred = pred_mask > 0
    gt = gt_mask > 0

    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    # High-contrast overlay: 55% dimmed original + colored highlights
    overlay = (img_rgb * 0.55).astype(np.uint8)
    overlay[tp] = np.clip(overlay[tp] * 0.4 + np.array([0, 230, 0]) * 0.6, 0, 255)    # Green = True Positive
    overlay[fp] = np.clip(overlay[fp] * 0.4 + np.array([240, 30, 30]) * 0.6, 0, 255)  # Red = False Positive
    overlay[fn] = np.clip(overlay[fn] * 0.4 + np.array([30, 100, 255]) * 0.6, 0, 255) # Blue = False Negative

    comp = Image.new("RGB", (224 * 4, 224))
    comp.paste(Image.fromarray(img_rgb), (0, 0))
    comp.paste(Image.fromarray((gt * 255).astype(np.uint8)).convert("RGB"), (224, 0))
    comp.paste(Image.fromarray((pred * 255).astype(np.uint8)).convert("RGB"), (224 * 2, 0))
    comp.paste(Image.fromarray(overlay), (224 * 3, 0))

    return comp


def load_manifest():
    """Load and parse the ground-truth manifest."""
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_benchmark():
    manifest_entries = load_manifest()
    if not manifest_entries:
        print("Manifest is empty or missing.")
        return []

    # Filter strictly for verified=yes
    verified_entries = [r for r in manifest_entries if r.get("verified", "").strip().lower() == "yes"]
    unverified_count = len(manifest_entries) - len(verified_entries)

    print("=" * 80)
    print("PHASE 3 SEGMENTATION BENCHMARK & EVALUATION")
    print("=" * 80)
    print(f"Total manifest entries: {len(manifest_entries)}")
    print(f"Verified entries (verified=yes): {len(verified_entries)}")
    print(f"Unverified entries (awaiting hand-annotation): {unverified_count}")
    print("-" * 80)

    # Always ensure results/phase3/eval.csv has valid headers even if 0 verified
    fieldnames = [
        "filename", "sample", "fruit", "stage", "split", "source",
        "kmeans_iou", "kmeans_dice", "hsv_iou", "hsv_dice"
    ]

    if not verified_entries:
        print("\n[!] NOTICE: No verified masks found (verified=yes) in dataset/manifest.csv.")
        print("-" * 80)

        # Write empty eval.csv with header
        with open(EVAL_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print(f"Initialized empty evaluation log at {EVAL_CSV_PATH}")
        return []

    # Display Warning Banner about Held-Out Set
    print("\n********************************************************************************")
    print("WARNING: Held-out set must NOT be used for tuning thresholds or algorithms.")
    print("Only use the 'tune' partition to guide algorithmic adjustments.")
    print("********************************************************************************\n")

    records = []

    for entry in verified_entries:
        mask_filename = entry["filename"]
        split = entry.get("split", "heldout").strip().lower()
        source = entry.get("source", "hand").strip()

        stem = mask_filename.replace("_mask.png", "")
        parts = stem.split("_")
        fruit, stage = parts[0], parts[1]

        img_path = FRUITS360_DIR / "Training" / fruit / f"{stem}.jpg"
        gt_path = GT_DIR / mask_filename

        if not img_path.exists():
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue
        if not gt_path.exists():
            print(f"Warning: GT mask {gt_path} not found. Skipping.")
            continue

        img_rgb = load_image(str(img_path))
        with Image.open(gt_path) as m:
            gt_mask = np.array(m.convert("L")) > 128

        _, _, _, _, valid_mask = get_valid_content_bbox(img_rgb)

        # Method 1: K-Means (K=3) + Foreground Scoring + Morphology
        _, cluster_map, bbox = segment_by_kmeans(img_rgb, k=3, max_iter=25)
        raw_kmeans = isolate_foreground_cluster(cluster_map, img_rgb, bbox=bbox)
        clean_kmeans = apply_morphological_cleanup(raw_kmeans, radius=3, keep_largest_only=True)

        # Method 2: Configured HSV Thresholding + Morphology
        raw_hsv = segment_by_configured_hsv(img_rgb, fruit, stage, valid_mask=valid_mask)
        clean_hsv = apply_morphological_cleanup(raw_hsv, radius=3, keep_largest_only=True)

        # Compute IoU and Sørensen-Dice
        km_iou = calculate_mask_iou(clean_kmeans, gt_mask)
        km_dice = calculate_mask_dice(clean_kmeans, gt_mask)
        hsv_iou = calculate_mask_iou(clean_hsv, gt_mask)
        hsv_dice = calculate_mask_dice(clean_hsv, gt_mask)

        records.append({
            "filename": mask_filename,
            "sample": stem,
            "fruit": fruit.capitalize(),
            "stage": stage.capitalize(),
            "split": split,
            "source": source,
            "kmeans_iou": round(km_iou, 4),
            "kmeans_dice": round(km_dice, 4),
            "hsv_iou": round(hsv_iou, 4),
            "hsv_dice": round(hsv_dice, 4),
        })

        # Save visual comparison overlay
        overlay_img = create_segmentation_overlay(img_rgb, clean_kmeans, gt_mask)
        overlay_img.save(PHASE3_DIR / f"overlay_{stem}.png", "PNG")

    # Write results to results/phase3/eval.csv with csv module
    with open(EVAL_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # 1. Per-Image Table
    print(f"{'Sample':<24} | {'Fruit':<8} | {'Stage':<10} | {'Split':<8} | {'KM IoU':<8} | {'KM Dice':<8} | {'HSV IoU':<8} | {'HSV Dice'}")
    print("-" * 95)
    for r in records:
        print(f"{r['sample']:<24} | {r['fruit']:<8} | {r['stage']:<10} | {r['split']:<8} | {r['kmeans_iou']:<8.4f} | {r['kmeans_dice']:<8.4f} | {r['hsv_iou']:<8.4f} | {r['hsv_dice']:.4f}")

    def calc_stats(subset):
        if not subset:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(np.mean([r["kmeans_iou"] for r in subset])),
            float(np.mean([r["kmeans_dice"] for r in subset])),
            float(np.mean([r["hsv_iou"] for r in subset])),
            float(np.mean([r["hsv_dice"] for r in subset]))
        )

    tune_records = [r for r in records if r["split"] == "tune"]
    heldout_records = [r for r in records if r["split"] == "heldout"]

    t_km_iou, t_km_dice, t_hsv_iou, t_hsv_dice = calc_stats(tune_records)
    h_km_iou, h_km_dice, h_hsv_iou, h_hsv_dice = calc_stats(heldout_records)
    o_km_iou, o_km_dice, o_hsv_iou, o_hsv_dice = calc_stats(records)

    # 2. Performance by Partition
    print("\n" + "=" * 95)
    print("PERFORMANCE BY PARTITION:")
    print(f"{'Partition':<18} | {'KM Mean IoU':<14} | {'KM Mean Dice':<14} | {'HSV Mean IoU':<14} | {'HSV Mean Dice'}")
    print("-" * 95)
    print(f"{f'Tune (n={len(tune_records)})':<18} | {t_km_iou:<14.4f} | {t_km_dice:<14.4f} | {t_hsv_iou:<14.4f} | {t_hsv_dice:.4f}")
    print(f"{f'Held-Out (n={len(heldout_records)})':<18} | {h_km_iou:<14.4f} | {h_km_dice:<14.4f} | {h_hsv_iou:<14.4f} | {h_hsv_dice:.4f}")
    print(f"{f'Overall (n={len(records)})':<18} | {o_km_iou:<14.4f} | {o_km_dice:<14.4f} | {o_hsv_iou:<14.4f} | {o_hsv_dice:.4f}")

    # 3. Mean IoU and Dice Per Fruit
    print("\n" + "=" * 95)
    print("MEAN IoU AND DICE PER FRUIT (ACROSS VERIFIED SAMPLES):")
    print(f"{'Fruit (Count)':<18} | {'KM Mean IoU':<14} | {'KM Mean Dice':<14} | {'HSV Mean IoU':<14} | {'HSV Mean Dice'}")
    print("-" * 95)
    fruit_names = sorted(list(set(r["fruit"] for r in records)))
    for fname in fruit_names:
        f_rec = [r for r in records if r["fruit"] == fname]
        f_km_iou, f_km_dice, f_hsv_iou, f_hsv_dice = calc_stats(f_rec)
        print(f"{f'{fname} (n={len(f_rec)})':<18} | {f_km_iou:<14.4f} | {f_km_dice:<14.4f} | {f_hsv_iou:<14.4f} | {f_hsv_dice:.4f}")

    print("\n" + "=" * 95)
    print("SAMPLES SCORING BELOW 0.60 IoU IN K-MEANS:")
    low_km = [r for r in records if r["kmeans_iou"] < 0.6]
    if low_km:
        for r in low_km:
            print(f"  - {r['sample']} ({r['fruit']} - {r['stage']} | {r['split']}): IoU = {r['kmeans_iou']:.4f}")
    else:
        print("  None. All verified samples scored >= 0.60 IoU.")

    print("\nSAMPLES SCORING BELOW 0.60 IoU IN HSV THRESHOLDING:")
    low_hsv = [r for r in records if r["hsv_iou"] < 0.6]
    if low_hsv:
        for r in low_hsv:
            print(f"  - {r['sample']} ({r['fruit']} - {r['stage']} | {r['split']}): IoU = {r['hsv_iou']:.4f}")
    else:
        print("  None. All verified samples scored >= 0.60 IoU.")

    print(f"\n[✓] Detailed evaluation metrics written to: {EVAL_CSV_PATH}")
    return records


if __name__ == "__main__":
    run_benchmark()
