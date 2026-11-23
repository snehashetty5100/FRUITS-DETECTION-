"""Select 30 annotation images (20 tune, 10 heldout) and run segmentation overlays.

Implements B.5 and B.6:
- 30 images selected with seed 42 (20 Train -> tune, 10 Test -> heldout).
- At least 4 per fruit, mixed stages.
- Saves images to dataset/annotation_queue/.
- Writes dataset/mask_manifest.csv (filename, split_role, source, verified).
- Evaluates K-Means (K=2,3,4,5 in RGB, HSV, Lab) and HSV thresholding on the 20 tune images.
- Saves high-contrast visual overlays to results/phase3/.
- Does not compute IoU.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import csv
import shutil
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.preprocessing import load_image, rgb_to_hsv, rgb_to_lab
from src.segmentation import (
    segment_by_configured_hsv,
    segment_by_kmeans,
    isolate_foreground_cluster,
    apply_morphological_cleanup,
)


def select_30_images(splits_path: str = "dataset/ripeness_splits.csv") -> Tuple[List[Dict], List[Dict]]:
    """Select 20 tune images (from train) and 10 heldout images (from test) with seed 42."""
    rng = random.Random(42)

    with open(splits_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fruits = ["apple", "banana", "mango", "orange", "tomato"]
    stages = ["unripe", "ripe", "overripe"]

    train_by_fruit = defaultdict(lambda: defaultdict(list))
    test_by_fruit = defaultdict(lambda: defaultdict(list))

    for r in rows:
        sp = r["split"]
        fr = r["fruit"]
        st = r["stage"]
        if sp == "train":
            train_by_fruit[fr][st].append(r)
        elif sp == "test":
            test_by_fruit[fr][st].append(r)

    tune_images = []
    # For 20 Train (tune) images: 4 per fruit (5 fruits)
    # Stage distribution per fruit: 1 unripe, 2 ripe, 1 overripe (or balanced)
    for fr in sorted(fruits):
        fr_tune = []
        # 1 from unripe
        if train_by_fruit[fr]["unripe"]:
            fr_tune.append(rng.choice(train_by_fruit[fr]["unripe"]))
        # 2 from ripe
        if len(train_by_fruit[fr]["ripe"]) >= 2:
            fr_tune.extend(rng.sample(train_by_fruit[fr]["ripe"], 2))
        elif train_by_fruit[fr]["ripe"]:
            fr_tune.append(train_by_fruit[fr]["ripe"][0])
        # 1 from overripe
        if train_by_fruit[fr]["overripe"]:
            fr_tune.append(rng.choice(train_by_fruit[fr]["overripe"]))

        # Fill if needed up to 4
        all_fr = [x for st in stages for x in train_by_fruit[fr][st] if x not in fr_tune]
        while len(fr_tune) < 4 and all_fr:
            chosen = rng.choice(all_fr)
            fr_tune.append(chosen)
            all_fr.remove(chosen)

        tune_images.extend(fr_tune[:4])

    heldout_images = []
    # For 10 Test (heldout) images: 2 per fruit (5 fruits)
    # Stage distribution: 1 unripe/overripe, 1 ripe
    for fr in sorted(fruits):
        fr_heldout = []
        # 1 from ripe
        if test_by_fruit[fr]["ripe"]:
            fr_heldout.append(rng.choice(test_by_fruit[fr]["ripe"]))
        # 1 from unripe or overripe
        other_stages = [x for st in ["unripe", "overripe"] for x in test_by_fruit[fr][st]]
        if other_stages:
            fr_heldout.append(rng.choice(other_stages))

        # Fill if needed up to 2
        all_test = [x for st in stages for x in test_by_fruit[fr][st] if x not in fr_heldout]
        while len(fr_heldout) < 2 and all_test:
            chosen = rng.choice(all_test)
            fr_heldout.append(chosen)
            all_test.remove(chosen)

        heldout_images.extend(fr_heldout[:2])

    return tune_images, heldout_images


def make_contour_overlay(img_rgb: np.ndarray, binary_mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Create a semi-transparent overlay with a solid 2px boundary contour."""
    mask_bool = binary_mask > 0
    overlay = img_rgb.copy().astype(np.float32)

    # 1. Semi-transparent foreground tint (25% blend)
    c_arr = np.array(color, dtype=np.float32)
    overlay[mask_bool] = overlay[mask_bool] * 0.75 + c_arr * 0.25

    # 2. Extract boundary contour (pixels where mask differs from neighbors)
    import scipy.ndimage as ndi
    eroded = ndi.binary_erosion(mask_bool, structure=np.ones((3, 3)))
    contour = mask_bool & ~eroded
    dilated_contour = ndi.binary_dilation(contour, structure=np.ones((2, 2)))

    overlay[dilated_contour] = c_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def generate_overlays_for_tune_images(tune_images: List[Dict], out_dir: str = "results/phase3"):
    """Run K-Means (K=2,3,4,5 in RGB, HSV, Lab) and HSV thresholding on the 20 tune images."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating segmentation overlays for {len(tune_images)} tune images in {out_dir}...")

    ks = [2, 3, 4, 5]
    color_spaces = ["RGB", "HSV", "Lab"]

    for idx, rec in enumerate(tune_images, 1):
        p = rec["path"]
        fname = os.path.basename(p)
        fruit = rec["fruit"]
        stage = rec["stage"]

        img_rgb = load_image(p)
        h, w = img_rgb.shape[:2]

        # 1. Run HSV thresholding + morphology
        raw_hsv_mask = segment_by_configured_hsv(img_rgb, fruit, stage)
        clean_hsv_mask = apply_morphological_cleanup(raw_hsv_mask)

        # 2. Run K-Means across K in [2,3,4,5] and color spaces [RGB, HSV, Lab]
        kmeans_results = {}
        for cs in color_spaces:
            if cs == "RGB":
                feat_img = img_rgb
            elif cs == "HSV":
                feat_img = rgb_to_hsv(img_rgb)
            elif cs == "Lab":
                feat_img = rgb_to_lab(img_rgb)

            for k in ks:
                _, cluster_map = segment_by_kmeans(feat_img, k=k, random_state=42)
                raw_fg = isolate_foreground_cluster(cluster_map, img_rgb)
                clean_fg = apply_morphological_cleanup(raw_fg)
                kmeans_results[(cs, k)] = clean_fg

        # 3. Assemble multi-panel comparison overlay image
        # Grid layout:
        # Top banner: Image info (Filename, Fruit, Stage, Tune)
        # Row 1: Original | HSV Threshold | K-Means Lab (K=3) | K-Means HSV (K=3) | K-Means RGB (K=3)
        # Row 2 (K=2): Lab (K=2) | HSV (K=2) | RGB (K=2)
        # Row 3 (K=4): Lab (K=4) | HSV (K=4) | RGB (K=4)
        # Row 4 (K=5): Lab (K=5) | HSV (K=5) | RGB (K=5)
        # Let's create a clean 5-panel primary comparison:
        # [Original] | [HSV Thresh] | [K-Means Lab K=3] | [K-Means HSV K=3] | [K-Means RGB K=3]
        panel_w, panel_h = 240, 240
        banner_h = 32
        text_h = 22
        pad = 6
        n_panels = 5

        sheet_w = n_panels * panel_w + (n_panels + 1) * pad
        # 2 rows of panels:
        # Row 1: Primary comparisons (Original, HSV Thresh, Lab K=3, HSV K=3, RGB K=3)
        # Row 2: Variations across K (Lab K=2, Lab K=4, Lab K=5, HSV K=4, RGB K=4)
        sheet_h = banner_h + 2 * (panel_h + text_h) + 3 * pad

        sheet = Image.new("RGB", (sheet_w, sheet_h), color=(25, 25, 30))
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()

        # Banner
        title = f"Tune Sample #{idx:02d}: {fname}  |  Fruit: {fruit.upper()}  |  Stage: {stage.upper()}  |  Split: TUNE"
        draw.text((pad + 4, 10), title, fill=(255, 255, 255), font=font)

        # Row 1 panels
        row1_panels = [
            ("Original Photo", img_rgb),
            ("HSV Threshold", make_contour_overlay(img_rgb, clean_hsv_mask, (255, 180, 0))),
            ("K-Means Lab (K=3)", make_contour_overlay(img_rgb, kmeans_results[("Lab", 3)], (0, 255, 120))),
            ("K-Means HSV (K=3)", make_contour_overlay(img_rgb, kmeans_results[("HSV", 3)], (0, 210, 255))),
            ("K-Means RGB (K=3)", make_contour_overlay(img_rgb, kmeans_results[("RGB", 3)], (255, 100, 200))),
        ]

        for c_idx, (label, arr) in enumerate(row1_panels):
            x = pad + c_idx * (panel_w + pad)
            y = banner_h + pad
            pil_p = Image.fromarray(arr).resize((panel_w, panel_h), Image.Resampling.LANCZOS)
            sheet.paste(pil_p, (x, y))
            draw.rectangle([x, y + panel_h, x + panel_w, y + panel_h + text_h], fill=(40, 40, 45))
            draw.text((x + 4, y + panel_h + 4), label, fill=(220, 220, 220), font=font)

        # Row 2 panels (showing K exploration across Lab, HSV, RGB)
        row2_panels = [
            ("K-Means Lab (K=2)", make_contour_overlay(img_rgb, kmeans_results[("Lab", 2)], (80, 255, 80))),
            ("K-Means Lab (K=4)", make_contour_overlay(img_rgb, kmeans_results[("Lab", 4)], (80, 255, 180))),
            ("K-Means Lab (K=5)", make_contour_overlay(img_rgb, kmeans_results[("Lab", 5)], (0, 255, 240))),
            ("K-Means HSV (K=4)", make_contour_overlay(img_rgb, kmeans_results[("HSV", 4)], (120, 180, 255))),
            ("K-Means RGB (K=4)", make_contour_overlay(img_rgb, kmeans_results[("RGB", 4)], (255, 140, 180))),
        ]

        for c_idx, (label, arr) in enumerate(row2_panels):
            x = pad + c_idx * (panel_w + pad)
            y = banner_h + pad + (panel_h + text_h + pad)
            pil_p = Image.fromarray(arr).resize((panel_w, panel_h), Image.Resampling.LANCZOS)
            sheet.paste(pil_p, (x, y))
            draw.rectangle([x, y + panel_h, x + panel_w, y + panel_h + text_h], fill=(40, 40, 45))
            draw.text((x + 4, y + panel_h + 4), label, fill=(220, 220, 220), font=font)

        base_no_ext = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, f"overlay_{base_no_ext}.png")
        sheet.save(out_path, quality=92)
        print(f"  [{idx:02d}/20] Saved overlay: {out_path}")


def main():
    print("=== Step B.5: Selecting 30 Images (20 Tune, 10 Heldout) ===")
    tune_images, heldout_images = select_30_images()

    print(f"Tune images (Train split): {len(tune_images)}")
    print(f"Heldout images (Test split): {len(heldout_images)}")
    assert len(tune_images) == 20, "Must have exactly 20 tune images"
    assert len(heldout_images) == 10, "Must have exactly 10 heldout images"

    # Verify fruit distribution
    fruit_counts = defaultdict(lambda: {"tune": 0, "heldout": 0})
    for x in tune_images:
        fruit_counts[x["fruit"]]["tune"] += 1
    for x in heldout_images:
        fruit_counts[x["fruit"]]["heldout"] += 1

    print("\nFruit distribution in selected 30 images:")
    for fr in sorted(fruit_counts.keys()):
        tu = fruit_counts[fr]["tune"]
        he = fruit_counts[fr]["heldout"]
        print(f"  {fr:<8}: Tune={tu}, Heldout={he}, Total={tu + he}")
        assert tu + he >= 4, f"Fruit {fr} must have at least 4 images"

    # Copy 30 images to dataset/annotation_queue/
    queue_dir = "dataset/annotation_queue"
    os.makedirs(queue_dir, exist_ok=True)

    all_selected = [(x, "tune") for x in tune_images] + [(x, "heldout") for x in heldout_images]

    manifest_rows = []
    for rec, role in all_selected:
        src_path = rec["path"]
        fname = os.path.basename(src_path)
        dst_path = os.path.join(queue_dir, fname)
        shutil.copyfile(src_path, dst_path)

        manifest_rows.append({
            "filename": fname,
            "split_role": role,
            "source": "none",
            "verified": "no",
        })

    # Write dataset/mask_manifest.csv
    mask_manifest_path = "dataset/mask_manifest.csv"
    with open(mask_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "split_role", "source", "verified"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nSaved {len(manifest_rows)} records to {mask_manifest_path}")
    print(f"Copied {len(manifest_rows)} image files to {queue_dir}/")

    # Step B.6: Run segmentation overlays for the 20 tune images
    print("\n=== Step B.6: Running Segmentation Pipelines for 20 Tune Images ===")
    generate_overlays_for_tune_images(tune_images, out_dir="results/phase3")
    print("\nAll 20 tune image overlays successfully generated in results/phase3/!")
    print("NOTE: No IoU or Dice metrics computed (waiting for verified hand-drawn masks).")


if __name__ == "__main__":
    main()
