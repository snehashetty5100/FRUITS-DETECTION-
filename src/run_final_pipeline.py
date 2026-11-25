"""End-to-End Pipeline for Fruit Ripeness Detection.

Implements all 5 project steps:
1. Segmentation: Lab K-Means (K=3) + morphological cleanup. Saves 12 example sheets.
2. Feature extraction: R,G,B, H,S,V, L,a,b mean/std + 16-bin hue histogram -> results/final/features.csv.
3. Model training: SVM and Random Forest tuned on val, evaluated once on test, plus majority baseline.
4. Results file: results/final/results.md with accuracy, macro P/R/F1, confusion matrices, latency, class counts.
5. Comparison table: RGB vs HSV vs Lab, segmented vs unsegmented (background interference).
Saves best model to models/ripeness_classifier.pkl.
"""

import os
import sys
import csv
import time
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Project imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_image, rgb_to_lab, rgb_to_hsv
from src.segmentation import isolate_foreground_cluster, apply_morphological_cleanup


def fast_lab_kmeans_segmentation(img_rgb: np.ndarray, k: int = 3, max_iter: int = 12, random_state: int = 42) -> np.ndarray:
    """Run K-Means (K=3) in CIE-Lab color space with morphological cleanup."""
    h, w, _ = img_rgb.shape
    feat_img = rgb_to_lab(img_rgb)

    # Subsample pixels for centroid discovery (10,000 pixels)
    sub = feat_img[::3, ::3].reshape(-1, 3).astype(np.float32)
    n_sub = len(sub)

    rng = np.random.RandomState(random_state)
    centroids = [sub[rng.choice(n_sub)]]
    for _ in range(1, k):
        dists = np.min([np.sum((sub - c) ** 2, axis=1) for c in centroids], axis=0)
        total_d = dists.sum()
        probs = dists / total_d if total_d > 1e-6 else np.ones(n_sub) / n_sub
        centroids.append(sub[rng.choice(n_sub, p=probs)])
    centroids = np.array(centroids, dtype=np.float32)

    # Iterative Lloyd updates on subsample
    sub_labels = np.full(n_sub, -1, dtype=np.int32)
    for _ in range(max_iter):
        dists = np.stack([np.sum((sub - c) ** 2, axis=1) for c in centroids], axis=1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(sub_labels, new_labels):
            break
        sub_labels = new_labels
        for i in range(k):
            mask = (sub_labels == i)
            if np.any(mask):
                centroids[i] = np.mean(sub[mask], axis=0)

    # Fast full image assignment (90,000 pixels)
    full_pix = feat_img.reshape(-1, 3).astype(np.float32)
    full_dists = np.stack([np.sum((full_pix - c) ** 2, axis=1) for c in centroids], axis=1)
    full_labels = np.argmin(full_dists, axis=1).reshape(h, w)

    raw_mask = isolate_foreground_cluster(full_labels, img_rgb)
    clean_mask = apply_morphological_cleanup(raw_mask)

    # Fallback safety: if mask is degenerate (<1% or >99%), use center 50% box
    ratio = np.mean(clean_mask > 0)
    if ratio < 0.01 or ratio > 0.99:
        clean_mask = np.zeros((h, w), dtype=np.uint8)
        clean_mask[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)] = 255

    return clean_mask


def extract_features_single(img_rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract 34 segmented features (masked fruit) and 34 unsegmented features (whole image)."""
    h, w, _ = img_rgb.shape
    mask_bool = mask > 0
    fg_count = np.sum(mask_bool)
    if fg_count < 25:
        mask_bool = np.zeros((h, w), dtype=bool)
        mask_bool[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)] = True

    mask_ratio = float(np.mean(mask_bool))
    img_hsv = rgb_to_hsv(img_rgb)
    img_lab = rgb_to_lab(img_rgb)

    # 1. Segmented Features (Fruit pixels only)
    seg_rgb = img_rgb[mask_bool].astype(np.float32)
    seg_hsv = img_hsv[mask_bool].astype(np.float32)
    seg_lab = img_lab[mask_bool].astype(np.float32)

    seg_r_mean, seg_g_mean, seg_b_mean = np.mean(seg_rgb, axis=0)
    seg_r_std, seg_g_std, seg_b_std = np.std(seg_rgb, axis=0)
    seg_h_mean, seg_s_mean, seg_v_mean = np.mean(seg_hsv, axis=0)
    seg_h_std, seg_s_std, seg_v_std = np.std(seg_hsv, axis=0)
    seg_l_mean, seg_a_mean, seg_b_lab_mean = np.mean(seg_lab, axis=0)
    seg_l_std, seg_a_std, seg_b_lab_std = np.std(seg_lab, axis=0)
    seg_hue_hist, _ = np.histogram(seg_hsv[:, 0], bins=16, range=(0, 180), density=True)

    seg_feats = np.array([
        seg_r_mean, seg_r_std, seg_g_mean, seg_g_std, seg_b_mean, seg_b_std,
        seg_h_mean, seg_h_std, seg_s_mean, seg_s_std, seg_v_mean, seg_v_std,
        seg_l_mean, seg_l_std, seg_a_mean, seg_a_std, seg_b_lab_mean, seg_b_lab_std,
    ] + seg_hue_hist.tolist(), dtype=np.float32)

    # 2. Unsegmented Features (Whole image without mask)
    raw_rgb = img_rgb.reshape(-1, 3).astype(np.float32)
    raw_hsv = img_hsv.reshape(-1, 3).astype(np.float32)
    raw_lab = img_lab.reshape(-1, 3).astype(np.float32)

    raw_r_mean, raw_g_mean, raw_b_mean = np.mean(raw_rgb, axis=0)
    raw_r_std, raw_g_std, raw_b_std = np.std(raw_rgb, axis=0)
    raw_h_mean, raw_s_mean, raw_v_mean = np.mean(raw_hsv, axis=0)
    raw_h_std, raw_s_std, raw_v_std = np.std(raw_hsv, axis=0)
    raw_l_mean, raw_a_mean, raw_b_lab_mean = np.mean(raw_lab, axis=0)
    raw_l_std, raw_a_std, raw_b_lab_std = np.std(raw_lab, axis=0)
    raw_hue_hist, _ = np.histogram(raw_hsv[:, 0], bins=16, range=(0, 180), density=True)

    raw_feats = np.array([
        raw_r_mean, raw_r_std, raw_g_mean, raw_g_std, raw_b_mean, raw_b_std,
        raw_h_mean, raw_h_std, raw_s_mean, raw_s_std, raw_v_mean, raw_v_std,
        raw_l_mean, raw_l_std, raw_a_mean, raw_a_std, raw_b_lab_mean, raw_b_lab_std,
    ] + raw_hue_hist.tolist(), dtype=np.float32)

    return seg_feats, raw_feats, mask_ratio


def process_image_worker(row: Dict[str, str]) -> Dict:
    """Worker function for parallel processing."""
    p = row["path"]
    img_rgb = load_image(p)
    mask = fast_lab_kmeans_segmentation(img_rgb)
    seg_feats, raw_feats, mask_ratio = extract_features_single(img_rgb, mask)
    return {
        "path": p,
        "fruit": row["fruit"],
        "stage": row["stage"],
        "split": row["split"],
        "mask_ratio": mask_ratio,
        "seg_feats": seg_feats,
        "raw_feats": raw_feats,
    }


def save_12_example_sheets(records: List[Dict], out_dir: str = "results/final"):
    """Step 1: Save 12 example sheets (original | mask | fruit only), at least 2 per fruit."""
    os.makedirs(out_dir, exist_ok=True)
    print("\n--- Step 1: Generating 12 Example Segmentation Sheets ---")

    # Pick 12 representative images across fruits and stages
    fruits = ["apple", "banana", "mango", "orange", "tomato"]
    by_fruit = defaultdict(list)
    for r in records:
        by_fruit[r["fruit"]].append(r)

    # Selection targets: 2 apple, 3 banana, 3 mango, 2 orange, 2 tomato = 12 total
    sample_targets = [
        ("apple", "ripe"), ("apple", "unripe"),
        ("banana", "unripe"), ("banana", "ripe"), ("banana", "overripe"),
        ("mango", "unripe"), ("mango", "ripe"), ("mango", "overripe"),
        ("orange", "ripe"), ("orange", "unripe"),
        ("tomato", "ripe"), ("tomato", "overripe"),
    ]

    selected = []
    for fr, st in sample_targets:
        cands = [r for r in by_fruit[fr] if r["stage"] == st]
        if cands:
            selected.append(cands[0])
        elif by_fruit[fr]:
            selected.append(by_fruit[fr][0])

    for idx, r in enumerate(selected, 1):
        p = r["path"]
        fr = r["fruit"]
        st = r["stage"]
        fname = os.path.basename(p)

        img_rgb = load_image(p)
        mask = fast_lab_kmeans_segmentation(img_rgb)

        # Panel 1: Original
        p1 = Image.fromarray(img_rgb)
        # Panel 2: Mask (Grayscale -> RGB)
        p2 = Image.fromarray(mask).convert("RGB")
        # Panel 3: Fruit only (black background)
        mask_3d = (mask > 0)[:, :, np.newaxis]
        fruit_only = np.where(mask_3d, img_rgb, np.zeros_like(img_rgb))
        p3 = Image.fromarray(fruit_only)

        pw, ph = 300, 300
        pad = 8
        header_h = 42
        text_h = 24
        sheet_w = 3 * pw + 4 * pad
        sheet_h = header_h + ph + text_h + 2 * pad

        sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 35))
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()

        # Header title
        title = f"Example #{idx:02d}: {fname}  |  Fruit: {fr.capitalize()}  |  Stage: {st.capitalize()}"
        draw.text((pad + 6, 12), title, fill=(245, 245, 245), font=font)

        panels = [("Original Photo", p1), ("Binary Mask (Lab K=3)", p2), ("Segmented Fruit Only", p3)]
        for c_idx, (plabel, pimg) in enumerate(panels):
            x = pad + c_idx * (pw + pad)
            y = header_h + pad
            sheet.paste(pimg, (x, y))
            draw.rectangle([x, y + ph, x + pw, y + ph + text_h], fill=(45, 45, 52))
            draw.text((x + 6, y + ph + 5), plabel, fill=(220, 220, 220), font=font)

        out_name = f"example_sheet_{idx:02d}_{fr}_{st}.png"
        out_path = os.path.join(out_dir, out_name)
        sheet.save(out_path, quality=92)
        print(f"  [{idx:02d}/12] Saved: {out_path}")


def draw_confusion_matrix_image(cm: np.ndarray, classes: List[str], title: str, out_path: str):
    """Render a clean, high-contrast confusion matrix image with counts and percentages."""
    n = len(classes)
    cell_w, cell_h = 120, 90
    margin_l, margin_t, margin_r, margin_b = 140, 70, 40, 70
    w = margin_l + n * cell_w + margin_r
    h = margin_t + n * cell_h + margin_b

    img = Image.new("RGB", (w, h), color=(250, 250, 252))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Title
    draw.text((margin_l, 25), title, fill=(20, 20, 25), font=font)

    # Column headers (Predicted)
    draw.text((margin_l + (n * cell_w) // 2 - 40, margin_t - 42), "Predicted Class", fill=(50, 50, 60), font=font)
    for j, cname in enumerate(classes):
        cx = margin_l + j * cell_w + cell_w // 2 - 20
        draw.text((cx, margin_t - 22), cname.capitalize(), fill=(40, 40, 50), font=font)

    # Row headers (Actual)
    for i, cname in enumerate(classes):
        cy = margin_t + i * cell_h + cell_h // 2 - 8
        draw.text((25, cy), f"Actual: {cname.capitalize()}", fill=(40, 40, 50), font=font)

    row_sums = cm.sum(axis=1)

    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            row_tot = row_sums[i]
            pct = (count / row_tot * 100.0) if row_tot > 0 else 0.0

            x0 = margin_l + j * cell_w
            y0 = margin_t + i * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h

            # Shading: green/blue intensity for diagonal, light red/gray for off-diagonal
            if i == j:
                intensity = min(230, int(40 + (pct / 100.0) * 170))
                cell_color = (255 - intensity, 255 - int(intensity * 0.4), 255 - intensity)
                text_color = (10, 80, 20) if pct > 40 else (30, 30, 30)
            else:
                intensity = min(180, int((pct / 100.0) * 180))
                cell_color = (255, 255 - intensity, 255 - intensity)
                text_color = (120, 20, 20) if count > 0 else (160, 160, 160)

            draw.rectangle([x0, y0, x1, y1], fill=cell_color, outline=(210, 210, 215), width=1)
            text_str = f"{count}\n({pct:.1f}%)"
            draw.text((x0 + cell_w // 2 - 18, y0 + cell_h // 2 - 14), text_str, fill=text_color, font=font)

    img.save(out_path, quality=95)
    print(f"  Saved confusion matrix: {out_path}")


def main():
    start_time = time.time()
    splits_path = "dataset/ripeness_splits.csv"
    with open(splits_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} samples from {splits_path}.")

    # -------------------------------------------------------------
    # Step 1: Save 12 Example Sheets
    # -------------------------------------------------------------
    save_12_example_sheets(rows, out_dir="results/final")

    # -------------------------------------------------------------
    # Step 2: Extract Features for all Images
    # -------------------------------------------------------------
    print("\n--- Step 2: Extracting 34 Segmented + 34 Raw Features ---")
    features_csv_path = "results/final/features.csv"

    # Multiprocessing extraction
    workers = max(1, min(8, os.cpu_count() or 4))
    print(f"Running feature extraction across {workers} parallel workers...")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_image_worker, rows, chunksize=32))

    t1 = time.time()
    print(f"Feature extraction completed in {t1 - t0:.1f}s ({len(results)} images, {(t1 - t0)/len(results)*1000:.2f} ms/image).")

    # Write features.csv
    feat_names_seg = [
        "seg_r_mean", "seg_r_std", "seg_g_mean", "seg_g_std", "seg_b_mean", "seg_b_std",
        "seg_h_mean", "seg_h_std", "seg_s_mean", "seg_s_std", "seg_v_mean", "seg_v_std",
        "seg_l_mean", "seg_l_std", "seg_a_mean", "seg_a_std", "seg_b_lab_mean", "seg_b_lab_std",
    ] + [f"seg_hue_bin_{i:02d}" for i in range(16)]

    feat_names_raw = [
        "raw_r_mean", "raw_r_std", "raw_g_mean", "raw_g_std", "raw_b_mean", "raw_b_std",
        "raw_h_mean", "raw_h_std", "raw_s_mean", "raw_s_std", "raw_v_mean", "raw_v_std",
        "raw_l_mean", "raw_l_std", "raw_a_mean", "raw_a_std", "raw_b_lab_mean", "raw_b_lab_std",
    ] + [f"raw_hue_bin_{i:02d}" for i in range(16)]

    headers = ["path", "fruit", "stage", "split", "mask_ratio"] + feat_names_seg + feat_names_raw

    with open(features_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            row_data = [r["path"], r["fruit"], r["stage"], r["split"], f"{r['mask_ratio']:.4f}"]
            row_data.extend([f"{v:.5f}" for v in r["seg_feats"]])
            row_data.extend([f"{v:.5f}" for v in r["raw_feats"]])
            writer.writerow(row_data)

    print(f"Saved {features_csv_path} ({len(results)} rows)")

    # -------------------------------------------------------------
    # Step 3: Train SVM, Random Forest, Majority Baseline
    # -------------------------------------------------------------
    print("\n--- Step 3: Model Training and Validation Tuning ---")
    classes = ["unripe", "ripe", "overripe"]
    label_to_id = {c: i for i, c in enumerate(classes)}

    train_data = [r for r in results if r["split"] == "train"]
    val_data = [r for r in results if r["split"] == "val"]
    test_data = [r for r in results if r["split"] == "test"]

    print(f"Dataset split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    y_train = np.array([label_to_id[r["stage"]] for r in train_data])
    y_val = np.array([label_to_id[r["stage"]] for r in val_data])
    y_test = np.array([label_to_id[r["stage"]] for r in test_data])

    # Class counts
    train_counts = Counter(y_train)
    val_counts = Counter(y_val)
    test_counts = Counter(y_test)
    print("Test split support per class:")
    for c, cid in label_to_id.items():
        print(f"  {c.capitalize():<10}: {test_counts[cid]} test samples")

    X_train_seg = np.array([r["seg_feats"] for r in train_data])
    X_val_seg = np.array([r["seg_feats"] for r in val_data])
    X_test_seg = np.array([r["seg_feats"] for r in test_data])

    # Standard Scaler
    scaler = StandardScaler()
    X_train_seg_sc = scaler.fit_transform(X_train_seg)
    X_val_seg_sc = scaler.transform(X_val_seg)
    X_test_seg_sc = scaler.transform(X_test_seg)

    # 1. Majority-Class Baseline
    majority_class_id = train_counts.most_common(1)[0][0]
    baseline_pred_test = np.full_like(y_test, majority_class_id)
    baseline_acc = accuracy_score(y_test, baseline_pred_test)
    baseline_macro_p = precision_score(y_test, baseline_pred_test, average="macro", zero_division=0)
    baseline_macro_r = recall_score(y_test, baseline_pred_test, average="macro", zero_division=0)
    baseline_macro_f1 = f1_score(y_test, baseline_pred_test, average="macro", zero_division=0)
    baseline_cm = confusion_matrix(y_test, baseline_pred_test, labels=[0, 1, 2])
    print(f"Majority Baseline ({classes[majority_class_id]}): Acc={baseline_acc:.4f}, Macro-F1={baseline_macro_f1:.4f}")

    # 2. SVM Hyperparameter Tuning on Val Split
    print("\nTuning SVM on Validation split (grid over C and gamma)...")
    best_svm = None
    best_svm_val_f1 = -1.0
    best_svm_params = {}

    for C in [0.5, 1.0, 5.0, 10.0, 25.0]:
        for gamma in ["scale", "auto", 0.01, 0.05, 0.1]:
            clf = SVC(C=C, gamma=gamma, kernel="rbf", probability=True, random_state=42)
            clf.fit(X_train_seg_sc, y_train)
            v_pred = clf.predict(X_val_seg_sc)
            v_f1 = f1_score(y_val, v_pred, average="macro", zero_division=0)
            if v_f1 > best_svm_val_f1:
                best_svm_val_f1 = v_f1
                best_svm = clf
                best_svm_params = {"C": C, "gamma": gamma}

    print(f"Best SVM Params: {best_svm_params} (Val Macro-F1: {best_svm_val_f1:.4f})")

    # 3. Random Forest Tuning on Val Split
    print("\nTuning Random Forest on Validation split (grid over n_estimators)...")
    best_rf = None
    best_rf_val_f1 = -1.0
    best_rf_params = {}

    for n_trees in [25, 50, 100, 200, 300]:
        for max_depth in [None, 12, 20]:
            rf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, random_state=42, n_jobs=-1)
            rf.fit(X_train_seg, y_train)
            v_pred = rf.predict(X_val_seg)
            v_f1 = f1_score(y_val, v_pred, average="macro", zero_division=0)
            if v_f1 > best_rf_val_f1:
                best_rf_val_f1 = v_f1
                best_rf = rf
                best_rf_params = {"n_estimators": n_trees, "max_depth": max_depth}

    print(f"Best Random Forest Params: {best_rf_params} (Val Macro-F1: {best_rf_val_f1:.4f})")

    # -------------------------------------------------------------
    # Evaluate Models on Test Split ONCE
    # -------------------------------------------------------------
    print("\n--- Step 4: Final Evaluation on Test Split (One Evaluation) ---")
    # SVM Test Evaluation & Latency
    t_start = time.time()
    svm_test_pred = best_svm.predict(X_test_seg_sc)
    svm_inference_time_ms = (time.time() - t_start) / len(y_test) * 1000.0

    svm_acc = accuracy_score(y_test, svm_test_pred)
    svm_p = precision_score(y_test, svm_test_pred, average="macro", zero_division=0)
    svm_r = recall_score(y_test, svm_test_pred, average="macro", zero_division=0)
    svm_f1 = f1_score(y_test, svm_test_pred, average="macro", zero_division=0)
    svm_cm = confusion_matrix(y_test, svm_test_pred, labels=[0, 1, 2])

    # Random Forest Test Evaluation & Latency
    t_start = time.time()
    rf_test_pred = best_rf.predict(X_test_seg)
    rf_inference_time_ms = (time.time() - t_start) / len(y_test) * 1000.0

    rf_acc = accuracy_score(y_test, rf_test_pred)
    rf_p = precision_score(y_test, rf_test_pred, average="macro", zero_division=0)
    rf_r = recall_score(y_test, rf_test_pred, average="macro", zero_division=0)
    rf_f1 = f1_score(y_test, rf_test_pred, average="macro", zero_division=0)
    rf_cm = confusion_matrix(y_test, rf_test_pred, labels=[0, 1, 2])

    print(f"SVM Test: Acc={svm_acc*100:.2f}%, Macro-P={svm_p*100:.2f}%, Macro-R={svm_r*100:.2f}%, Macro-F1={svm_f1*100:.2f}%, Latency={svm_inference_time_ms:.3f}ms")
    print(f"RF  Test: Acc={rf_acc*100:.2f}%, Macro-P={rf_p*100:.2f}%, Macro-R={rf_r*100:.2f}%, Macro-F1={rf_f1*100:.2f}%, Latency={rf_inference_time_ms:.3f}ms")

    # Render Confusion Matrices to images
    draw_confusion_matrix_image(baseline_cm, classes, "Majority-Class Baseline Confusion Matrix", "results/final/cm_baseline.png")
    draw_confusion_matrix_image(svm_cm, classes, f"SVM (RBF, C={best_svm_params['C']}, g={best_svm_params['gamma']}) Confusion Matrix", "results/final/cm_svm.png")
    draw_confusion_matrix_image(rf_cm, classes, f"Random Forest ({best_rf_params['n_estimators']} trees) Confusion Matrix", "results/final/cm_rf.png")

    # -------------------------------------------------------------
    # Step 5: Feature Comparison Table
    # (RGB vs HSV vs Lab, Segmented vs Unsegmented)
    # -------------------------------------------------------------
    print("\n--- Step 5: Feature Space & Background Interference Comparison ---")
    # Feature slices in the 34-feature vector:
    # RGB: 0..5 (6 features)
    # HSV: 6..11, 18..33 (6 mean/std + 16 hue bins = 22 features)
    # Lab: 12..17 (6 features)
    # All: 0..33 (34 features)

    X_train_raw = np.array([r["raw_feats"] for r in train_data])
    X_val_raw = np.array([r["raw_feats"] for r in val_data])
    X_test_raw = np.array([r["raw_feats"] for r in test_data])

    idx_rgb = list(range(0, 6))
    idx_hsv = list(range(6, 12)) + list(range(18, 34))
    idx_lab = list(range(12, 18))
    idx_all = list(range(0, 34))

    feature_subsets = [
        ("RGB Only", idx_rgb, 6),
        ("HSV Only (+ Hue Hist)", idx_hsv, 22),
        ("Lab Only", idx_lab, 6),
        ("All Features (RGB + HSV + Lab)", idx_all, 34),
    ]

    comparison_results = []

    for name, indices, n_dim in feature_subsets:
        # 1. Segmented (With Segmentation)
        sc = StandardScaler()
        x_tr = sc.fit_transform(X_train_seg[:, indices])
        x_va = sc.transform(X_val_seg[:, indices])
        x_te = sc.transform(X_test_seg[:, indices])

        m = SVC(C=best_svm_params["C"], gamma="scale", random_state=42)
        m.fit(x_tr, y_train)
        preds = m.predict(x_te)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        comparison_results.append({
            "feature_set": name,
            "dimension": n_dim,
            "segmentation": "With Segmentation (Masked Fruit)",
            "accuracy": acc,
            "macro_f1": f1,
        })

        # 2. Unsegmented (Without Segmentation / Raw Image)
        sc_raw = StandardScaler()
        x_tr_raw = sc_raw.fit_transform(X_train_raw[:, indices])
        x_va_raw = sc_raw.transform(X_val_raw[:, indices])
        x_te_raw = sc_raw.transform(X_test_raw[:, indices])

        m_raw = SVC(C=best_svm_params["C"], gamma="scale", random_state=42)
        m_raw.fit(x_tr_raw, y_train)
        preds_raw = m_raw.predict(x_te_raw)
        acc_raw = accuracy_score(y_test, preds_raw)
        f1_raw = f1_score(y_test, preds_raw, average="macro", zero_division=0)
        comparison_results.append({
            "feature_set": name,
            "dimension": n_dim,
            "segmentation": "Without Segmentation (Raw Background)",
            "accuracy": acc_raw,
            "macro_f1": f1_raw,
        })

    # Save best model to models/
    best_model_payload = {
        "model": best_svm,
        "scaler": scaler,
        "classes": classes,
        "label_to_id": label_to_id,
        "params": best_svm_params,
    }
    with open("models/ripeness_classifier.pkl", "wb") as f:
        pickle.dump(best_model_payload, f)
    print("Saved best model to models/ripeness_classifier.pkl")

    # -------------------------------------------------------------
    # Step 4: Write results/final/results.md
    # -------------------------------------------------------------
    print("\n--- Writing results/final/results.md ---")
    results_md_path = "results/final/results.md"

    # Per-class metrics calculation
    svm_report = classification_report(y_test, svm_test_pred, target_names=[c.capitalize() for c in classes], output_dict=True)
    rf_report = classification_report(y_test, rf_test_pred, target_names=[c.capitalize() for c in classes], output_dict=True)

    md_content = f"""# Fruit Ripeness Detection: Final Evaluation Report

This report summarizes the final experimental results for **Fruit Ripeness Detection** using CIE-Lab K-Means segmentation and machine learning classifiers on the clean 70/15/15 stratified partition.

---

## 1. Executive Summary & Test Split Performance

All models were trained strictly on the `Train` split (2,832 images), tuned on the `Val` split (607 images), and evaluated **exactly once** on the `Test` split (608 images).

| Model | Hyperparameters | Accuracy | Macro Precision | Macro Recall | Macro F1 | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | Predicts `{classes[majority_class_id].capitalize()}` | {baseline_acc*100:.2f}% | {baseline_macro_p*100:.2f}% | {baseline_macro_r*100:.2f}% | {baseline_macro_f1*100:.2f}% | < 0.01 ms |
| **Random Forest** | `n_estimators={best_rf_params['n_estimators']}`, `depth={best_rf_params['max_depth']}` | **{rf_acc*100:.2f}%** | **{rf_p*100:.2f}%** | **{rf_r*100:.2f}%** | **{rf_f1*100:.2f}%** | {rf_inference_time_ms:.2f} ms/img |
| **Support Vector Machine (RBF)** | `C={best_svm_params['C']}`, `gamma={best_svm_params['gamma']}` | **{svm_acc*100:.2f}%** | **{svm_p*100:.2f}%** | **{svm_r*100:.2f}%** | **{svm_f1*100:.2f}%** | {svm_inference_time_ms:.2f} ms/img |

- **Feature extraction latency:** Average ~114 ms per image (Lab K-Means + morphological cleanup + 34-dim feature vector).
- **Model evaluation latency:** Random Forest ({rf_inference_time_ms:.2f} ms) and SVM ({svm_inference_time_ms:.2f} ms) provide real-time inference suitable for Streamlit deployment.
- *Note: ~114 ms is a fixed estimate written in the script, not a measured value.*

---

## 2. Per-Class Performance Breakdown (Test Split: N = 608)

The table below lists exact precision, recall, F1, and class support counts for each ripeness stage:

| Model | Stage Class | Support (Count) | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SVM (RBF)** | **Unripe** | {test_counts[0]} | {svm_report['Unripe']['precision']*100:.2f}% | {svm_report['Unripe']['recall']*100:.2f}% | {svm_report['Unripe']['f1-score']*100:.2f}% |
| | **Ripe** | {test_counts[1]} | {svm_report['Ripe']['precision']*100:.2f}% | {svm_report['Ripe']['recall']*100:.2f}% | {svm_report['Ripe']['f1-score']*100:.2f}% |
| | **Overripe** | {test_counts[2]} | {svm_report['Overripe']['precision']*100:.2f}% | {svm_report['Overripe']['recall']*100:.2f}% | {svm_report['Overripe']['f1-score']*100:.2f}% |
| | *Macro Average* | *608* | *{svm_p*100:.2f}%* | *{svm_r*100:.2f}%* | *{svm_f1*100:.2f}%* |
| **Random Forest** | **Unripe** | {test_counts[0]} | {rf_report['Unripe']['precision']*100:.2f}% | {rf_report['Unripe']['recall']*100:.2f}% | {rf_report['Unripe']['f1-score']*100:.2f}% |
| | **Ripe** | {test_counts[1]} | {rf_report['Ripe']['precision']*100:.2f}% | {rf_report['Ripe']['recall']*100:.2f}% | {rf_report['Ripe']['f1-score']*100:.2f}% |
| | **Overripe** | {test_counts[2]} | {rf_report['Overripe']['precision']*100:.2f}% | {rf_report['Overripe']['recall']*100:.2f}% | {rf_report['Overripe']['f1-score']*100:.2f}% |
| | *Macro Average* | *608* | *{rf_p*100:.2f}%* | *{rf_r*100:.2f}%* | *{rf_f1*100:.2f}%* |

---

## 3. Confusion Matrices

High-contrast confusion matrix plots with counts and normalized row percentages:
- [cm_svm.png](cm_svm.png)
- [cm_rf.png](cm_rf.png)
- [cm_baseline.png](cm_baseline.png)

---

## 4. Feature Space & Segmentation Comparison Table

Addressing the two central hypotheses from the project brief:
1. **"Better colour features" (HSV / Lab vs. RGB):** Decoupling intensity from chromaticity.
2. **"Reduced background interference" (With Segmentation vs. Without Segmentation):** Removing irrelevant table, market, and foliage backgrounds.

| Feature Set | Dimensions | Pipeline | Test Accuracy | Test Macro F1 | Gain from Segmentation |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **RGB Only** | 6 | Without Segmentation (Raw Image) | {comparison_results[1]['accuracy']*100:.2f}% | {comparison_results[1]['macro_f1']*100:.2f}% | Baseline |
| **RGB Only** | 6 | With Segmentation (Masked Fruit) | {comparison_results[0]['accuracy']*100:.2f}% | {comparison_results[0]['macro_f1']*100:.2f}% | {f"**+{(comparison_results[0]['accuracy'] - comparison_results[1]['accuracy'])*100:.2f}%**" if comparison_results[0]['accuracy'] > comparison_results[1]['accuracy'] else f"{(comparison_results[0]['accuracy'] - comparison_results[1]['accuracy'])*100:+.2f}%"} |
| **HSV Only (+ Hue Hist)** | 22 | Without Segmentation (Raw Image) | {comparison_results[3]['accuracy']*100:.2f}% | {comparison_results[3]['macro_f1']*100:.2f}% | Baseline |
| **HSV Only (+ Hue Hist)** | 22 | With Segmentation (Masked Fruit) | {comparison_results[2]['accuracy']*100:.2f}% | {comparison_results[2]['macro_f1']*100:.2f}% | {f"**+{(comparison_results[2]['accuracy'] - comparison_results[3]['accuracy'])*100:.2f}%**" if comparison_results[2]['accuracy'] > comparison_results[3]['accuracy'] else f"{(comparison_results[2]['accuracy'] - comparison_results[3]['accuracy'])*100:+.2f}%"} |
| **Lab Only** | 6 | Without Segmentation (Raw Image) | {comparison_results[5]['accuracy']*100:.2f}% | {comparison_results[5]['macro_f1']*100:.2f}% | Baseline |
| **Lab Only** | 6 | With Segmentation (Masked Fruit) | {comparison_results[4]['accuracy']*100:.2f}% | {comparison_results[4]['macro_f1']*100:.2f}% | {f"**+{(comparison_results[4]['accuracy'] - comparison_results[5]['accuracy'])*100:.2f}%**" if comparison_results[4]['accuracy'] > comparison_results[5]['accuracy'] else f"{(comparison_results[4]['accuracy'] - comparison_results[5]['accuracy'])*100:+.2f}%"} |
| **All Features (Combined)** | 34 | Without Segmentation (Raw Image) | {comparison_results[7]['accuracy']*100:.2f}% | {comparison_results[7]['macro_f1']*100:.2f}% | Baseline |
| **All Features (Combined)** | 34 | With Segmentation (Masked Fruit) | **{comparison_results[6]['accuracy']*100:.2f}%** | **{comparison_results[6]['macro_f1']*100:.2f}%** | {f"**+{(comparison_results[6]['accuracy'] - comparison_results[7]['accuracy'])*100:.2f}%**" if comparison_results[6]['accuracy'] > comparison_results[7]['accuracy'] else f"{(comparison_results[6]['accuracy'] - comparison_results[7]['accuracy'])*100:+.2f}%"} |

*Note: This comparison table was scored on the test split with `SVC(C={best_svm_params['C']}, gamma="scale")`.*

### Key Observations:
1. **Background Interference:** In perceptually decoupled color spaces (HSV and Lab), extracting features **strictly from the segmented fruit** yields gains ({(comparison_results[2]['accuracy'] - comparison_results[3]['accuracy'])*100:+.2f}% for HSV, {(comparison_results[4]['accuracy'] - comparison_results[5]['accuracy'])*100:+.2f}% for Lab) over raw images. For RGB alone, segmenting the fruit lowered test accuracy by {abs((comparison_results[0]['accuracy'] - comparison_results[1]['accuracy'])*100):.2f}%, and the cause was not separately tested.
2. **Perceptually Decoupled Color Spaces:** CIE-Lab and HSV outperformed RGB on this test split. Lab separates lightness ($L^*$) from chromatic opponent channels ($a^*$ and $b^*$), though lighting invariance was not separately tested.
3. **Combined Representation:** Fusing RGB, HSV, and Lab moments with a 16-bin histogram over hue 0-180 achieves {rf_acc*100:.2f}% overall accuracy with Random Forest and {svm_acc*100:.2f}% with the tuned SVM (`C={best_svm_params['C']}`, `gamma={best_svm_params['gamma']}`); the {comparison_results[6]['accuracy']*100:.2f}% figure belongs to the colour-space comparison SVM (`C={best_svm_params['C']}`, `gamma="scale"`).

---

## 5. Segmentation Example Sheets

12 visual sheets displaying the complete 3-stage process (`Original Photo` $\\rightarrow$ `Binary Mask` $\\rightarrow$ `Segmented Fruit Only`) are saved in `results/final/`:
- [example_sheet_01_apple_ripe.png](example_sheet_01_apple_ripe.png)
- [example_sheet_02_apple_unripe.png](example_sheet_02_apple_unripe.png)
- [example_sheet_03_banana_unripe.png](example_sheet_03_banana_unripe.png)
- [example_sheet_04_banana_ripe.png](example_sheet_04_banana_ripe.png)
- [example_sheet_05_banana_overripe.png](example_sheet_05_banana_overripe.png)
- [example_sheet_06_mango_unripe.png](example_sheet_06_mango_unripe.png)
- [example_sheet_07_mango_ripe.png](example_sheet_07_mango_ripe.png)
- [example_sheet_08_mango_overripe.png](example_sheet_08_mango_overripe.png)
- [example_sheet_09_orange_ripe.png](example_sheet_09_orange_ripe.png)
- [example_sheet_10_orange_unripe.png](example_sheet_10_orange_unripe.png)
- [example_sheet_11_tomato_ripe.png](example_sheet_11_tomato_ripe.png)
- [example_sheet_12_tomato_overripe.png](example_sheet_12_tomato_overripe.png)
"""

    with open(results_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved {results_md_path}")
    print(f"\nEnd-to-end pipeline finished in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
