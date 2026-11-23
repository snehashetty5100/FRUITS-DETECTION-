"""Phase 4: Color and Texture Feature Extraction Pipeline

Computes comprehensive handcrafted features inside the segmented K-Means fruit mask
(excluding letterbox padding), with an automated fallback to the content bounding box.

Feature Categories:
1. Color Moments (H, S, V and L, a, b): Mean, Standard Deviation, Skewness (18 features).
2. Hue Circular Histogram: 16 bins spanning [0, 180) (16 features).
3. CIE-Lab 2D Histogram: 8x8 bins spanning [-128, 128] for a* and b* (64 features).
4. Ripeness Color Fractions: Green, Yellow, Red, Dark-Brown pixel proportions (4 features).
5. Texture: Local Standard Deviation, Sobel Edge Density, NumPy GLCM Contrast & Homogeneity (4 features).
6. One-hot fruit variety encoding (6 binary features).
7. Metadata & Quality: filename, fruit, stage (label), fallback flag (1 if fallback used, 0 otherwise).

No cv2, no pandas, no skimage. Uses Pillow, NumPy, SciPy, and the standard csv module.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import scipy.stats as stats

from src.preprocessing import rgb_to_hsv, rgb_to_lab, load_image
from src.segmentation import (
    segment_by_kmeans,
    isolate_foreground_cluster,
    apply_morphological_cleanup,
    get_valid_content_bbox
)

FRUITS = ["apple", "banana", "mango", "orange", "papaya", "tomato"]
STAGES = ["unripe", "ripe", "overripe"]


def compute_moments(values: np.ndarray) -> tuple:
    """Compute mean, standard deviation, and skewness of a 1D array."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    if std_val < 1e-6:
        skew_val = 0.0
    else:
        # Fisher-Pearson skewness coefficient
        skew_val = float(stats.skew(values, bias=False))
        if np.isnan(skew_val):
            skew_val = 0.0
    return mean_val, std_val, skew_val


def compute_hue_histogram(h_values: np.ndarray, num_bins: int = 16) -> list:
    """Compute normalized 16-bin circular histogram for Hue in [0, 180)."""
    if len(h_values) == 0:
        return [0.0] * num_bins
    counts, _ = np.histogram(h_values, bins=num_bins, range=(0, 180))
    total = np.sum(counts)
    if total > 0:
        norm_counts = (counts / total).tolist()
    else:
        norm_counts = [0.0] * num_bins
    return [round(float(v), 5) for v in norm_counts]


def compute_lab_2d_histogram(a_values: np.ndarray, b_values: np.ndarray, bins: int = 8) -> list:
    """Compute normalized 8x8 2D histogram for CIE-Lab a* and b* in [-128, 128]."""
    if len(a_values) == 0 or len(b_values) == 0:
        return [0.0] * (bins * bins)
    counts, _, _ = np.histogram2d(
        a_values, b_values,
        bins=bins,
        range=[[-128, 128], [-128, 128]]
    )
    total = np.sum(counts)
    if total > 0:
        norm_counts = (counts / total).flatten().tolist()
    else:
        norm_counts = [0.0] * (bins * bins)
    return [round(float(v), 5) for v in norm_counts]


def compute_color_fractions(hsv_pixels: np.ndarray) -> dict:
    """Calculate the fraction of green, yellow, red, and dark-brown pixels."""
    total = len(hsv_pixels)
    if total == 0:
        return {"frac_green": 0.0, "frac_yellow": 0.0, "frac_red": 0.0, "frac_dark_brown": 0.0}

    h = hsv_pixels[:, 0]
    s = hsv_pixels[:, 1]
    v = hsv_pixels[:, 2]

    # Green: Hue 36..85 with sufficient saturation and brightness
    green_mask = (h >= 36) & (h <= 85) & (s >= 25) & (v >= 25)

    # Yellow: Hue 18..35 with saturation and brightness
    yellow_mask = (h >= 18) & (h < 36) & (s >= 25) & (v >= 40)

    # Red: Hue 0..17 or 165..179 with saturation and brightness
    red_mask = ((h < 18) | (h >= 165)) & (s >= 25) & (v >= 30)

    # Dark-Brown / Mottled overripe:
    # Warm amber/brown hues (10..30) with low value, or very dark mottled patches
    dark_brown_mask = (
        ((h >= 10) & (h <= 30) & (s >= 20) & (v < 110) & (v >= 20)) |
        ((v < 40) & (v >= 10))
    )

    return {
        "frac_green": round(float(np.sum(green_mask) / total), 5),
        "frac_yellow": round(float(np.sum(yellow_mask) / total), 5),
        "frac_red": round(float(np.sum(red_mask) / total), 5),
        "frac_dark_brown": round(float(np.sum(dark_brown_mask) / total), 5),
    }


def compute_numpy_glcm_features(img_gray: np.ndarray, mask: np.ndarray, levels: int = 16) -> tuple:
    """Compute GLCM Contrast and Homogeneity using NumPy (horizontal displacement d=1, theta=0)."""
    # Quantize grayscale image to discrete levels [0..levels-1]
    quant = np.clip((img_gray / (256 / levels)).astype(int), 0, levels - 1)

    # Find valid horizontal pairs where both (r, c) and (r, c+1) are within the mask
    left_mask = mask[:, :-1]
    right_mask = mask[:, 1:]
    pair_mask = left_mask & right_mask

    if np.sum(pair_mask) == 0:
        return 0.0, 1.0

    p_left = quant[:, :-1][pair_mask]
    p_right = quant[:, 1:][pair_mask]

    # Accumulate 2D co-occurrence matrix
    glcm, _, _ = np.histogram2d(p_left, p_right, bins=levels, range=[[0, levels], [0, levels]])
    total = np.sum(glcm)
    if total > 0:
        glcm = glcm / total
    else:
        return 0.0, 1.0

    # Coordinate grids
    i_indices, j_indices = np.indices((levels, levels))
    diff_sq = (i_indices - j_indices) ** 2

    # Contrast: sum( |i - j|^2 * P(i,j) )
    contrast = float(np.sum(diff_sq * glcm))

    # Homogeneity: sum( P(i,j) / (1 + |i - j|^2) )
    homogeneity = float(np.sum(glcm / (1.0 + diff_sq)))

    return round(contrast, 5), round(homogeneity, 5)


def compute_texture_features(img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Compute local standard deviation, edge density, and GLCM metrics inside the mask."""
    # Convert to luminance grayscale using standard ITU-R 601 coefficients
    img_gray = (0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]).astype(np.float64)

    if np.sum(mask) == 0:
        return {
            "local_std_mean": 0.0,
            "edge_density": 0.0,
            "glcm_contrast": 0.0,
            "glcm_homogeneity": 1.0
        }

    # 1. Local standard deviation in 5x5 window: Var = E[X^2] - (E[X])^2
    mean_val = ndi.uniform_filter(img_gray, size=5)
    mean_sq = ndi.uniform_filter(img_gray ** 2, size=5)
    local_var = np.maximum(0.0, mean_sq - (mean_val ** 2))
    local_std = np.sqrt(local_var)
    local_std_mean = float(np.mean(local_std[mask]))

    # 2. Edge density via Sobel filter
    sobel_h = ndi.sobel(img_gray, axis=0)
    sobel_v = ndi.sobel(img_gray, axis=1)
    edge_mag = np.hypot(sobel_h, sobel_v)
    edge_density = float(np.mean(edge_mag[mask] > 25.0))

    # 3. NumPy GLCM contrast and homogeneity
    glcm_contrast, glcm_homogeneity = compute_numpy_glcm_features(img_gray, mask, levels=16)

    return {
        "local_std_mean": round(local_std_mean, 5),
        "edge_density": round(edge_density, 5),
        "glcm_contrast": glcm_contrast,
        "glcm_homogeneity": glcm_homogeneity
    }


def extract_features_single_image(img_rgb: np.ndarray, fruit: str, stage: str, filename: str, use_full_content: bool = False) -> dict:
    """Extract full feature dictionary for one image.
    
    If use_full_content is True (ablation mode), skips K-Means masking and computes
    features on the entire active content box.
    """
    rmin, rmax, cmin, cmax, valid_content_mask = get_valid_content_bbox(img_rgb)
    content_area = (rmax - rmin + 1) * (cmax - cmin + 1)

    fallback = 0

    if use_full_content:
        # Ablation mode: full content bounding box
        effective_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
        effective_mask[rmin:rmax+1, cmin:cmax+1] = True
    else:
        # K-Means (K=3) segmentation with multi-factor foreground scoring
        _, cluster_map, bbox = segment_by_kmeans(img_rgb, k=3, max_iter=25)
        raw_mask = isolate_foreground_cluster(cluster_map, img_rgb, bbox=bbox)
        clean_mask = apply_morphological_cleanup(raw_mask, radius=3, keep_largest_only=True)

        # Enforce valid content boundary
        clean_bool = (clean_mask > 0) & valid_content_mask

        mask_area = int(np.sum(clean_bool))
        mask_ratio = mask_area / content_area if content_area > 0 else 0.0

        # Fallback check: if mask area is under 2% or over 90% of content box
        if mask_ratio < 0.02 or mask_ratio > 0.90:
            effective_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
            effective_mask[rmin:rmax+1, cmin:cmax+1] = True
            fallback = 1
        else:
            effective_mask = clean_bool
            fallback = 0

    # Color space conversions
    hsv_img = rgb_to_hsv(img_rgb)
    lab_img = rgb_to_lab(img_rgb)

    # Pixel extractions inside the effective mask
    h_vals = hsv_img[:, :, 0][effective_mask]
    s_vals = hsv_img[:, :, 1][effective_mask]
    v_vals = hsv_img[:, :, 2][effective_mask]
    hsv_pixels = np.column_stack([h_vals, s_vals, v_vals])

    l_vals = lab_img[:, :, 0][effective_mask]
    a_vals = lab_img[:, :, 1][effective_mask]
    b_vals = lab_img[:, :, 2][effective_mask]

    # 1. Color Moments (18 features)
    h_mean, h_std, h_skew = compute_moments(h_vals)
    s_mean, s_std, s_skew = compute_moments(s_vals)
    v_mean, v_std, v_skew = compute_moments(v_vals)
    l_mean, l_std, l_skew = compute_moments(l_vals)
    a_mean, a_std, a_skew = compute_moments(a_vals)
    b_mean, b_std, b_skew = compute_moments(b_vals)

    # 2. Hue Circular Histogram (16 features)
    hue_hist = compute_hue_histogram(h_vals, num_bins=16)

    # 3. Lab 2D Histogram (64 features)
    lab_2d_hist = compute_lab_2d_histogram(a_vals, b_vals, bins=8)

    # 4. Color Fractions (4 features)
    fractions = compute_color_fractions(hsv_pixels)

    # 5. Texture Features (4 features)
    texture = compute_texture_features(img_rgb, effective_mask)

    # Assemble feature vector
    feat_dict = {
        "filename": filename,
        "fruit": fruit,
        "stage": stage,
        "fallback": fallback,
        # One-hot fruit encoding
        "is_apple": 1 if fruit == "apple" else 0,
        "is_banana": 1 if fruit == "banana" else 0,
        "is_mango": 1 if fruit == "mango" else 0,
        "is_orange": 1 if fruit == "orange" else 0,
        "is_papaya": 1 if fruit == "papaya" else 0,
        "is_tomato": 1 if fruit == "tomato" else 0,
        # Moments
        "h_mean": round(h_mean, 4), "h_std": round(h_std, 4), "h_skew": round(h_skew, 4),
        "s_mean": round(s_mean, 4), "s_std": round(s_std, 4), "s_skew": round(s_skew, 4),
        "v_mean": round(v_mean, 4), "v_std": round(v_std, 4), "v_skew": round(v_skew, 4),
        "l_mean": round(l_mean, 4), "l_std": round(l_std, 4), "l_skew": round(l_skew, 4),
        "a_mean": round(a_mean, 4), "a_std": round(a_std, 4), "a_skew": round(a_skew, 4),
        "b_mean": round(b_mean, 4), "b_std": round(b_std, 4), "b_skew": round(b_skew, 4),
        # Color Fractions
        "frac_green": fractions["frac_green"],
        "frac_yellow": fractions["frac_yellow"],
        "frac_red": fractions["frac_red"],
        "frac_dark_brown": fractions["frac_dark_brown"],
        # Texture
        "local_std_mean": texture["local_std_mean"],
        "edge_density": texture["edge_density"],
        "glcm_contrast": texture["glcm_contrast"],
        "glcm_homogeneity": texture["glcm_homogeneity"],
    }

    # Hue histogram features (hue_bin_00 .. hue_bin_15)
    for idx, val in enumerate(hue_hist):
        feat_dict[f"hue_bin_{idx:02d}"] = val

    # Lab 2D histogram features (lab2d_00 .. lab2d_63)
    for idx, val in enumerate(lab_2d_hist):
        feat_dict[f"lab2d_{idx:02d}"] = val

    return feat_dict


def process_dataset_features(
    input_dir: str = "dataset/fruits360/Training",
    out_csv: str = "results/phase4/features.csv",
    use_full_content: bool = False
) -> list:
    """Extract features for all processed images and export to CSV."""
    in_path = Path(input_dir)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    image_files = sorted(in_path.glob("*/*/*.png"))

    mode_name = "Full-Image Content Box (Ablation)" if use_full_content else "Segmented K-Means Mask"
    print(f"=== Extracting Features ({mode_name}) for {len(image_files)} Images ===")

    fallback_count = 0

    for idx, p in enumerate(image_files, 1):
        filename = p.name
        stage = p.parent.name
        fruit = p.parent.parent.name

        img_rgb = load_image(str(p))
        feat = extract_features_single_image(
            img_rgb, fruit=fruit, stage=stage, filename=filename, use_full_content=use_full_content
        )
        if feat["fallback"] == 1:
            fallback_count += 1
        records.append(feat)

        if idx % 10 == 0 or idx == len(image_files):
            print(f"  Processed {idx}/{len(image_files)} images...")

    if not records:
        print("No images found to process.")
        return []

    # Write to CSV
    fieldnames = list(records[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[✓] Saved {len(records)} feature rows to {out_path}.")
    if not use_full_content:
        print(f"    Fallback to full content box triggered on {fallback_count}/{len(records)} images.")
    return records


if __name__ == "__main__":
    # 1. Main extraction (inside K-Means mask)
    process_dataset_features(
        input_dir="dataset/fruits360/Training",
        out_csv="results/phase4/features.csv",
        use_full_content=False
    )
    # 2. Ablation extraction (full content box)
    process_dataset_features(
        input_dir="dataset/fruits360/Training",
        out_csv="results/phase4/features_full.csv",
        use_full_content=True
    )
