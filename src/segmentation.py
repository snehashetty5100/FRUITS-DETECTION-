"""Module 2: Segmentation

Pillow + NumPy + SciPy implementation of:
1. Configurable HSV color thresholding (with unripe, ripe, and overripe ranges).
2. Vectorized K-Means clustering in RGB, HSV, and Lab color spaces with K-Means++ initialization.
3. Multi-factor foreground cluster isolation (Border Ratio, Center Bias, Color Saturation).
4. Morphological post-processing (Opening, Closing, Hole Filling, Largest Connected Component).
5. Evaluation metrics (IoU, Sørensen-Dice).
"""

from typing import Tuple, Dict, Optional, Union, List
import numpy as np
import scipy.ndimage as ndi
from src.preprocessing import rgb_to_hsv, rgb_to_lab


# Baseline HSV Threshold Configuration (H: 0-179, S: 0-255, V: 0-255)
# Note: Initial uncalibrated baseline; to be evaluated against hand-drawn masks.
HSV_CONFIG = {
    # Generic stage baselines
    "general_unripe": ((18, 25, 20), (95, 255, 255)),    # Broad green/olive
    "general_ripe": ((5, 40, 40), (45, 255, 255)),       # Warm yellow/orange/red
    "general_overripe": ((8, 20, 15), (35, 200, 200)),   # Dark brown/mottled

    # Fruit-specific initial baselines
    "apple": {
        "unripe": ((25, 25, 25), (95, 255, 255)),
        "ripe": [((0, 50, 40), (12, 255, 255)), ((165, 50, 40), (179, 255, 255))],
        "overripe": ((5, 25, 20), (30, 200, 180)),
    },
    "banana": {
        "unripe": ((30, 35, 30), (85, 255, 255)),
        "ripe": ((18, 45, 50), (38, 255, 255)),
        "overripe": ((10, 25, 20), (35, 210, 180)),
    },
    "mango": {
        "unripe": ((18, 25, 20), (90, 255, 255)),
        "ripe": ((10, 40, 45), (38, 255, 255)),
        "overripe": ((8, 20, 15), (30, 190, 160)),
    },
    "orange": {
        "unripe": ((18, 25, 30), (90, 255, 255)),
        "ripe": ((8, 60, 60), (28, 255, 255)),
        "overripe": ((5, 20, 20), (30, 180, 160)),
    },
    "tomato": {
        "unripe": ((25, 30, 30), (95, 255, 255)),
        "ripe": [((0, 60, 40), (12, 255, 255)), ((165, 60, 40), (179, 255, 255))],
        "overripe": ((0, 35, 25), (25, 200, 180)),
    },
}


def segment_by_color_threshold(
    image_hsv: np.ndarray,
    lower_bound: Tuple[int, int, int],
    upper_bound: Tuple[int, int, int]
) -> np.ndarray:
    """Create a binary mask (uint8 0 or 255) isolating pixels within HSV thresholds."""
    h, s, v = image_hsv[..., 0], image_hsv[..., 1], image_hsv[..., 2]

    if lower_bound[0] > upper_bound[0]:
        h_mask = (h >= lower_bound[0]) | (h <= upper_bound[0])
    else:
        h_mask = (h >= lower_bound[0]) & (h <= upper_bound[0])

    s_mask = (s >= lower_bound[1]) & (s <= upper_bound[1])
    v_mask = (v >= lower_bound[2]) & (v <= upper_bound[2])

    binary_mask = (h_mask & s_mask & v_mask)
    return binary_mask.astype(np.uint8) * 255


def segment_by_configured_hsv(
    image_rgb: np.ndarray,
    fruit: str,
    stage: str
) -> np.ndarray:
    """Segment image using fruit- and stage-specific HSV configuration."""
    img_hsv = rgb_to_hsv(image_rgb)
    fruit_key = fruit.lower()
    stage_key = stage.lower()

    if fruit_key in HSV_CONFIG and stage_key in HSV_CONFIG[fruit_key]:
        cfg = HSV_CONFIG[fruit_key][stage_key]
    else:
        cfg = HSV_CONFIG.get(f"general_{stage_key}", ((0, 30, 20), (179, 255, 255)))

    if isinstance(cfg, list):
        combined = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        for lower, upper in cfg:
            m = segment_by_color_threshold(img_hsv, lower, upper)
            combined = np.bitwise_or(combined, m)
        return combined

    lower, upper = cfg
    return segment_by_color_threshold(img_hsv, lower, upper)


def init_kmeans_plus_plus(pixels: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """Initialize centroids using K-Means++ to guarantee distinct initial centers."""
    n = len(pixels)
    centroids = [pixels[rng.choice(n)]]
    for _ in range(1, k):
        sq_dists = np.min([np.sum((pixels - c) ** 2, axis=1) for c in centroids], axis=0)
        total = np.sum(sq_dists)
        probs = sq_dists / total if total > 1e-6 else np.ones(n) / n
        next_c = pixels[rng.choice(n, p=probs)]
        centroids.append(next_c)
    return np.array(centroids, dtype=np.float32)


def segment_by_kmeans(
    image_feature: np.ndarray,
    k: int = 3,
    max_iter: int = 25,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment image in feature space (RGB, HSV, or Lab) using K-Means++.

    Args:
        image_feature: (H, W, 3) image in target color space.
        k: Number of clusters (e.g. 2, 3, 4, 5).
        max_iter: Max clustering iterations.
        random_state: Seed for reproducibility.

    Returns:
        tuple: (quantized_image, cluster_labels_2d)
    """
    h, w, c = image_feature.shape
    pixels = image_feature.reshape(-1, c).astype(np.float32)
    rng = np.random.RandomState(random_state)

    centroids = init_kmeans_plus_plus(pixels, k, rng)
    labels = np.full(len(pixels), -1, dtype=np.int32)

    for _ in range(max_iter):
        diff = pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        new_labels = np.argmin(sq_dist, axis=1)

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for i in range(k):
            cluster_pixels = pixels[labels == i]
            if len(cluster_pixels) > 0:
                centroids[i] = np.mean(cluster_pixels, axis=0)

    quantized = centroids[labels].astype(np.uint8).reshape(h, w, c)
    cluster_map = labels.reshape(h, w)
    return quantized, cluster_map


def isolate_foreground_cluster(
    cluster_map: np.ndarray,
    image_rgb: np.ndarray
) -> np.ndarray:
    """Isolate foreground from cluster map using a multi-factor score:
    Score = (Center Bias * 0.40) + (Mean Saturation * 0.45) - (Border Ratio * 0.55)

    The cluster with the lowest score is classified as background; remaining clusters
    form the initial foreground mask.
    """
    h, w = cluster_map.shape

    # 1. Border mask (outer 2% perimeter)
    margin_y = max(1, int(round(h * 0.02)))
    margin_x = max(1, int(round(w * 0.02)))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:margin_y, :] = border_mask[-margin_y:, :] = True
    border_mask[:, :margin_x] = border_mask[:, -margin_x:] = True
    total_border = float(np.sum(border_mask))

    # 2. Center box (central 50% box)
    center_mask = np.zeros((h, w), dtype=bool)
    center_mask[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)] = True
    total_center = float(np.sum(center_mask))

    # 3. Saturation in HSV
    img_hsv = rgb_to_hsv(image_rgb)
    sat = img_hsv[..., 1].astype(float) / 255.0

    unique_clusters = np.unique(cluster_map)
    scores = {}

    for c in unique_clusters:
        c_mask = (cluster_map == c)
        count = np.sum(c_mask)
        if count == 0:
            scores[c] = -999.0
            continue

        border_ratio = np.sum(c_mask & border_mask) / total_border
        center_ratio = np.sum(c_mask & center_mask) / total_center
        mean_sat = float(np.mean(sat[c_mask]))

        # Foreground score
        score = (center_ratio * 0.40) + (mean_sat * 0.45) - (border_ratio * 0.55)
        scores[c] = score

    # Lowest score is background
    bg_cluster = min(scores.keys(), key=lambda k: scores[k])
    fg_mask = (cluster_map != bg_cluster).astype(np.uint8) * 255
    return fg_mask


def filter_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """Retain only the largest connected component in a binary mask."""
    bool_mask = binary_mask > 0
    labeled_array, num_features = ndi.label(bool_mask)
    if num_features <= 1:
        return binary_mask

    sizes = ndi.sum(bool_mask, labeled_array, range(1, num_features + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return (labeled_array == largest_label).astype(np.uint8) * 255


def apply_morphological_cleanup(
    binary_mask: np.ndarray,
    radius: int = 3,
    keep_largest_only: bool = True
) -> np.ndarray:
    """Denoise binary mask using morphological opening, closing, hole filling,
    and largest connected component filtering via scipy.ndimage.
    """
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    structure = (x**2 + y**2 <= radius**2)

    bool_mask = binary_mask > 0

    # 1. Morphological Opening (remove small speckle noise)
    opened = ndi.binary_opening(bool_mask, structure=structure)

    # 2. Morphological Closing (fill small gaps)
    closed = ndi.binary_closing(opened, structure=structure)

    # 3. Fill internal holes
    filled = ndi.binary_fill_holes(closed)

    cleaned = filled.astype(np.uint8) * 255

    # 4. Filter largest connected component
    if keep_largest_only:
        cleaned = filter_largest_component(cleaned)

    return cleaned


def run_pipeline(
    image_rgb: np.ndarray,
    method: str = "kmeans",
    color_space: str = "Lab",
    k: int = 3,
    fruit: str = "apple",
    stage: str = "ripe",
    apply_morphology: bool = True
) -> np.ndarray:
    """Run full segmentation pipeline on an RGB image.
    
    Args:
        image_rgb: Input RGB image (H, W, 3)
        method: "kmeans" or "hsv_thresh"
        color_space: "RGB", "HSV", or "Lab" (for kmeans)
        k: Number of clusters (for kmeans)
        fruit: Fruit name (for hsv_thresh)
        stage: Stage name (for hsv_thresh)
        apply_morphology: Whether to run opening, closing, hole filling, largest component.
    """
    if method == "hsv_thresh":
        raw_mask = segment_by_configured_hsv(image_rgb, fruit, stage)
    elif method == "kmeans":
        cs = color_space.upper()
        if cs == "RGB":
            feat_img = image_rgb
        elif cs == "HSV":
            feat_img = rgb_to_hsv(image_rgb)
        elif cs == "LAB":
            feat_img = rgb_to_lab(image_rgb)
        else:
            raise ValueError(f"Unknown color space: {color_space}")

        _, cluster_map = segment_by_kmeans(feat_img, k=k, random_state=42)
        raw_mask = isolate_foreground_cluster(cluster_map, image_rgb)
    else:
        raise ValueError(f"Unknown method: {method}")

    if apply_morphology:
        return apply_morphological_cleanup(raw_mask)
    return raw_mask


def calculate_mask_iou(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between binary masks."""
    pred = predicted_mask > 0
    gt = ground_truth_mask > 0

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def calculate_mask_dice(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """Calculate Sørensen–Dice similarity coefficient."""
    pred = predicted_mask > 0
    gt = ground_truth_mask > 0

    intersection = np.logical_and(pred, gt).sum()
    total_area = pred.sum() + gt.sum()

    if total_area == 0:
        return 1.0 if intersection == 0 else 0.0

    return float((2.0 * intersection) / total_area)
