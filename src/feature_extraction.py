"""Module 3: Feature Extraction

Pure NumPy + Pillow implementation of:
1. Color moments extraction (Mean, Standard Deviation, Skewness) across RGB, HSV, Lab.
2. Color histograms using np.histogram.
3. Feature aggregation per fruit instance for traditional ML classifiers.
4. Exporting feature tables using the standard csv module.
"""

from typing import Dict, List, Optional
import csv
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.preprocessing import rgb_to_hsv, rgb_to_lab


def calculate_channel_moments(channel: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculate statistical color moments (Mean, Std Dev, Skewness) for a single channel.

    Args:
        channel (np.ndarray): 2D array of pixel values.
        mask (np.ndarray, optional): Binary mask selecting valid fruit pixels.

    Returns:
        dict: Mean, standard deviation, and skewness of the channel.
    """
    valid_pixels = channel[mask > 0] if mask is not None else channel.flatten()

    if len(valid_pixels) == 0:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0}

    mean_val = float(np.mean(valid_pixels))
    std_val = float(np.std(valid_pixels))

    # Skewness calculation: E[((x - mu) / sigma)^3]
    if std_val > 1e-6:
        skew_val = float(np.mean(((valid_pixels - mean_val) / std_val) ** 3))
    else:
        skew_val = 0.0

    return {"mean": mean_val, "std": std_val, "skew": skew_val}


def extract_color_features(image_rgb: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Extract full statistical color feature vector across RGB, HSV, and Lab color spaces.

    Args:
        image_rgb (np.ndarray): Input RGB image.
        mask (np.ndarray, optional): Foreground segmentation mask.

    Returns:
        dict: Flattened dictionary of extracted color features.
    """
    features: Dict[str, float] = {}

    image_hsv = rgb_to_hsv(image_rgb)
    image_lab = rgb_to_lab(image_rgb)

    spaces = {
        "rgb": (image_rgb, ["r", "g", "b"]),
        "hsv": (image_hsv, ["h", "s", "v"]),
        "lab": (image_lab, ["l", "a", "b"])
    }

    for space_name, (img, channel_names) in spaces.items():
        for idx, ch_name in enumerate(channel_names):
            ch_data = img[:, :, idx]
            moments = calculate_channel_moments(ch_data, mask)
            for m_key, m_val in moments.items():
                features[f"{space_name}_{ch_name}_{m_key}"] = m_val

    return features


def extract_color_histogram(
    image_rgb: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bins: int = 16
) -> np.ndarray:
    """Extract normalized 1D color histogram across R, G, and B channels using np.histogram.

    Args:
        image_rgb (np.ndarray): Input RGB image.
        mask (np.ndarray, optional): Foreground mask.
        bins (int): Number of bins per channel. Defaults to 16.

    Returns:
        np.ndarray: Concatenated and normalized histogram feature vector.
    """
    hist_features = []
    valid_pixels = image_rgb[mask > 0] if mask is not None else image_rgb.reshape(-1, 3)

    if len(valid_pixels) == 0:
        return np.zeros(bins * 3, dtype=np.float32)

    for i in range(3):
        channel_data = valid_pixels[:, i]
        counts, _ = np.histogram(channel_data, bins=bins, range=(0, 256))
        total = np.sum(counts)
        norm_counts = (counts / total).astype(np.float32) if total > 0 else counts.astype(np.float32)
        hist_features.extend(norm_counts)

    return np.array(hist_features, dtype=np.float32)


def save_feature_table_to_csv(records: List[Dict[str, any]], filepath: str) -> None:
    """Save feature records to CSV file using standard library csv module.

    Args:
        records (list): List of dictionaries containing features and labels.
        filepath (str): Output CSV file path.
    """
    if not records:
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = list(records[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
