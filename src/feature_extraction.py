"""Module 3: Feature Extraction

This module handles:
1. Color moments extraction (Mean, Standard Deviation, Skewness) across RGB, HSV, Lab.
2. Color histograms.
3. Feature aggregation per fruit instance for traditional ML classifiers.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import cv2


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

    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

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
    """Extract normalized 1D color histogram across R, G, and B channels.

    Args:
        image_rgb (np.ndarray): Input RGB image.
        mask (np.ndarray, optional): Foreground mask.
        bins (int): Number of bins per channel. Defaults to 16.

    Returns:
        np.ndarray: Concatenated and normalized histogram feature vector.
    """
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([image_rgb], [i], mask, [bins], [0, 256])
        cv2.normalize(hist, hist)
        hist_features.extend(hist.flatten())
    return np.array(hist_features, dtype=np.float32)


def create_feature_table(records: List[Dict[str, any]]) -> pd.DataFrame:
    """Convert list of feature records into a structured pandas DataFrame.

    Args:
        records (list): List of dictionaries containing features and labels.

    Returns:
        pd.DataFrame: Formatted DataFrame ready for machine learning.
    """
    return pd.DataFrame(records)
