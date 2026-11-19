"""Module 2: Segmentation

This module handles:
1. Color-based thresholding in HSV/Lab spaces.
2. K-means clustering for color-based background/foreground separation.
3. Morphological operations (opening/closing) for noise removal.
4. Foreground extraction from binary masks.
5. Intersection over Union (IoU) calculation when ground-truth masks are available.
"""

from typing import Tuple
import numpy as np
import cv2


def segment_by_color_threshold(
    image_hsv: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray
) -> np.ndarray:
    """Create a binary mask isolating pixels within specific HSV thresholds.

    Args:
        image_hsv (np.ndarray): Image in HSV color space.
        lower_bound (np.ndarray): Lower bound array [H, S, V].
        upper_bound (np.ndarray): Upper bound array [H, S, V].

    Returns:
        np.ndarray: Binary mask (0 or 255) of selected regions.
    """
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    return mask


def segment_by_kmeans(image_rgb: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Segment image using K-Means color clustering.

    Args:
        image_rgb (np.ndarray): Input RGB image.
        k (int): Number of color clusters. Defaults to 3.

    Returns:
        tuple: (segmented_image, cluster_labels)
    """
    # Reshape image into a 2D array of pixels (N, 3)
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define stopping criteria for K-Means (100 iterations or epsilon = 0.2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert cluster centers back to 8-bit values and reconstruct image
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image_rgb.shape)

    return segmented_image, labels.reshape(image_rgb.shape[:2])


def apply_morphological_cleanup(
    binary_mask: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5)
) -> np.ndarray:
    """Remove noise and fill small holes using morphological opening and closing.

    Args:
        binary_mask (np.ndarray): Binary input mask (0 and 255).
        kernel_size (tuple): Structuring element dimensions. Defaults to (5, 5).

    Returns:
        np.ndarray: Cleaned binary mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    # Opening removes small noise spots
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # Closing bridges small gaps and fills small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def extract_segmented_foreground(
    image_rgb: np.ndarray,
    binary_mask: np.ndarray
) -> np.ndarray:
    """Extract the foreground fruit object by masking out the background.

    Args:
        image_rgb (np.ndarray): Original RGB image.
        binary_mask (np.ndarray): Single-channel binary mask.

    Returns:
        np.ndarray: RGB image with background masked to black.
    """
    return cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)


def calculate_mask_iou(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray
) -> float:
    """Calculate Intersection over Union (IoU) between predicted and ground-truth masks.

    Args:
        predicted_mask (np.ndarray): Predicted binary mask (0 or 255).
        ground_truth_mask (np.ndarray): Ground truth binary mask (0 or 255).

    Returns:
        float: IoU score between 0.0 and 1.0.
    """
    pred = (predicted_mask > 0).astype(bool)
    gt = (ground_truth_mask > 0).astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)
