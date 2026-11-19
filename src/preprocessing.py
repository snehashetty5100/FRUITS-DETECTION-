"""Module 1: Preprocessing

This module handles:
1. Loading images from disk.
2. Resizing images to a standard resolution.
3. Converting images between color spaces (RGB, HSV, Lab).
4. Pixel intensity normalization.
"""

from typing import Tuple, Dict
import numpy as np
import cv2


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk and convert it to RGB format.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Image array in RGB format.
    """
    # OpenCV loads images in BGR format by default
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found or unable to read: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize image to standard dimensions for consistent processing.

    Args:
        image (np.ndarray): Input RGB image.
        target_size (tuple): Target (width, height). Defaults to (224, 224).

    Returns:
        np.ndarray: Resized image.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def convert_color_spaces(image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert an RGB image into HSV and CIE-Lab color spaces.

    Args:
        image_rgb (np.ndarray): Input image in RGB format.

    Returns:
        dict: Dictionary containing 'RGB', 'HSV', and 'Lab' representations.
    """
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    return {
        "RGB": image_rgb,
        "HSV": image_hsv,
        "Lab": image_lab
    }


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values from [0, 255] to [0.0, 1.0].

    Args:
        image (np.ndarray): Input image with values in range [0, 255].

    Returns:
        np.ndarray: Normalized image as float32 in range [0.0, 1.0].
    """
    return image.astype(np.float32) / 255.0
