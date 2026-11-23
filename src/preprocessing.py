"""Module 1: Preprocessing

Pillow + NumPy implementation of:
1. Loading images from disk and ensuring 3-channel RGB format (compositing RGBA onto white).
2. Direct image resizing (without letterbox padding; images are natively 300x300).
3. RGB to HSV color space conversion (pure NumPy, 8-bit scale: H: 0-179, S: 0-255, V: 0-255).
4. RGB to CIE-Lab color space conversion (pure NumPy, CIE 1931 D65, 8-bit scale: L: 0-255, a: 0-255, b: 0-255).
5. Intensity normalization.
"""

from typing import Tuple, Dict, Union
import numpy as np
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk and return as a uint8 RGB numpy array (H, W, 3).
    
    If image has an alpha channel, composites onto a solid white background.
    """
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA") or ("A" in img.mode):
            alpha = img.split()[-1]
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img.convert("RGB"), mask=alpha)
            return np.array(bg, dtype=np.uint8)
        elif img.mode != "RGB":
            return np.array(img.convert("RGB"), dtype=np.uint8)
        return np.array(img, dtype=np.uint8)


def resize_image(
    image: Union[np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (300, 300)
) -> np.ndarray:
    """Directly resize an image to target (width, height) without letterbox padding."""
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = image

    resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(resized, dtype=np.uint8)


def rgb_to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image array to HSV color space using pure NumPy.

    Matches standard 8-bit scale:
    - Hue: [0, 179]
    - Saturation: [0, 255]
    - Value: [0, 255]
    """
    arr = image_rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue calculation
    h = np.zeros_like(cmax)
    mask_r = (cmax == r) & (delta > 1e-6)
    mask_g = (cmax == g) & (delta > 1e-6)
    mask_b = (cmax == b) & (delta > 1e-6)

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    # Saturation calculation
    s = np.zeros_like(cmax)
    mask_s = cmax > 1e-6
    s[mask_s] = delta[mask_s] / cmax[mask_s]

    # Value calculation
    v = cmax

    # Quantize to 8-bit ranges
    h_8u = np.clip(h / 2.0, 0, 179).astype(np.uint8)
    s_8u = np.clip(s * 255.0, 0, 255).astype(np.uint8)
    v_8u = np.clip(v * 255.0, 0, 255).astype(np.uint8)

    return np.stack([h_8u, s_8u, v_8u], axis=-1)


def rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image array to CIE-Lab color space using standard CIE 1931 D65.

    Matches standard 8-bit scale:
    - L: [0, 255] (scaled from 0..100)
    - a: [0, 255] (shifted by +128)
    - b: [0, 255] (shifted by +128)
    """
    arr = image_rgb.astype(np.float32) / 255.0

    # sRGB linearization
    mask = arr > 0.04045
    arr[mask] = ((arr[mask] + 0.055) / 1.055) ** 2.4
    arr[~mask] = arr[~mask] / 12.92

    # Linear sRGB to XYZ (D65 illuminant matrix)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Normalize by D65 reference white
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    x = x / xn
    y = y / yn
    z = z / zn

    def f_transform(t):
        t_mask = t > 0.008856
        res = np.empty_like(t)
        res[t_mask] = t[t_mask] ** (1.0 / 3.0)
        res[~t_mask] = (7.787 * t[~t_mask]) + (16.0 / 116.0)
        return res

    fx = f_transform(x)
    fy = f_transform(y)
    fz = f_transform(z)

    l_val = 116.0 * fy - 16.0
    a_val = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)

    # Quantize to 8-bit representation
    l_8u = np.clip(l_val * 255.0 / 100.0, 0, 255).astype(np.uint8)
    a_8u = np.clip(a_val + 128.0, 0, 255).astype(np.uint8)
    b_8u = np.clip(b_val + 128.0, 0, 255).astype(np.uint8)

    return np.stack([l_8u, a_8u, b_8u], axis=-1)


def convert_color_spaces(image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert RGB image into RGB, HSV, and CIE-Lab representations."""
    return {
        "RGB": image_rgb,
        "HSV": rgb_to_hsv(image_rgb),
        "Lab": rgb_to_lab(image_rgb),
    }


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel intensities to float32 range [0.0, 1.0]."""
    return image.astype(np.float32) / 255.0
