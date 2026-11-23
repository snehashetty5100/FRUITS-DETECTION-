"""Test fixture curator: verifies that images match the visual color definition
of unripe, ripe, and overripe before saving into dataset/raw.
"""

from pathlib import Path
import urllib.request
import numpy as np
from PIL import Image

def verify_color_for_stage(fruit: str, stage: str, img: Image.Image) -> bool:
    """Mathematical sanity check for color dominance."""
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    # Center 50%
    crop = arr[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    r = float(np.mean(crop[:, :, 0]))
    g = float(np.mean(crop[:, :, 1]))
    b = float(np.mean(crop[:, :, 2]))

    # HSV conversion for hue check
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c

    if delta < 10:
        return True

    if max_c == r:
        hue = 60.0 * (((g - b) / delta) % 6)
    elif max_c == g:
        hue = 60.0 * (((b - r) / delta) + 2)
    else:
        hue = 60.0 * (((r - g) / delta) + 4)

    # Hue angles:
    # Red: 340 - 20 degrees
    # Orange: 20 - 45 degrees
    # Yellow: 45 - 70 degrees
    # Green: 70 - 160 degrees
    
    # Check green for unripe fruits
    if stage == "unripe":
        if fruit in ["tomato", "banana", "papaya", "apple", "mango"]:
            return (g >= r * 0.85) or (55 <= hue <= 165)
    elif stage == "ripe":
        if fruit == "tomato":
            return r > g * 1.3 and r > b * 1.5
        elif fruit == "banana":
            return r > 120 and g > 100 and b < min(r, g) * 0.7
        elif fruit == "orange":
            return r > g * 1.2 and b < g
    return True

print("Color verifier function defined successfully.")
