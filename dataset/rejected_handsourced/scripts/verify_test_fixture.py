"""Visual and Color Verification of Test Fixture Images."""

from pathlib import Path
import numpy as np
from PIL import Image

def analyze_image_colors(img_path: Path):
    with Image.open(img_path) as img:
        arr = np.array(img.convert("RGB"))
        r = arr[:, :, 0].astype(float)
        g = arr[:, :, 1].astype(float)
        b = arr[:, :, 2].astype(float)

        mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)

        # Simple color dominance heuristic
        # Green dominance: G significantly higher than R (e.g. unripe banana, unripe tomato)
        # Red dominance: R significantly higher than G and B (e.g. ripe tomato, ripe apple)
        # Yellow dominance: Both R and G high, B low (e.g. ripe banana, ripe mango)
        return {
            "size": img.size,
            "mean_r": round(mean_r, 1),
            "mean_g": round(mean_g, 1),
            "mean_b": round(mean_b, 1),
        }

if __name__ == "__main__":
    raw_dir = Path("dataset/raw")
    for f in sorted(raw_dir.glob("*/*/*.jpg")):
        stats = analyze_image_colors(f)
        rel = f.relative_to(raw_dir)
        print(f"{str(rel):<45} | Size: {stats['size']} | RGB: ({stats['mean_r']}, {stats['mean_g']}, {stats['mean_b']})")
