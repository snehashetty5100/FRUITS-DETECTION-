"""Sanity check script for Fruits-360 images."""

from pathlib import Path
import numpy as np
from PIL import Image

fruits360_dir = Path("dataset/fruits360/Training")

print(f"{'File':<48} | {'Orig Size':<12} | {'Aspect':<8} | {'Mean RGB (R, G, B)'}")
print("-" * 95)

records = []
if fruits360_dir.exists():
    for folder in sorted(fruits360_dir.iterdir()):
        if not folder.is_dir():
            continue
        for img_path in sorted(folder.glob("*.jpg"))[:3]:  # sample 3 per class
            with Image.open(img_path) as img:
                w, h = img.size
                aspect = w / h
                arr = np.array(img.convert("RGB"))
                mean_r, mean_g, mean_b = arr.mean(axis=(0, 1))
                rel_path = str(img_path.relative_to(fruits360_dir))
                print(f"{rel_path:<48} | {f'{w}x{h}':<12} | {aspect:<8.2f} | ({mean_r:5.1f}, {mean_g:5.1f}, {mean_b:5.1f})")
                records.append({
                    "path": img_path,
                    "rel_path": rel_path,
                    "width": w,
                    "height": h,
                    "aspect": aspect,
                    "rgb": (mean_r, mean_g, mean_b)
                })

print("-" * 95)
print(f"Total sampled: {len(records)}")
